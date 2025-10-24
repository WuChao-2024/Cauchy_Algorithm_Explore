# Python Librarys
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Any, Optional, Tuple, Union

# Third Party Librarys
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers import AutoModelForCausalLM

# HBDK and leap tools
from hbdk4.compiler import leap, save, statistics
from leap_llm.nn.utils import Model, timeit  # noqa: E402
from leap_llm.nn.utils import Module
from dataclasses import dataclass, asdict
from leap_llm.nn.modules import FakeQuantAdd, ConstFakeQuant, FakeQuantLinear, FakeQuantMatmul, FakeQuantMul, FakeQuantSoftmax, LayerNorm, LayerNormSplit

LS_WORKSPACE_NAME = "build"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-path', type=str, default="raw_imgs", help="")
    ''' for example: 
    raw_imgs/
    ├── 000000121031.jpg
    ...
    ├── 000000125936.jpg
    └── 000000125952.jpg
    '''
    parser.add_argument('--export-path', type=str, default="bpu-siglip-so400m-patch14-224", help="")
    parser.add_argument('--export-name', type=str, default="bpu-siglip-so400m-patch14-224.hbm", help="")
    parser.add_argument('--siglip-weight-path', type=str, default="siglip-so400m-patch14-224", help="")
    ''' for example: 
    siglip-so400m-patch14-384/
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── README.md
    ├── special_tokens_map.json
    ├── spiece.model
    ├── tokenizer_config.json
    └── tokenizer.json
    '''
    parser.add_argument('--device', type=str, default="cuda:0", help="cpu / cuda / cuda:0")
    parser.add_argument('--optimized-level', type=int, default=2, help="0: O0; 1: O1; 2: O2;")
    parser.add_argument('--march', type=str, default="nash-m", help="nash-e / nash-m")
    parser.add_argument('--jobs', type=int, default=64, help="")
    opt = parser.parse_args()

    # Create WorkSpace
    device = torch.device("cpu")
    os.makedirs(os.path.join(opt.export_path, LS_WORKSPACE_NAME), exist_ok=True)

    # Convert transformers.model.siglip to pth
    m = SiglipModel.from_pretrained(opt.siglip_weight_path).to(torch.device("cpu"))
    torch.save(m.state_dict(), os.path.join(opt.export_path, LS_WORKSPACE_NAME, 'model.pth'))
    print(f"pth saved to: {os.path.join(opt.export_path, LS_WORKSPACE_NAME, 'model.pth')}.")
    configs = SiglipVisionConfig
    configs.attention_dropout = m.config.vision_config.attention_dropout
    configs.hidden_size = m.config.vision_config.hidden_size
    configs.image_size = m.config.vision_config.image_size
    configs.intermediate_size = m.config.vision_config.intermediate_size
    configs.layer_norm_eps = m.config.vision_config.layer_norm_eps
    configs.num_attention_heads = m.config.vision_config.num_attention_heads
    configs.num_channels = m.config.vision_config.num_channels
    configs.num_hidden_layers = m.config.vision_config.num_hidden_layers
    configs.patch_size = m.config.vision_config.patch_size
    del(m)

    # load pth to leap.Model
    model = SiglipVisionModel(configs)
    pth = torch.load(os.path.join(opt.export_path, LS_WORKSPACE_NAME, 'model.pth'), map_location="cpu")
    pth['vision_model.head.attention.q_proj.weight'] = pth['vision_model.head.attention.in_proj_weight'][0*configs.hidden_size:1*configs.hidden_size,:].detach()
    pth['vision_model.head.attention.q_proj.bias']   = pth['vision_model.head.attention.in_proj_bias'][0*configs.hidden_size:1*configs.hidden_size].detach()
    pth['vision_model.head.attention.k_proj.weight'] = pth['vision_model.head.attention.in_proj_weight'][1*configs.hidden_size:2*configs.hidden_size,:].detach()
    pth['vision_model.head.attention.k_proj.bias']   = pth['vision_model.head.attention.in_proj_bias'][1*configs.hidden_size:2*configs.hidden_size].detach()
    pth['vision_model.head.attention.v_proj.weight'] = pth['vision_model.head.attention.in_proj_weight'][2*configs.hidden_size:3*configs.hidden_size,:].detach()
    pth['vision_model.head.attention.v_proj.bias']   = pth['vision_model.head.attention.in_proj_bias'][2*configs.hidden_size:3*configs.hidden_size].detach()
    model.load_state_dict(pth, strict=False)

    # calibrations
    model.compile_mode(False)
    model = model.to(device)
    model = model.eval()
    names = os.listdir(opt.raw_data_path)
    for cnt in tqdm(range(len(names)), desc="Calibration", unit=" sample"):
        with torch.no_grad():
            model.forward(
                preprocess(
                    cv2.imread(os.path.join(opt.raw_data_path, names[cnt])), 
                    configs.image_size
                )
            )
    
    # combine
    model.compile_mode(True)
    model.to("cpu")
    model.compile(output_model_path=opt.export_path, 
                  hbm_model_name=opt.export_name,
                  march=opt.march,
                  opt=opt.optimized_level,
                  jobs=opt.jobs
                  )


def preprocess(image, target_size=384):
    # HWC, BGR, 0~255 -> (1, 3, 384, 384), RGB, -1~1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    image_padded = cv2.copyMakeBorder(
        image_resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[127, 127, 127]  # 灰色边框
    )
    # 4. HWC -> CHW -> NCHW
    image_chw = np.transpose(image_padded, (2, 0, 1))  # HWC -> CHW
    image_nchw = np.expand_dims(image_chw, axis=0)     # CHW -> NCHW (batch=1)
    # 5. 归一化: / 127.5 - 1 → [-1, 1]
    image_normalized = image_nchw.astype(np.float32)
    image_normalized = image_normalized / 127.5 - 1.0
    tensor = torch.from_numpy(image_normalized)  # shape: (1, 3, 384, 384)
    return tensor

@dataclass
class SiglipVisionConfig:
    attention_dropout: float = 0.0
    hidden_size: int = 1152
    image_size: int = 384
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-06
    num_attention_heads: int = 16
    num_channels: int = 3
    num_batch: int = 1
    num_hidden_layers: int = 27
    patch_size: int = 14
    # default_siglip_name: str = "siglip-so400m-patch14-384"
    # hidden_act: str = "gelu_pytorch_tanh"
    # model_type: str = "siglip_vision_model"
    # transformers_version: str = "4.41.0"

@dataclass
class CompileArgs:
    name: str = "SiglipVision"  # function name for compile


class FakeQuantGeluPytorchTanh(Module):
    def __init__(
        self,
        quantized: bool = True,
        quant_bits: int = 16,
    ) -> None:
        super().__init__()
        self.quant_bits = quant_bits
        self.out_quant = ConstFakeQuant(quant_bits)
        self.act_fn = lambda x: torch.nn.functional.gelu(x, approximate='tanh')
        self.quantized = quantized
    def build(self, x):
        if self.quantized:
            out = leap.gelu(x, approximate='tanh')
            out = self.out_quant(out)
            return out
        return leap.gelu(x, approximate='tanh')
    def forward(self, x):
        if self.quantized:
            out = self.act_fn(x)
            out = self.out_quant(out)
            return out
        return self.act_fn(x)
    

class PatchEmbedding(Module):
    def __init__(self, embed_dim, num_channels, patch_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_channels, patch_size,
                patch_size)
        )
        self.num_channels = num_channels
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.patch_size = patch_size
        self.x_fake_quant = ConstFakeQuant(8)
        self.absmax_weight = None
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.x_fake_quant(x)
        return F.conv2d(input=x, weight=self.weight, bias=self.bias,
                stride=(self.patch_size, self.patch_size))

    def build(self, x):
        x_cast = leap.cast_type(x, output_type=leap.float32)
        x_quant = self.x_fake_quant(x_cast)
        if self.absmax_weight is None:
            last_dim = self.patch_size * self.patch_size * self.num_channels
            weight = torch.reshape(self.weight.data, [self.embed_dim, last_dim])
            per_channel_max, _ = torch.max(weight.abs(), dim=1)
            self.absmax_weight = per_channel_max
        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()
        # NCHW->NHWC
        weight = self.weight.data.permute(0, 2, 3, 1).contiguous()
        weight_quant = leap.const_fake_quant(
            weight,
            weight_min,
            weight_max,
            8,
            True,
            axis=0,
        )
        conv_res = leap.conv2d(
            input=x_quant,
            weight=weight_quant,
            bias=self.bias.data,
            stride=(self.patch_size, self.patch_size),
        )
        return conv_res

class position_embedding(Module):
    def __init__(self, num_positions, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_positions, embed_dim))
        self.num_positions = num_positions
        self.embed_dim = embed_dim

    def forward(self, x):
        return F.embedding(x, self.weight)

    def build(self, x):
        # print("pos_ids", self.weight.data.shape)
        return leap.reshape(self.weight.data, shape=[1, self.num_positions, self.embed_dim])

class SiglipVisionEmbeddings(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = PatchEmbedding(self.embed_dim, config.num_channels, self.patch_size)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = position_embedding(self.num_positions, self.embed_dim)
        self.position_ids = torch.arange(self.num_positions).unsqueeze(0).repeat(1, 1).contiguous()

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding=False) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def build(self, pixel_values: torch.Tensor, interpolate_pos_encoding=False) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds
        embeddings = leap.reshape(patch_embeds, shape=[self.config.num_batch, self.num_positions, self.embed_dim])
        embeddings = leap.add(embeddings, self.position_embedding(self.position_ids))
        return embeddings


class SiglipAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout


        linear_quant_bits = 16  # 设置为8没有性能提升
        self.k_proj = FakeQuantLinear(
            self.embed_dim, self.embed_dim, bias=True, quant_bits=linear_quant_bits
        )
        self.v_proj = FakeQuantLinear(
            self.embed_dim, self.embed_dim, bias=True, quant_bits=linear_quant_bits
        )
        self.q_proj = FakeQuantLinear(
            self.embed_dim, self.embed_dim, bias=True, quant_bits=linear_quant_bits
        )
        self.out_proj = FakeQuantLinear(
            self.embed_dim, self.embed_dim, bias=True, quant_bits=linear_quant_bits
        )
        q_quant_bits = 8
        # self.qk = FakeQuantMatmul(q_quant_bits, 16)
        self.qk = FakeQuantMatmul(q_quant_bits, 8)
        v_quant_bits = 8
        self.sv = FakeQuantMatmul(None, v_quant_bits)

        self.mul_attn_weight = FakeQuantMul(quantized=False)
        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(
            quant_bits=softmax_out_quant_bits, quantized=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = self.qk(query_states, key_states.transpose(2, 3)) * self.scale

        # upcast attention to fp32
        attn_weights = self.softmax(attn_weights)
        attn_output = self.sv(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
    
    def build(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """Input shape: Batch x Time x Channel"""

        batch_size = hidden_states.type.shape[0]
        q_len = hidden_states.type.shape[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = leap.reshape(query_states, [batch_size, q_len, self.num_heads, self.head_dim])
        query_states = leap.transpose(query_states, [0, 2, 1, 3])
        key_states = leap.reshape(key_states, [batch_size, q_len, self.num_heads, self.head_dim])
        key_states = leap.transpose(key_states, [0, 2, 3, 1])
        value_states = leap.reshape(value_states, [batch_size, q_len, self.num_heads, self.head_dim])
        value_states = leap.transpose(value_states, [0, 2, 1, 3])

        attn_weights = self.qk(query_states, key_states)
        attn_weights = leap.mul(attn_weights, self.scale)

        # upcast attention to fp32
        attn_weights = self.softmax(attn_weights)
        attn_output = self.sv(attn_weights, value_states)
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [batch_size, q_len, self.embed_dim])
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = FakeQuantGeluPytorchTanh(True, 16)
        self.fc1 = FakeQuantLinear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.fc2 = FakeQuantLinear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    def build(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->Siglip
class SiglipEncoderLayer(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = LayerNormSplit(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = LayerNormSplit(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs
    
    def build(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        hidden_states = leap.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        outputs = (hidden_states,)

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->Siglip
class SiglipEncoder(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        return hidden_states
    
    def build(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        return hidden_states

class SiglipVisionTransformer(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        embed_dim = SiglipVisionConfig.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = LayerNormSplit(embed_dim, eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(config)

        self.stage = "pooler_output"  # "last_hidden_state"

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) :
        hidden_states =  self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooler = self.head(last_hidden_state)
        if self.stage == "pooler_output":
            return pooler
        if self.stage == "last_hidden_state":
            return last_hidden_state
        raise "state error."
    
    def build(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) :
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooler = self.head(last_hidden_state)
        if self.stage == "pooler_output":
            return pooler
        if self.stage == "last_hidden_state":
            return last_hidden_state
        raise "state error."


# test
# in_proj_weight: torch.Size([3456, 1152])  # 1152 * 3 = 3546
# in_proj_bias: torch.Size([3456])
# out_proj.weight: torch.Size([1152, 1152])
# out_proj.bias: torch.Size([1152])
class FakeQuantMultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.embed_dim = embed_dim # 1152
        self.num_heads = num_heads # 16
        self.head_dim = self.embed_dim // self.num_heads  # 72
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = 1 / torch.sqrt(torch.tensor(self.head_dim))
        self.q_proj = FakeQuantLinear(embed_dim, embed_dim, bias=True, quant_bits=16)
        self.k_proj = FakeQuantLinear(embed_dim, embed_dim, bias=True, quant_bits=16)
        self.v_proj = FakeQuantLinear(embed_dim, embed_dim, bias=True, quant_bits=16)
        self.out_proj = FakeQuantLinear(embed_dim, embed_dim, bias=True, quant_bits=16)
        self.qk = FakeQuantMatmul(x_bits=8, y_bits=16, out_bits=16)
        self.sv = FakeQuantMatmul(x_bits=16, y_bits=8, out_bits=16)
        self.mul = FakeQuantMul(quantized=True)
        self.softmax = FakeQuantSoftmax(quant_bits=16, quantized=True)

        
    def forward(self, q, k, v):
        # q: (1, 1, 1152), k, v: (1, 729, 1152)
        B, T_q, D = q.shape
        _, T_kv, _ = k.shape
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        q = self.q_proj(q).view(T_q, self.num_heads, self.head_dim).permute(1, 0, 2)
        q = self.mul(q, self.scale)
        k = self.k_proj(k).view(T_kv, self.num_heads, self.head_dim).permute(1, 2, 0)
        v = self.v_proj(v).view(T_kv, self.num_heads, self.head_dim).permute(1, 0, 2)
        s = self.softmax(self.qk(q, k))
        o = self.sv(s, v).view(1, 1, self.embed_dim)
        o = self.out_proj(o)
        return o

    def build(self, q, k, v):
        # q: (1, 1, 1152), k, v: (1, 729, 1152)
        B, T_q, D = q.shape
        _, T_kv, _ = k.type.shape
        q = q.permute(1,0,2)
        k = leap.transpose(k, [1,0,2])
        v = leap.transpose(v, [1,0,2])
        q = leap.reshape(self.q_proj(q), [T_q, self.num_heads, self.head_dim])
        q = leap.transpose(q, [1, 0, 2])
        q = self.mul(q, self.scale)
        k = leap.reshape(self.k_proj(k), [T_kv, self.num_heads, self.head_dim])
        k = leap.transpose(k, [1, 2, 0])
        v = leap.reshape(self.v_proj(v), [T_kv, self.num_heads, self.head_dim])
        v = leap.transpose(v, [1, 0, 2])
        s = self.softmax(self.qk(q, k))
        o = leap.reshape(self.sv(s, v), [1, 1, self.embed_dim])
        o = self.out_proj(o)
        return o

class SiglipMultiheadAttentionPoolingHead(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = FakeQuantMultiheadAttention(config.hidden_size, config.num_attention_heads)
        self.layernorm = LayerNormSplit(config.hidden_size, eps=config.layer_norm_eps) # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        hidden_state = self.attention(self.probe, hidden_state, hidden_state)  # (1, 1, 1152)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state


    def build(self, hidden_state):
        hidden_state = self.attention(self.probe, hidden_state, hidden_state)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = leap.add(residual, self.mlp(hidden_state))
        return hidden_state    

class SiglipVisionModel(Model):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SiglipVisionTransformer"]

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        return_dict = None

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def build(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        pixel_values = leap.transpose(pixel_values, dims=[0, 2, 3, 1])
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def compile(
        self,
        output_model_path: str = "bpu_output",
        hbm_model_name: str = "siglip.hbm",
        march: str = "nash-m",
        opt: int = 2,
        jobs: int = 32,
        advice: int = 0,
    ):
        assert self.is_compiled, "Model must be compiled before compiling."
        input_types = [leap.TensorType([self.config.num_batch, self.config.num_channels, self.config.image_size, self.config.image_size], leap.float32)]

        bc_model_list = []
        stages = ['pooler_output', 'last_hidden_state']
        for stage in stages:
            self.vision_model.stage = stage
            print(f"export_module: {stage}")
            bc_module = self.export_module(input_types, 
                                           stage, 
                                           os.path.join(output_model_path, LS_WORKSPACE_NAME, f"{stage}.bc")
                                           )
            bc_model_list.append(bc_module)

        hbos = []
        for bc_module, stage in zip(bc_model_list, stages):
            mlir_module = self.convert_mlir(bc_module, 
                                            os.path.join(output_model_path, LS_WORKSPACE_NAME, f"{stage}.bc"), 
                                            enable_vpu=True,
                                            march=march
                                            )
            statistics(mlir_module)
            hbo_model = self.compile_hbo(
                mlir_module,
                os.path.join(output_model_path, LS_WORKSPACE_NAME, f"{stage}.hbo"),
                march=march,
                opt=opt,   # 0: O0, 2: O2
                jobs = jobs,
                advice = advice,
                debug = True,
                progress_bar = True,
                input_no_padding = True,
                output_no_padding = True,
            )
            hbos.append(hbo_model)
        hbm = self.link_models(hbos,
                               os.path.join(output_model_path, hbm_model_name)
                               )
        return hbm

if __name__ == "__main__":
    main()
