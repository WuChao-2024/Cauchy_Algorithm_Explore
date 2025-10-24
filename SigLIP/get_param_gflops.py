'''
Date: 2025-10-22
Author: Cauchy-WuChao
Description: 获取SigLIP模型Vision部分的参数量
pip install fvcore
'''

# Download ImageNet-1k DataSet: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/4603483700ee984ea9debe3ddbfdeae86f6489eb
from transformers import SiglipModel, SiglipProcessor, SiglipTokenizer, SiglipVisionModel
import os
import numpy as np
import argparse
from tqdm import tqdm
from scipy.special import expit as sigmoid
import torch
import cv2
from fvcore.nn import FlopCountAnalysis, parameter_count
# import matplotlib.pyplot as plt

def main():
    result_str = ""
    names = ["siglip-base-patch16-224",
             "siglip-base-patch16-384",
             "siglip-base-patch16-512",
             "siglip-large-patch16-256",
             "siglip-large-patch16-384",
             "siglip-so400m-patch14-224",
             "siglip-so400m-patch14-384",
             "siglip-so400m-patch16-256-i18n"]
    for s in names:
        device = torch.device("cpu")
        model = SiglipVisionModel.from_pretrained(s).to(device)

        input_tensor = preprocess(cv2.imread("/home/users/chao01.wu/SigLIP_Leap/raw_imgs/000000121591.jpg"), model.config.image_size)
        model.eval()
        param = c(model)
        # param = sum(parameter_count(model).values()) / 1e9

        # floper = FlopCountAnalysis(model, input_tensor)
        # flops_g = floper.total() / 1e9

        # print(f"{s}: {param:.4f} B, {flops_g:.4f} G")
        result_str += f"{s}: {param:.4f} B\n"
    print("\n\n")
    print(result_str)

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

def c(model):
    def fps(x):
        return 1/(0.0008583*x + 0.0035866)
    # 计算所有参数的数量
    total_params = sum(p.numel() for p in model.parameters())
    para = total_params / 1e9
    return para
    print(f"参数量: {para:.2f} M, {model.__module__} ")

if __name__ == "__main__":
    main()




# siglip-so400m-patch14-384 (FP32)
# TOP1: 78.72 %
# TOP5: 94.33 %


# siglip-so400m-patch14-384 (BPU, cos=0.67)
# S100 inference time: 340.074 ms
# TOP1: 62.31 %  (79.15 %)
# TOP5: 85.14 %  (90.26 %)


# siglip-so400m-patch14-384 (VPU优化BPU, cos=0.995)
# S100 inference time: 256.134 ms
# TOP1: 78.93 %  (100.00 %)
# TOP5: 94.47 %  (100.00 %)

# siglip-so400m-patch14-224 (FP32)
# TOP1: 76.59 %
# TOP1: 93.61 %

# siglip-base-patch16-224
# TOP1: 71.23 %
# TOP1: 91.43 %

# siglip-large-patch16-384 (FP32)
# TOP1: 75.84 %
# TOP1: 92.52 %