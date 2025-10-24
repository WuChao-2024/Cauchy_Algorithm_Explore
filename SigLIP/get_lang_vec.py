'''
Date: 2025-10-22
Author: Cauchy-WuChao
Description: 
  01: 读取ImageNet-1k的50,000张图片, 使用SigLIP生成图像嵌入
* 02: 读取ImageNet-1k的标签, 使用SigLIP生成词嵌入
  03: 比较词嵌入的cos, 的到TOP1和TOP5的acc
  04: 通过hbm模型生成图像嵌入, 重复03, 获取对应的TOP1和TOP5的acc
'''

# Download ImageNet-1k DataSet: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/4603483700ee984ea9debe3ddbfdeae86f6489eb
from classes import IMAGENET2012_CLASSES

from transformers import SiglipModel, SiglipTokenizer
import torch, os
import numpy as np
import argparse

from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cpu", help="cpu / cuda / cuda:1")
    s = "siglip-base-patch16-512"
    parser.add_argument('--text-vec-path', type=str, default=f"ImageNet_TextVec_{s}.npz", help="")
    parser.add_argument('--siglip-weight-path', type=str, default=f"{s}", help="")
    # parser.add_argument('--text-vec-path', type=str, default="ImageNet_TextVec_siglip-large-patch16-256.npz", help="")
    # parser.add_argument('--siglip-weight-path', type=str, default="siglip-large-patch16-256", help="")
    ''' for example: 
    siglip-so400m-patch14-224: https://huggingface.co/google/siglip-so400m-patch14-224
    siglip-so400m-patch14-384: https://huggingface.co/google/siglip-so400m-patch14-384
    .
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── README.md
    ├── special_tokens_map.json
    ├── spiece.model
    ├── tokenizer_config.json
    └── tokenizer.json

    0 directories, 8 files
    '''
    opt = parser.parse_args()
    device = torch.device(opt.device)
    tk = SiglipTokenizer.from_pretrained(opt.siglip_weight_path)
    model = SiglipModel.from_pretrained(opt.siglip_weight_path).to(device)
    print(f"Success load SigLIP at {opt.siglip_weight_path}")
    logit_scale = model.logit_scale.detach().cpu().numpy()
    logit_bias = model.logit_bias.detach().cpu().numpy()
    texts = []
    for key in IMAGENET2012_CLASSES.keys():
        texts.append(f"This is a photo of {IMAGENET2012_CLASSES[key]}.")  # 64
        # texts.append(f"{IMAGENET2012_CLASSES[key]}")  # 16
    tokens = tk(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=None)['input_ids'].to(device) # (bs, 64)
    text_emb = model.get_text_features(tokens).detach().numpy() # (bs, 1152)
    text_emb = text_emb / np.linalg.norm(text_emb, ord=2, axis=-1, keepdims=True)  # (M, 1152)
    np.savez(
        opt.text_vec_path, 
        text_emb=text_emb,
        logit_scale=logit_scale,
        logit_bias=logit_bias
        )
    print(f"{text_emb.shape = }, save at {opt.text_vec_path}")

if __name__ == "__main__":
    main()

exit()

def softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

device = torch.device("cpu")
siglip_path = "siglip-so400m-patch14-224"

tk = SiglipTokenizer.from_pretrained(siglip_path)
model = SiglipModel.from_pretrained(siglip_path).to(device)

logit_scale = model.logit_scale.detach().cpu().numpy()
logit_bias = model.logit_bias.detach().cpu().numpy()

str2id = {}
for cnt, key in enumerate(IMAGENET2012_CLASSES.keys()):
    str2id[key] = cnt 

texts = []
for key in IMAGENET2012_CLASSES.keys():
    texts.append(f"This is a photo of {IMAGENET2012_CLASSES[key]}.")

tokens = tk(texts, return_tensors="pt", padding="max_length", truncation=None, max_length=None)['input_ids'].to(device) # (bs, 64)
text_emb = model.get_text_features(tokens).detach().numpy() # (bs, 1152)
text_emb = text_emb / np.linalg.norm(text_emb, ord=2, axis=-1, keepdims=True)  # (M, 1152)

np.save("text_emb.npy", text_emb)

emb_path = f"ImageNet_embedding-siglip-large-patch16-384"
truth = []
emb_names = os.listdir(emb_path)
img_embs = []
for name in  emb_names:
    img_embs.append(np.load(os.path.join(emb_path, name)))
    truth.append(str2id[name[24:33]])
img_embs = np.stack(img_embs, axis=0)
img_embs = img_embs / np.linalg.norm(img_embs, ord=2, axis=-1, keepdims=True)
logits_per_text = np.dot(text_emb, img_embs.T)  # shape (M, N)
logits_per_text = logits_per_text * np.exp(logit_scale) + logit_bias
scores = sigmoid(logits_per_text)

top1_indices = np.argmax(scores, axis=0)
top5_indices = np.argpartition(scores, kth=-5, axis=0)[-5:, :]

top1_cnt = 0
top5_cnt = 0
total_cnt = 0

for top1, top5, tr in zip(top1_indices, top5_indices.T, truth):
    total_cnt += 1
    if int(top1) == int(tr):
        top1_cnt += 1
        top5_cnt += 1
        continue
    elif int(tr) in [int(_) for _ in top5]:
        top5_cnt +=1
        continue

print(f"TOP1: {(100*top1_cnt/total_cnt):.2f} %")
print(f"TOP1: {(100*top5_cnt/total_cnt):.2f} %")


# siglip-so400m-patch14-384
# TOP1: 78.72 %
# TOP5: 94.33 %

# siglip-large-patch16-384
# TOP1: 75.84 %
# TOP5: 92.52 %

