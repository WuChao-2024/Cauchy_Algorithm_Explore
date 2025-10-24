'''
Date: 2025-10-22
Author: Cauchy-WuChao
Description: 
  01: 读取ImageNet-1k的50,000张图片, 使用SigLIP生成图像嵌入
  02: 读取ImageNet-1k的标签, 使用SigLIP生成词嵌入
* 03: 比较词嵌入的cos, 的到TOP1和TOP5的acc
  04: 通过hbm模型生成图像嵌入, 重复03, 获取对应的TOP1和TOP5的acc
'''

# Download ImageNet-1k DataSet: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/4603483700ee984ea9debe3ddbfdeae86f6489eb
from classes import IMAGENET2012_CLASSES
import os
import numpy as np
import argparse
from tqdm import tqdm
from scipy.special import expit as sigmoid
# import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-vec-path', type=str, default="ImageNet_ImgVec_siglip-base-patch16-224.npz", help="")
    parser.add_argument('--text-vec-path', type=str, default="ImageNet_TextVec_siglip-base-patch16-224.npz", help="")
    opt = parser.parse_args()

    text_emb = np.load(opt.text_vec_path)

    logit_scale = text_emb["logit_scale"]
    logit_bias = text_emb["logit_bias"]

    str2id = {}
    for cnt, key in enumerate(IMAGENET2012_CLASSES.keys()):
        str2id[key] = cnt 

    text_emb = text_emb["text_emb"]

    img_embs_ = np.load(opt.img_vec_path)
    truth = []
    img_embs = []
    for name in tqdm(img_embs_.keys(), desc="Read Datas: ", unit=" sample"):
        img_embs.append(img_embs_[name][0])
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
    # for top1, top5, tr in tqdm(zip(top1_indices, top5_indices.T, truth), desc="Computing Acc", unit=" sample"):
    for top1, top5, tr in zip(top1_indices, top5_indices.T, truth):
        total_cnt += 1
        if int(top1) == int(tr):
            top1_cnt += 1
            top5_cnt += 1
            continue
        elif int(tr) in [int(_) for _ in top5]:
            top5_cnt +=1
            continue

    print(f"TOP1: {(top1_cnt/total_cnt):.4f}")
    print(f"TOP1: {(top5_cnt/total_cnt):.4f}")

def softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

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