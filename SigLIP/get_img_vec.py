'''
Date: 2025-10-22
Author: Cauchy-WuChao
Description: 
* 01: 读取ImageNet-1k的50,000张图片, 使用SigLIP生成图像嵌入
  02: 读取ImageNet-1k的标签, 使用SigLIP生成词嵌入
  03: 比较词嵌入的cos, 的到TOP1和TOP5的acc
  04: 通过hbm模型生成图像嵌入, 重复03, 获取对应的TOP1和TOP5的acc
'''

from transformers import SiglipModel, SiglipProcessor, SiglipTokenizer
import torch, cv2, time, os
import numpy as np
import time
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:5", help="cpu / cuda / cuda:1")
    s = "siglip-base-patch16-384"
    parser.add_argument('--img-vec-path', type=str, default=f"ImageNet_ImgVec_{s}.npz", help="")
    parser.add_argument('--siglip-weight-path', type=str, default=f"{s}", help="")
    # parser.add_argument('--img-vec-path', type=str, default="ImageNet_ImgVec_siglip-large-patch16-256.npz", help="")
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
    parser.add_argument('--raw-data-path', type=str, default="/home/users/chao01.wu/Datasets/ImageNet/val_images", help="")
    ''' 
    Download ImageNet-1k DataSet: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/4603483700ee984ea9debe3ddbfdeae86f6489eb
    for example: 
    raw_imgs/
    ├── ILSVRC2012_val_00033334_n02106550.JPEG
    ...
    ├── ILSVRC2012_val_00016661_n09428293.JPEG
    └── ILSVRC2012_val_00016667_n03902125.JPEG

    0 directories, 24 files
    '''
    opt = parser.parse_args()
    device = torch.device(opt.device)
    model = SiglipModel.from_pretrained(opt.siglip_weight_path).to(device)
    names = os.listdir(opt.raw_data_path)
    img_vecs = {}
    for cnt in tqdm(range(len(names)), desc="Prepare Calibration Datas: ", unit=" sample"):
        name = names[cnt]
        img_path = os.path.join(opt.raw_data_path, name)
        with torch.no_grad():
            img_vec = model.get_image_features(preprocess(cv2.imread(img_path), model.config.vision_config.image_size).to(device)).detach().cpu().numpy()
            img_vecs[name] = img_vec
    np.savez(opt.img_vec_path, **img_vecs)
    
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

if __name__ == "__main__":
    main()