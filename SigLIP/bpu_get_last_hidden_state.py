from libpyCauchyKesai import CauchyKesai
import cv2, time, os
import numpy as np
import time
from tqdm import tqdm
import argparse

def main():
    names = ["siglip-base-patch16-224",
             "siglip-base-patch16-384",
             "siglip-base-patch16-512",
             "siglip-large-patch16-256",
             "siglip-large-patch16-384",
             "siglip-so400m-patch14-224",
             "siglip-so400m-patch14-384",
             "siglip-so400m-patch16-256-i18n"]
    for s in names:
        parser = argparse.ArgumentParser()
        parser.add_argument('--img-vec-path', type=str, default=f"COCO2017_BPU_nashm_featuremaplast_hidden_state_{s}", help="")
        parser.add_argument('--bpu-siglip-weight-path', type=str, default=f"bpu-nashm-featuremap-{s}.hbm", help="")
        # parser.add_argument('--img-vec-path', type=str, default="ImageNet_ImgVec_siglip-large-patch16-256.npz", help="")
        # parser.add_argument('--siglip-weight-path', type=str, default="siglip-large-patch16-256", help="")
        parser.add_argument('--raw-data-path', type=str, default="/root/ssd/DataSets/COCO2017/val2017", help="")
        opt = parser.parse_args()
        os.makedirs(opt.img_vec_path, exist_ok=True)
        try:
            img_siz = int(s[-3:])
        except:
            img_siz = 256
        model = CauchyKesai(opt.bpu_siglip_weight_path, 1, 0)

        names = os.listdir(opt.raw_data_path)
        img_vecs = {}
        for cnt in tqdm(range(len(names)), desc=f"{s}", unit=" item"):
            name = names[cnt]
            img_path = os.path.join(opt.raw_data_path, name)
            last_hidden_state = model([preprocess(cv2.imread(img_path), img_siz).copy()])[0]
            np.save(os.path.join(opt.img_vec_path, f"{name[:-4]}.npy"), last_hidden_state)

    
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
    return image_normalized

if __name__ == "__main__":
    main()
    