
from transformers import SiglipModel, SiglipProcessor, SiglipTokenizer
import torch, cv2, time, os
import numpy as np
import time

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

device = torch.device("cuda:2")
siglip_path = "siglip-so400m-patch14-384"

model = SiglipModel.from_pretrained(siglip_path).to(device)
# tk = SiglipTokenizer.from_pretrained(siglip_path).to(device)

for video_name in ["crash1", "crash2", "crash3"]:
    imagenet_path = f"3d_printer_crash/{video_name}"
    names = os.listdir(imagenet_path)
    processed_num = 0
    total_num = len(names)
    for cnt, name in enumerate(names):
        img_path = os.path.join(imagenet_path, name)
        img = cv2.imread(img_path)
        input_tensor = preprocess(img).to(device)
        forward_begin_time = time.time()
        with torch.no_grad():
            img_vec = model.get_image_features(input_tensor).to(device).detach().cpu().numpy()
        forward_time = time.time() - forward_begin_time
        save_begin_time = time.time()
        np.save(f"3d_printer_crash/{video_name}_emb/{name[:-4]}.npy", img_vec)
        save_time = time.time() - save_begin_time
        processed_num += 1
        rate = 100* processed_num/total_num
        print(f"Process {video_name}: [{processed_num} / {total_num}] {rate:.2f}%, forward: {forward_time:.3f} s, save: {save_time:.3f} s.")


print(f"{processed_num = }")

exit()

img = cv2.imread("/home/users/chao01.wu/Datasets/COCO2017/val2017/000000039769.jpg")
tensor = preprocess(img)
img_vec = model.get_image_features(tensor).detach() # (bs, 1152)

