import os
import numpy as np
from tqdm import tqdm

def cosine_similarity(A, B):
    # 将张量展平为一维向量
    A_flat = A.flatten()
    B_flat = B.flatten()
    # 计算点积和范数
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    # 避免除以零
    if norm_A == 0 or norm_B == 0:
        return 0
    cos = dot_product / (norm_A * norm_B)
    error = A - B
    mse = np.mean((A - B) ** 2)
    return (cos, mse)

names = ["siglip-base-patch16-224",
            "siglip-base-patch16-384",
            "siglip-base-patch16-512",
            "siglip-large-patch16-256",
            "siglip-large-patch16-384",
            "siglip-so400m-patch14-224",
            "siglip-so400m-patch14-384",
            "siglip-so400m-patch16-256-i18n"]
result_str_coses = ""
result_str_mses = ""
for s in names:
    bpu_p = f"COCO2017_BPU_last_hidden_state_{s}"
    gpu_all = np.load(f"COCO2017_last_hidden_state_siglip/COCO2017_last_hidden_state_{s}.npz")
    names = os.listdir(bpu_p)
    coses, mses = [], []
    for cnt in tqdm(range(len(names)), desc=f"{s}", unit=" item"):
        name = names[cnt]
        bpu = np.load(os.path.join(bpu_p, name))
        gpu = gpu_all[name[:-4] + ".jpg"]
        cos, mse = cosine_similarity(bpu, gpu)
        coses.append(cos)
        mses.append(mse)
    coses, mses = np.array(coses), np.array(mses)
    result_str_coses += f"| {s} | {np.mean(coses):.3f} ( {coses.min():.3f} ~ {coses.max():.3f} ), {np.percentile(coses, 1):.3f} | {np.mean(mses):.3f} ( {mses.min():.3f} ~ {mses.max():.3f} ), {np.percentile(mses, 1):.3f} | \n"
    # result_str_mses += f"| {s} | {np.mean(mses):.5f} ( {mses.min():.5f} ~ {mses.max():.5f} ) | {np.percentile(mses, 1):.5f} |  \n"

print()
print(result_str_coses)
print(result_str_mses)