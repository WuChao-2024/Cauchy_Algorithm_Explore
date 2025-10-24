import os

names = ["siglip-base-patch16-224",
            "siglip-base-patch16-384",
            "siglip-base-patch16-512",
            "siglip-large-patch16-256",
            "siglip-large-patch16-384",
            "siglip-so400m-patch14-224",
            "siglip-so400m-patch14-384",
            "siglip-so400m-patch16-256-i18n"]
for s in names:
    print(f"=" * 100)
    os.system(f"hrt_model_exec model_info --model_file bpu-{s}.hbm")