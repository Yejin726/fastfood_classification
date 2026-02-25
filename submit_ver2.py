import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# =========================
# Config
# =========================
DATA_ROOT = "./data"
TEST_DIR = f"{DATA_ROOT}/Test"
NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 64

CKPT_TINY = "./checkpoint_v2/convnextv2_tiny.pt"
CKPT_BASE = "./checkpoint_v2/convnextv2_base.pt"

W_TINY = 1.0
W_BASE = 1.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)


# =========================
# Model
# =========================
def build_model(model_name: str, num_classes: int):
    if model_name == "convnextv2_tiny":
        return timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
    if model_name == "convnextv2_base":
        return timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_classes,
        )
    raise ValueError(model_name)


def load_model(model_name, ckpt_path):
    model = build_model(model_name, NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =========================
# Dataset
# =========================
class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.files = sorted([
            p for p in Path(test_dir).rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, p.name


# =========================
# Main
# =========================
def main():

    # 모델 로드
    model_tiny = load_model("convnextv2_tiny", CKPT_TINY)
    model_base = load_model("convnextv2_base", CKPT_BASE)

    # mean/std (base 기준으로 통일)
    data_cfg = resolve_data_config({}, model=model_base)
    mean = data_cfg.get("mean", IMAGENET_DEFAULT_MEAN)
    std = data_cfg.get("std", IMAGENET_DEFAULT_STD)

    test_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_ds = TestDataset(TEST_DIR, test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    all_names = []
    all_preds = []

    with torch.no_grad():
        for x, names in test_loader:
            x = x.to(DEVICE, non_blocking=True)

            logits_tiny = model_tiny(x)
            logits_base = model_base(x)

            logits = (W_TINY * logits_tiny + W_BASE * logits_base) / (W_TINY + W_BASE)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_names.extend(names)
            all_preds.extend(preds.tolist())

    df = pd.DataFrame({
        "Image": all_names,
        "Class": all_preds
    })

    df.to_csv("result.csv", index=False)
    print("Saved result.csv")
    print(df.head())


if __name__ == "__main__":
    main()