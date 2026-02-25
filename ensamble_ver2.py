import os, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 43
    data_root: str = "./data"
    train_dir: str = "Train"
    num_classes: int = 10

    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2

    # checkpoints (너가 저장한 경로로 바꿔)
    ckpt_tiny: str = "./checkpoint_v2/convnextv2_tiny.pt"
    ckpt_base: str = "./checkpoint_v2/convnextv2_base.pt"

    # ensemble weights
    w_tiny: float = 1.0
    w_base: float = 1.2

    # TTA
    use_tta: bool = True  # 앙상블 TTA 평가할 때만 True

cfg = CFG()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================
# Metrics
# =========================
def macro_f1_from_confmat(cm: np.ndarray) -> float:
    eps = 1e-12
    f1s = []
    C = cm.shape[0]
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


# =========================
# Dataset wrappers
# =========================
class SubsetWithTransform(Dataset):
    def __init__(self, base_ds, indices, transform):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        path, y = self.base_ds.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y


class SubsetWithPIL(Dataset):
    """TTA용: PIL 반환(Resize/CenterCrop까지만 적용)"""
    def __init__(self, base_ds, indices, pil_transform):
        self.base_ds = base_ds
        self.indices = indices
        self.pil_transform = pil_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        path, y = self.base_ds.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.pil_transform(img)
        return img, y


# =========================
# Model
# =========================
def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "convnextv2_tiny":
        return timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k",
                                 pretrained=True, num_classes=num_classes)
    if model_name == "convnextv2_base":
        return timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k",
                                 pretrained=True, num_classes=num_classes)
    raise ValueError(model_name)


def load_model_from_ckpt(model_name: str, ckpt_path: str, num_classes: int):
    model = build_model(model_name, num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =========================
# Transforms
# =========================
def make_val_tf(img_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def make_val_tta_base(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
    ])

def make_to_tensor_norm(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.no_grad()
def tta_logits(model, img_pil, img_size, to_tensor_norm):
    """
    FiveCrop + HFlip = 10 views
    img_pil: PIL (Resize/CenterCrop 적용된 상태)
    return: (num_classes,) logits mean
    """
    crops = transforms.FiveCrop(img_size)(img_pil)
    views = []
    for c in crops:
        views.append(to_tensor_norm(c))
        views.append(to_tensor_norm(transforms.functional.hflip(c)))
    x = torch.stack(views, dim=0).to(device)
    logits = model(x)
    return logits.mean(dim=0)  # (C,)


# =========================
# Evaluation
# =========================
@torch.no_grad()
def eval_single_no_tta(model, loader, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    return macro_f1_from_confmat(cm)


@torch.no_grad()
def eval_ensemble_no_tta(model_a, model_b, loader, num_classes: int, w_a=1.0, w_b=1.0):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        la = model_a(x)
        lb = model_b(x)
        logits = (w_a * la + w_b * lb) / (w_a + w_b)

        pred = torch.argmax(logits, dim=1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    return macro_f1_from_confmat(cm)


@torch.no_grad()
def eval_ensemble_tta(model_a, model_b, loader_pil, num_classes: int,
                      img_size: int, to_tensor_norm, w_a=1.0, w_b=1.0):
    """
    loader_pil: batch_size=1, returns (PIL, y)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # for img_pil, y in loader_pil:
    #     img_pil = img_pil[0]  # batch_size=1
    #     y = int(y.item())
    for imgs, ys in loader_pil:
        img_pil = imgs[0]
        y = int(ys[0].item())

        la = tta_logits(model_a, img_pil, img_size, to_tensor_norm)
        lb = tta_logits(model_b, img_pil, img_size, to_tensor_norm)
        logits = (w_a * la + w_b * lb) / (w_a + w_b)

        pred = int(torch.argmax(logits).item())
        cm[y, pred] += 1

    return macro_f1_from_confmat(cm)

def pil_collate(batch):
    # batch: list of (PIL, y)
    imgs, ys = zip(*batch)
    return list(imgs), torch.tensor(ys, dtype=torch.long)


def main():
    root = Path(cfg.data_root)
    train_path = root / cfg.train_dir

    base_ds = ImageFolder(train_path.as_posix(), transform=None)
    targets = np.array(base_ds.targets)

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
    tr_idx, va_idx = next(sss.split(np.arange(len(targets)), targets))

    # --- load models ---
    model_tiny = load_model_from_ckpt("convnextv2_tiny", cfg.ckpt_tiny, cfg.num_classes)
    model_base = load_model_from_ckpt("convnextv2_base", cfg.ckpt_base, cfg.num_classes)

    # --- mean/std: timm cfg는 모델마다 다를 수 있어서, 우선 base 기준으로 통일 ---
    data_cfg = resolve_data_config({}, model=model_base)
    mean = data_cfg.get("mean", IMAGENET_DEFAULT_MEAN)
    std  = data_cfg.get("std", IMAGENET_DEFAULT_STD)

    val_tf = make_val_tf(cfg.img_size, mean, std)
    val_ds = SubsetWithTransform(base_ds, va_idx, val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # --- no TTA eval ---
    f1_tiny = eval_single_no_tta(model_tiny, val_loader, cfg.num_classes)
    f1_base = eval_single_no_tta(model_base, val_loader, cfg.num_classes)
    f1_ens  = eval_ensemble_no_tta(model_tiny, model_base, val_loader, cfg.num_classes,
                                   w_a=cfg.w_tiny, w_b=cfg.w_base)

    print(f"[VAL no TTA] tiny macroF1 = {f1_tiny:.4f}")
    print(f"[VAL no TTA] base macroF1 = {f1_base:.4f}")
    print(f"[VAL no TTA] ens  macroF1 = {f1_ens:.4f}  (w_tiny={cfg.w_tiny}, w_base={cfg.w_base})")

    # --- TTA ensemble eval (slow) ---
    if cfg.use_tta:
        val_tta_base = make_val_tta_base(cfg.img_size)
        to_tensor_norm = make_to_tensor_norm(mean, std)
        val_ds_pil = SubsetWithPIL(base_ds, va_idx, val_tta_base)
        # val_loader_pil = DataLoader(val_ds_pil, batch_size=1, shuffle=False, num_workers=0)
        val_loader_pil = DataLoader(
            val_ds_pil,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=pil_collate,
        )

        f1_ens_tta = eval_ensemble_tta(model_tiny, model_base, val_loader_pil, cfg.num_classes,
                                       cfg.img_size, to_tensor_norm,
                                       w_a=cfg.w_tiny, w_b=cfg.w_base)
        print(f"[VAL ens + TTA] macroF1 = {f1_ens_tta:.4f}")

if __name__ == "__main__":
    main()