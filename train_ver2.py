import os, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandAugment
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from torch.cuda.amp import autocast, GradScaler

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
    test_dir: str  = "Test"

    num_classes: int = 10
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 100
    lr: float = 1e-4 #3e-4
    weight_decay: float = 5e-2 #1e-4

    use_weighted_sampler: bool = False
    use_class_weight_loss: bool = True

    # validation TTA (느림)
    use_val_tta: bool = False         # ✅ 필요할 때만 True
    val_tta_every: int = 5            # use_val_tta=True면 몇 epoch마다 할지

    model_name: str = "convnextv2_base"  # tiny부터 추천 (T4 기준)

cfg = CFG()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
scaler = GradScaler(enabled=(device.type == "cuda"))


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


def per_class_f1_from_confmat(cm: np.ndarray):
    eps = 1e-12
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)
    return np.array(f1s)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_f1_bar(per_f1, class_names):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(class_names, per_f1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# =========================
# Dataset wrapper
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
    """TTA용: PIL을 반환"""
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
        img = self.pil_transform(img)  # PIL -> PIL
        return img, y


# =========================
# Model
# =========================
def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "convnextv2_tiny":
        return timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=num_classes)
    # if model_name == "convnextv2_small":
    #     return timm.create_model("convnextv2_small.fcmae_ft_in1k", pretrained=True, num_classes=num_classes)
    if model_name == "convnextv2_small":
        return timm.create_model(
            "convnextv2_small",
            pretrained=True,
            num_classes=num_classes,
        )
    if model_name == "convnextv2_base":
        return timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, num_classes=num_classes)
    raise ValueError(model_name)


# =========================
# CutMix
# =========================
def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, p=0.5):
    if np.random.rand() > p or alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B = x.size(0)
    index = torch.randperm(B).to(x.device)
    y_a = y
    y_b = y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1.0 - area / (x.size(2) * x.size(3))
    return x_cut, y_a, y_b, lam

def mixup_data(x, y, alpha=0.2, p=1.0):
    if np.random.rand() > p:
        return x, y, y, 1.0  # mixup 안함

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================
# TTA helpers (val 옵션)
# =========================
def make_to_tensor_norm(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

@torch.no_grad()
def tta_logits(model, img_pil, device, img_size, to_tensor_norm):
    crops = transforms.FiveCrop(img_size)(img_pil)
    all_imgs = []
    for c in crops:
        all_imgs.append(to_tensor_norm(c))
        all_imgs.append(to_tensor_norm(transforms.functional.hflip(c)))
    x = torch.stack(all_imgs, dim=0).to(device)
    logits = model(x)
    return logits.mean(dim=0)  # (num_classes,)


# =========================
# Main
# =========================
root = Path(cfg.data_root)
train_path = root / cfg.train_dir

base_ds = ImageFolder(train_path.as_posix(), transform=None)
targets = np.array(base_ds.targets)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
tr_idx, va_idx = next(sss.split(np.arange(len(targets)), targets))

# 1) 모델 먼저 만든다 (timm cfg 얻기 위해)
model = build_model(cfg.model_name, cfg.num_classes).to(device)

# 2) timm cfg로 mean/std 확보
data_cfg = resolve_data_config({}, model=model)
mean = data_cfg.get("mean", IMAGENET_DEFAULT_MEAN)
std  = data_cfg.get("std", IMAGENET_DEFAULT_STD)

# 3) transforms (이제서야 정의)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(cfg.img_size, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_tf = transforms.Compose([
    transforms.Resize(int(cfg.img_size * 1.15)),
    transforms.CenterCrop(cfg.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# (옵션) TTA용 PIL base
val_tta_base = transforms.Compose([
    transforms.Resize(int(cfg.img_size * 1.15)),
    transforms.CenterCrop(cfg.img_size),
])
to_tensor_norm = make_to_tensor_norm(mean, std)

# 4) datasets / loaders
train_ds = SubsetWithTransform(base_ds, tr_idx, train_tf)

if cfg.use_val_tta:
    # TTA 평가할 땐 PIL dataset + batch_size=1 권장
    val_ds = SubsetWithPIL(base_ds, va_idx, val_tta_base)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
else:
    val_ds = SubsetWithTransform(base_ds, va_idx, val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

# imbalance
counts = np.bincount(targets[tr_idx], minlength=cfg.num_classes).astype(np.float32)

sampler = None
if cfg.use_weighted_sampler:
    class_w = 1.0 / np.maximum(counts, 1.0)
    ys = targets[tr_idx]
    sample_w = class_w[ys]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )

loss_weights = None
if cfg.use_class_weight_loss:
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.mean()
    loss_weights = torch.tensor(w, dtype=torch.float32).to(device)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=cfg.num_workers,
    pin_memory=True
)

# opt
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

# logging
best_f1 = -1.0
wait = 0
patience = 100
min_delta = 1e-4

best_path = Path(f"./checkpoint_v2/{cfg.model_name}.pt")
best_path.parent.mkdir(parents=True, exist_ok=True)

run_name = f"{cfg.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
global_step = 0

class_names = base_ds.classes
dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(device)
writer.add_graph(model, dummy)

base_lr = cfg.lr
warmup_epochs = 1

# =========================
# Train / Validate
# =========================
for epoch in range(1, cfg.epochs + 1):
    model.train()
    tr_loss = 0.0
    
    if epoch <= warmup_epochs:
        lr_now = base_lr * (epoch / warmup_epochs) * 0.1  # 0.1배로 시작
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
    
    
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        


        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            #x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0, p=0.3)
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, p=0.5)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tr_loss += loss.item() * x.size(0)
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        global_step += 1

    scheduler.step()
    tr_loss /= len(train_loader.dataset)

    # ---------- validation ----------
    model.eval()
    va_loss = 0.0
    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)

    with torch.no_grad():
        if (cfg.use_val_tta and (epoch % cfg.val_tta_every == 0)):
            # TTA 평가
            for img_pil, y in val_loader:
                img_pil = img_pil[0]  # batch_size=1
                y = y.to(device)

                with autocast(enabled=(device.type == "cuda")):
                    logits1 = tta_logits(model, img_pil, device, cfg.img_size, to_tensor_norm).unsqueeze(0)
                    loss = criterion(logits1, y)

                va_loss += loss.item()
                pred = torch.argmax(logits1, dim=1).item()
                cm[int(y.item()), int(pred)] += 1
            va_loss /= len(val_loader.dataset)
        else:
            # 일반 평가(빠름)
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)
                va_loss += loss.item() * x.size(0)
                pred = torch.argmax(logits, dim=1)
                for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                    cm[t, p] += 1
            va_loss /= len(val_loader.dataset)

    f1 = macro_f1_from_confmat(cm)
    lr = optimizer.param_groups[0]["lr"]

    improved = (f1 > best_f1 + min_delta)
    if improved:
        best_f1 = f1
        wait = 0
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)
        print("  -> saved best:", best_path.as_posix())
    else:
        wait += 1
        print(f"  -> no improve. wait={wait}/{patience}")

    if wait >= patience:
        print(f"Early stopping at epoch {epoch}. Best val macro F1={best_f1:.4f}")
        break

    writer.add_scalar("train/loss_epoch", tr_loss, epoch)
    writer.add_scalar("val/loss", va_loss, epoch)
    writer.add_scalar("val/macro_f1", f1, epoch)
    writer.add_scalar("train/lr", lr, epoch)

    per_f1 = per_class_f1_from_confmat(cm)
    fig = plot_f1_bar(per_f1, class_names)
    writer.add_figure("val/per_class_f1", fig, epoch)
    plt.close(fig)

    if epoch % 5 == 0:
        fig = plot_confusion_matrix(cm, class_names)
        writer.add_figure("val/confusion_matrix", fig, epoch)
        plt.close(fig)

    print(f"[{epoch:02d}/{cfg.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_macroF1={f1:.4f} lr={lr:.2e}")

writer.close()
print("Best val macro F1:", best_f1)