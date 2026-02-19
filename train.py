
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
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import RandAugment

from torch.cuda.amp import autocast, GradScaler

# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 42
    data_root: str = "./data"     # <- 수정
    train_dir: str = "Train"             # train/<class_name>/*.jpg
    test_dir: str  = "Test"              # test/*.jpg

    num_classes: int = 10
    img_size: int = 320
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 25 #15
    lr: float = 3e-4
    weight_decay: float = 1e-4



    use_weighted_sampler: bool = False
    use_class_weight_loss: bool = True

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
# Metrics: Macro F1
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
# Transforms (augmentation)
# =========================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    RandAugment(num_ops=2, magnitude=9), #9
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    #transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_tf = transforms.Compose([
    transforms.Resize(int(cfg.img_size * 1.15)),
    transforms.CenterCrop(cfg.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# =========================
# Dataset split: Stratified
# =========================
root = Path(cfg.data_root)
train_path = root / cfg.train_dir
test_path  = root / cfg.test_dir

# ImageFolder will map class_name -> index alphabetically
base_ds = ImageFolder(train_path.as_posix(), transform=None)
targets = np.array(base_ds.targets)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
tr_idx, va_idx = next(sss.split(np.arange(len(targets)), targets))


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

# wrap subset with transform
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

train_ds = SubsetWithTransform(base_ds, tr_idx, train_tf)
val_ds   = SubsetWithTransform(base_ds, va_idx, val_tf)

# =========================
# Imbalance options
# =========================
counts = np.bincount(targets[tr_idx], minlength=cfg.num_classes).astype(np.float32)

sampler = None
if cfg.use_weighted_sampler:
    class_w = 1.0 / np.maximum(counts, 1.0)
    # compute sample weights for train subset
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
    w = w / w.mean()   # 평균 1로 정규화 (강도 완화)
    loss_weights = torch.tensor(w, dtype=torch.float32).to(device)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=cfg.num_workers,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)

# =========================
# Model: CNN + ImageNet weights
# (ResNet50 baseline)
# =========================
def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        return m

    if model_name == "convnext_small":
        m = torchvision.models.convnext_small(
            weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        return m
    
    if model_name == "convnext_base":
        m = torchvision.models.convnext_base(
            weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        return m

    if model_name == "efficientnet_b2":
        m = torchvision.models.efficientnet_b2(
            weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        return m

    if model_name == "resnext50_32x4d":
        m = torchvision.models.resnext50_32x4d(
            weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if model_name == "wide_resnet50_2":
        m = torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    # fallback
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

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

def plot_f1_bar(per_f1, class_names):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(class_names, per_f1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

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

#convnext_tiny, convnext_small, convnext_base, efficientnet_b2, resnext50_32x4d, wide_resnet50_2
cfg.model_name = "resnext50_32x4d"
model = build_model(cfg.model_name, cfg.num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

# =========================
# Train / Validate (best by macro F1)
# =========================
best_f1 = -1.0
wait = 0 
#best_path = root / "best_resnet50.pt"
best_path = Path(f"./checkpoint/320/{cfg.model_name}.pt")


#run_name = f"resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_name = f"{cfg.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
global_step = 0
patience = 7          # 개선 없으면 몇 epoch 기다릴지 (추천 6~8)
min_delta = 1e-4      # 이 정도 이상 좋아져야 "개선"으로 인정


class_names = base_ds.classes  # ImageFolder 기준
dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(device)
writer.add_graph(model, dummy)


for epoch in range(1, cfg.epochs + 1):
    model.train()
    tr_loss = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # optimizer.zero_grad(set_to_none=True)
        # logits = model(x)
        # loss = criterion(logits, y)
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4, p=0.5)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            # logits = model(x)
            # loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        bs = x.size(0)
        tr_loss += loss.item() * bs

        # (선택) 배치 단위 loss 로깅 - 너무 자주 찍히면 느릴 수 있음
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        global_step += 1

    scheduler.step()
    tr_loss /= len(train_loader.dataset)

    # ---------- validation ----------
    model.eval()
    va_loss = 0.0
    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # logits = model(x)
            # loss = criterion(logits, y)
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
        print("  -> saved best:", best_path if isinstance(best_path, str) else best_path.as_posix())
    else:
        wait += 1
        print(f"  -> no improve. wait={wait}/{patience}")

    if wait >= patience:
        print(f"Early stopping at epoch {epoch}. Best val macro F1={best_f1:.4f}")
        break
    
    

    # ---------- epoch 단위 로깅 ----------
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

    if f1 > best_f1:
        best_f1 = f1
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)
        print("  -> saved best:", best_path.as_posix())

writer.close()


print("Best val macro F1:", best_f1)









# =========================
# (Optional) Test inference -> submission.csv
# =========================
# class TestDataset(Dataset):
#     def __init__(self, img_dir: Path, transform):
#         self.img_dir = img_dir
#         self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
#         self.transform = transform

#     def __len__(self): return len(self.files)

#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         img = Image.open(self.img_dir / fname).convert("RGB")
#         img = self.transform(img)
#         return img, fname

# load best
# ckpt = torch.load(best_path, map_location="cpu")
# model.load_state_dict(ckpt["model"])
# model.eval().to(device)

# test_ds = TestDataset(test_path, val_tf)
# test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
#                          num_workers=cfg.num_workers, pin_memory=True)

# imgs, preds = [], []
# with torch.no_grad():
#     for x, fnames in test_loader:
#         x = x.to(device, non_blocking=True)
#         logits = model(x)
#         p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
#         preds.extend(p)
#         imgs.extend(list(fnames))

# import pandas as pd
# sub = pd.DataFrame({"Image": imgs, "Class": preds})
# out_path = root / "result.csv"
# sub.to_csv(out_path, index=False)
# print("Saved submission:", out_path.as_posix())
>>>>>>> 49ba4c9 (update train.py)
