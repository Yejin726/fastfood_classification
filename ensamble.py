import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.amp import autocast
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


val_tf_224 = transforms.Compose([
    transforms.Resize(int(224 * 1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tf_320 = transforms.Compose([
    transforms.Resize(int(320 * 1.15)),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

root = Path(cfg.data_root)
train_path = root / cfg.train_dir

base_ds = ImageFolder(train_path.as_posix(), transform=None)
targets = np.array(base_ds.targets)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
tr_idx, va_idx = next(sss.split(np.arange(len(targets)), targets))


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

val_ds_224   = SubsetWithTransform(base_ds, va_idx, val_tf_224)
val_ds_320   = SubsetWithTransform(base_ds, va_idx, val_tf_320)


val_loader_224 = DataLoader(
    val_ds_224,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)
val_loader_320 = DataLoader(
    val_ds_320,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)


# ----------------------------
# 모델 로드
# ----------------------------
def load_model_from_ckpt(model_name, ckpt_path, num_classes):
    model = build_model(model_name, num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ----------------------------
# 단일 모델 validation 평가
# ----------------------------
@torch.no_grad()
def eval_single_model(model, val_loader):
    all_preds = []
    all_targets = []

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    return f1_score(all_targets, all_preds, average="macro")


# ----------------------------
# 앙상블 validation 평가 (logits 평균)
# ----------------------------
@torch.no_grad()
def eval_ensemble(models_info, num_classes, tta_mode="none"):

    all_logits_sum = None
    all_targets = None
    total_weight = 0.0

    for info in models_info:

        weight = info.get("weight", 1.0)
        total_weight += weight

        model = load_model_from_ckpt(
            info["model_name"],
            info["ckpt"],
            num_classes
        )

        logits_list = []
        targets_list = []

        for x, y in info["val_loader"]:
            x = x.to(device)
            y = y.to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = forward_with_tta(model, x, tta_mode)

            logits_list.append(logits.cpu())
            targets_list.append(y.cpu())

        logits_all = torch.cat(logits_list) * weight   # ⭐ 여기 weight 적용

        if all_logits_sum is None:
            all_logits_sum = logits_all
            all_targets = torch.cat(targets_list)
        else:
            all_logits_sum += logits_all

        del model
        torch.cuda.empty_cache()

    # 평균으로 정규화 (선택이지만 추천)
    all_logits_sum /= total_weight

    preds = torch.argmax(all_logits_sum, dim=1).numpy()
    targets = all_targets.numpy()

    return f1_score(targets, preds, average="macro")


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

# ----------------------------
# TTA forward
# ----------------------------
def forward_with_tta(model, x, tta_mode="none"):
    """
    tta_mode:
        "none"  -> 기본
        "flip"  -> 원본 + horizontal flip (2회)
        "4crop" -> 4회 (원본, flip, resize1.0, resize1.0+flip)
    """

    if tta_mode == "none":
        return model(x)

    # -----------------------
    # 2회 TTA (원본 + flip)
    # -----------------------
    if tta_mode == "flip":
        logits1 = model(x)
        x_flip = torch.flip(x, dims=[3])
        logits2 = model(x_flip)
        return (logits1 + logits2) / 2.0

    # -----------------------
    # 4회 TTA
    # -----------------------
    if tta_mode == "4crop":

        logits_list = []

        # 1. 원본
        logits_list.append(model(x))

        # 2. flip
        logits_list.append(model(torch.flip(x, dims=[3])))

        # 3. slight scale (1.05배 resize 후 center crop)
        B, C, H, W = x.shape
        scaled = torch.nn.functional.interpolate(
            x, scale_factor=1.05, mode="bilinear", align_corners=False
        )
        scaled = scaled[:, :, :H, :W]  # center crop 비슷하게 맞춤
        logits_list.append(model(scaled))

        # 4. scaled + flip
        logits_list.append(model(torch.flip(scaled, dims=[3])))

        return torch.stack(logits_list).mean(0)

    raise ValueError("Unknown tta_mode")


num_classes = 10

models = [
    {"model_name": "convnext_small", "ckpt": "./checkpoint/224/convnext_small.pt", "val_loader": val_loader_224},
    {"model_name": "convnext_small", "ckpt": "./checkpoint/320/convnext_small.pt", "val_loader": val_loader_320},
    {"model_name": "convnext_tiny",  "ckpt": "./checkpoint/224/convnext_tiny.pt",  "val_loader": val_loader_224},
    {"model_name": "convnext_tiny",  "ckpt": "./checkpoint/320/convnext_tiny.pt",  "val_loader": val_loader_320},
    {"model_name": "resnext50_32x4d","ckpt": "./checkpoint/224/resnext50_32x4d.pt","val_loader": val_loader_224},
    {"model_name": "resnext50_32x4d","ckpt": "./checkpoint/320/resnext50_32x4d.pt","val_loader": val_loader_320},
]

# print("=== Single Model Validation Scores ===")

# for m in models:
#     model = load_model_from_ckpt(m["model_name"], m["ckpt"], num_classes)
#     f1 = eval_single_model(model, m["val_loader"])
#     print(f"{m['ckpt']}  ->  {f1:.4f}")
#     del model
#     torch.cuda.empty_cache()
    
    
# ensemble1 = [models[0], models[1]]
# f1 = eval_ensemble(ensemble1, num_classes)
# print("Ensemble small 224+320:", f1)

# ensemble2 = [models[0], models[1], models[2], models[3]]
# f1 = eval_ensemble(ensemble2, num_classes)
# print("Ensemble small+tiny:", f1)

f1 = eval_ensemble(models, num_classes)
print("Ensemble all 6:", f1)

models_weighted = [
    {"model_name": "convnext_small", "ckpt": "./checkpoint/224/convnext_small.pt",
     "val_loader": val_loader_224, "weight": 1.2},

    {"model_name": "convnext_small", "ckpt": "./checkpoint/320/convnext_small.pt",
     "val_loader": val_loader_320, "weight": 1.0},

    {"model_name": "convnext_tiny",  "ckpt": "./checkpoint/224/convnext_tiny.pt",
     "val_loader": val_loader_224, "weight": 1.0},

    {"model_name": "convnext_tiny",  "ckpt": "./checkpoint/320/convnext_tiny.pt",
     "val_loader": val_loader_320, "weight": 1.0},

    {"model_name": "resnext50_32x4d","ckpt": "./checkpoint/224/resnext50_32x4d.pt",
     "val_loader": val_loader_224, "weight": 0.9},

    {"model_name": "resnext50_32x4d","ckpt": "./checkpoint/320/resnext50_32x4d.pt",
     "val_loader": val_loader_320, "weight": 0.9},
]

ensemble2 = [models_weighted[0], models_weighted[1], models_weighted[2], models_weighted[3]]
f1 = eval_ensemble(ensemble2, num_classes)
print("Weighted ensemble small+tiny:", f1)

f1 = eval_ensemble(models_weighted, num_classes)
print("Weighted ensemble:", f1)

# f1 = eval_ensemble(models, num_classes, tta_mode="flip")
# print("Ensemble all 6 + flip:", f1)

# f1 = eval_ensemble(models, num_classes, tta_mode="4crop")
# print("Ensemble all 6 + 4TTA:", f1)