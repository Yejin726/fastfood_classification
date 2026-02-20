import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.amp import autocast

# ----------------------------
# Config
# ----------------------------
DATA_ROOT = "./data"
TEST_DIR = f"{DATA_ROOT}/Test/Test"
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 너가 이미 만든 체크포인트 구조
# ./checkpoint/224/convnext_small.pt 이런 구조
ENSEMBLE_ITEMS = [
    {"img_size": 224, "model_name": "convnext_small", "ckpt": "./checkpoint/224/convnext_small.pt", "weight": 1.25},
    {"img_size": 320, "model_name": "convnext_small", "ckpt": "./checkpoint/320/convnext_small.pt", "weight": 1.10},
    {"img_size": 224, "model_name": "convnext_tiny",  "ckpt": "./checkpoint/224/convnext_tiny.pt",  "weight": 1.00},
    {"img_size": 320, "model_name": "convnext_tiny",  "ckpt": "./checkpoint/320/convnext_tiny.pt",  "weight": 1.00},
]
# 위 weight는 예시야. 너가 val에서 best 나왔던 weight로 바꿔줘.


# ----------------------------
# Build model (너 코드 그대로)
# ----------------------------
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

    if model_name == "resnext50_32x4d":
        m = torchvision.models.resnext50_32x4d(
            weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    # fallback
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_model_from_ckpt(model_name: str, ckpt_path: str, num_classes: int):
    model = build_model(model_name, num_classes).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ----------------------------
# Test transform
# ----------------------------
def make_test_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


# ----------------------------
# Test dataset: (tensor, filename)
# ----------------------------
class TestDataset(Dataset):
    def __init__(self, test_dir: str, transform):
        self.test_dir = Path(test_dir)
        self.transform = transform

        # ✅ 하위 폴더까지 재귀적으로 이미지 검색
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [p for p in self.test_dir.rglob("*") if p.suffix.lower() in exts]

        # ✅ 숫자 파일명 정렬 (0.jpeg -> 0). 숫자 아닌 건 뒤로 보냄
        def sort_key(p: Path):
            try:
                return (0, int(p.stem))
            except ValueError:
                return (1, p.name)

        self.files = sorted(files, key=sort_key)

        # ✅ 디버그 출력
        print(f"[TestDataset] dir = {self.test_dir.resolve()}")
        print(f"[TestDataset] found {len(self.files)} image files")
        if len(self.files) > 0:
            print("[TestDataset] first 5:", [f.name for f in self.files[:5]])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, p.name

# ----------------------------
# Sequential weighted ensemble inference
# (해상도별로 데이터 로더 생성 -> 모델 하나씩 로드하여 logits 누적)
# ----------------------------
@torch.no_grad()
def predict_weighted_ensemble(test_dir: str, ensemble_items, num_classes: int, batch_size: int = 64, num_workers: int = 2):
    # img_size별로 그룹화
    by_size = {}
    for it in ensemble_items:
        by_size.setdefault(int(it["img_size"]), []).append(it)

    filenames_ref = None
    logits_sum_total = None
    total_weight = 0.0
    

    for img_size, items in sorted(by_size.items()):
        tf = make_test_transform(img_size)
        ds = TestDataset(test_dir, tf)
        
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # 파일 순서(224/320)가 완전히 같아야 한다
        cur_names = [p.name for p in ds.files]
        if filenames_ref is None:
            filenames_ref = cur_names
            logits_sum_total = torch.zeros((len(ds), num_classes), dtype=torch.float32)
        else:
            if cur_names != filenames_ref:
                raise RuntimeError(f"Test file order mismatch between img_size groups. size={img_size} differs.")

        for it in items:
            model_name = it["model_name"]
            ckpt_path  = it["ckpt"]
            w = float(it.get("weight", 1.0))
            total_weight += w

            print(f"[Load] {model_name} @ {img_size} | w={w} | {ckpt_path}")
            model = load_model_from_ckpt(model_name, ckpt_path, num_classes)

            offset = 0
            for x, _names in loader:
                b = x.size(0)
                x = x.to(DEVICE, non_blocking=True)

                with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = model(x)

                logits_sum_total[offset:offset+b] += (logits.float().cpu() * w)
                offset += b

            del model
            torch.cuda.empty_cache()

    logits_sum_total /= total_weight
    preds = torch.argmax(logits_sum_total, dim=1).numpy().tolist()
    return filenames_ref, preds


def main():
    print("TEST_DIR exists?", Path(TEST_DIR).exists(), "->", Path(TEST_DIR).resolve())
    print("TEST_DIR listing sample:", list(Path(TEST_DIR).iterdir())[:5])

    test_dir = TEST_DIR
    assert Path(test_dir).exists(), f"Test dir not found: {test_dir}"

    # 배치가 너무 크면 줄여
    batch_size = 64 if DEVICE.type == "cuda" else 16

    imgs, preds = predict_weighted_ensemble(
        test_dir=test_dir,
        ensemble_items=ENSEMBLE_ITEMS,
        num_classes=NUM_CLASSES,
        batch_size=batch_size,
        num_workers=2
    )

    # ✅ 제출 포맷: Image,Class
    df = pd.DataFrame({"Image": imgs, "Class": preds})
    out_path = "result.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(df.head(5))


if __name__ == "__main__":
    main()