# backend/agents/classifier_agent.py
from __future__ import annotations

from pathlib import Path
import uuid
import shutil

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from fastapi import UploadFile


# ---------------- 기본 경로 / 디바이스 ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR = PROJECT_ROOT / "classifier_weight"
UPLOAD_DIR = PROJECT_ROOT / "uploads_cls"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[CLS] DEVICE      :", DEVICE)
print("[CLS] WEIGHT_DIR  :", WEIGHT_DIR)


# ---------------- 이미지 전처리 ----------------
base_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


# ---------------- 모델 래퍼 ----------------
class SegmentationClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # DeepLabV3 의 출력 중 "out" (N, C, H, W)을 평균 pooling → (N, C)
        return self.backbone(x)["out"].mean(dim=(2, 3))


# 카테고리별 모델 캐시 (capsule, tile, leather)
_model_cache: dict[str, tuple[nn.Module, dict[int, str]]] = {}


def load_model_for_category(category: str):
    """
    category: "capsule" / "tile" / "leather"
    해당 카테고리용 best_*.pth 를 불러와서 (model, idx_to_class) 반환
    """
    key = category.lower()
    if key in _model_cache:
        return _model_cache[key]

    ckpt_path = WEIGHT_DIR / f"best_{key}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[CLS] checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    num_classes = len(checkpoint["class_to_idx"])

    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    backbone = deeplabv3_resnet50(weights=weights)
    in_ch = backbone.classifier[-1].in_channels
    backbone.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    model = SegmentationClassifier(backbone, num_classes).to(DEVICE)
    model.load_state_dict(
        {k: v.to(DEVICE) for k, v in checkpoint["model_state"].items()}
    )
    model.eval()

    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

    _model_cache[key] = (model, idx_to_class)
    print(f"[CLS] loaded model for {category}: {ckpt_path}")
    return model, idx_to_class


def _predict_path(category: str, img_path: Path) -> str:
    """
    저장된 이미지 파일 경로를 받아서
    해당 카테고리의 모델로 결함 클래스를 예측 → 라벨 문자열 반환
    """
    model, idx_to_class = load_model_for_category(category)

    img = Image.open(img_path).convert("RGB")
    x = base_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_idx = model(x).argmax(dim=1).item()

    return idx_to_class[pred_idx]


# ---------------- main entry: FastAPI 에서 사용하는 함수 ----------------
async def classify_image(category: str, file: UploadFile) -> dict:
    """
    FastAPI /api/classify 엔드포인트에서 호출하는 함수.

    - 업로드된 이미지를 디스크에 저장
    - 카테고리별 분류 모델로 예측
    - 프론트가 바로 쓸 수 있는 dict 반환
    """
    suffix = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"cls_{uid}{suffix}"

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    pred_label = _predict_path(category, save_path)

    return {
        "ok": True,
        "category": category,
        "filename": file.filename,
        "predicted_defect": pred_label,
    }

# ---------------- main entry: FastAPI 에서 사용하는 함수 ----------------
async def classify_image(category: str, file: UploadFile) -> dict:
    """
    FastAPI /api/classify 엔드포인트에서 호출하는 함수.

    - 업로드된 이미지를 디스크에 저장
    - 카테고리별 분류 모델로 예측
    - 프론트가 바로 쓸 수 있는 dict 반환
    """
    suffix = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"cls_{uid}{suffix}"

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    pred_label = _predict_path(category, save_path)

    return {
        "ok": True,
        "category": category,
        "filename": file.filename,
        "predicted_defect": pred_label,
    }


# ✅ main.py 가 여전히 run_classification 을 import 해도 동작하도록 alias 추가
async def run_classification(category: str, file: UploadFile) -> dict:
    """
    Backward-compat wrapper.
    main.py 에서 사용하던 이름(run_classification)을 그대로 유지하기 위한 래퍼.
    실제 로직은 classify_image 를 그대로 호출한다.
    """
    return await classify_image(category, file)


