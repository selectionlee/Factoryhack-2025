# backend/agents/adaptive_agent.py
from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from fastapi import UploadFile

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# ================== 공통 경로 / 디렉토리 ==================

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../backend
ADAPT_UPLOAD_DIR = PROJECT_ROOT / "uploads_adapt"
ADAPT_WEIGHT_DIR = PROJECT_ROOT / "adaptive_weight"

ADAPT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ADAPT_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[ADAPT] PROJECT_ROOT     = {PROJECT_ROOT}")
print(f"[ADAPT] ADAPT_WEIGHT_DIR = {ADAPT_WEIGHT_DIR}")
print(f"[ADAPT] DEVICE           = {DEVICE}")

# ================== 이미지 전처리 & 메타 정보 ==================

base_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

GOOD_CLASS_NAME = "good"

# 제외된 결함 정보(로그용)
EXCLUDED_DEFECTS = {
    "capsule": ["crack", "squeeze"],
    "tile": ["oil", "glue_strip"],
    "leather": ["cut", "poke"],
}

# ================== DeepLab 기반 classifier 래퍼 ==================


class SegmentationClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # DeepLabV3 출력: {"out": (B, C, H, W)}
        return self.backbone(x)["out"].mean(dim=(2, 3))


# ================== 제외 모델 로딩 / 추론 ==================


def _load_model_excluded(category: str) -> tuple[nn.Module, dict[int, str]]:
    """
    category에 대해 '제외 학습' 모델 로드.
    실제 파일 위치 : backend/adaptive_weight/best_{category}_excl.pth
    """
    ckpt_name = f"best_{category}_excl.pth"
    ckpt_path = (ADAPT_WEIGHT_DIR / ckpt_name).resolve()

    print(f"[ADAPT] loading excluded model for category={category}")
    print(f"[ADAPT] ckpt_path = {ckpt_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"[ADAPT] 체크포인트를 찾을 수 없습니다: {ckpt_path}")

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
    return model, idx_to_class


def _predict_with_probs_excluded(
    category: str,
    img_path: str,
    good_class_name: str = GOOD_CLASS_NAME,
):
    """
    '제외 학습 모델'에 대한 확률 기반 예측.
    """
    model, idx_to_class = _load_model_excluded(category)

    img = Image.open(img_path).convert("RGB")
    x = base_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, 1)

    probs = probs[0].cpu().numpy()

    class_probs = {idx_to_class[i]: float(p) for i, p in enumerate(probs)}

    pred_idx = int(probs.argmax())
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx])

    good_idx = None
    for i, name in idx_to_class.items():
        if name == good_class_name:
            good_idx = i
            break
    if good_idx is None:
        raise ValueError(f"'{good_class_name}' 클래스가 {category} 모델에 없습니다.")

    p_good = float(probs[good_idx])
    p_defect = 1.0 - p_good

    return {
        "category": category,
        "img_path": img_path,
        "pred_label": pred_label,
        "pred_conf": pred_conf,
        "class_probs": class_probs,
        "p_good": p_good,
        "p_defect": p_defect,
        "excluded_defects": EXCLUDED_DEFECTS.get(category, []),
    }


def _infer_image_excluded(category: str, img_path: str):
    """
    FastAPI 래퍼에서 쓰기 좋은 형태로 축약.
    """
    result = _predict_with_probs_excluded(category, img_path)
    return {
        "category": result["category"],
        "img_path": result["img_path"],
        "pred_label": result["pred_label"],
        "pred_conf": result["pred_conf"],
        "p_good": result["p_good"],
        "p_defect": result["p_defect"],
        "class_probs": result["class_probs"],
        "excluded_defects": result["excluded_defects"],
    }


# ================== FastAPI에서 호출하는 3단계 엔트리 ==================


async def run_adaptive(category: str, file: UploadFile):
    """
    3단계: 제외 결함 기반 적응학습 분석용 엔트리.
    - 1,2단계와 독립적으로 동작 (이미지 + category 만 필요)
    - 예측 확률(p_good, p_defect)을 기반으로
      '새로운 결함 유형 후보인지'를 판단해 요약 텍스트 생성
    """
    suffix = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex

    save_path = ADAPT_UPLOAD_DIR / f"adapt_{uid}{suffix}"
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 이 파일 안의 추론 함수 사용 (밖에 predict_excluded_probs.py 필요 없음)
    result = _infer_image_excluded(category, str(save_path))

    p_good = float(result.get("p_good", 0.0))
    p_defect = float(result.get("p_defect", 0.0))
    pred_label = result.get("pred_label")

    # ---------------------- 규칙 기반 해석 ----------------------
    if p_good >= 0.7:
        level_msg = "이 샘플은 대체로 정상에 가까운 패턴으로 보입니다."
        is_new = False
    elif p_good >= 0.4:
        level_msg = (
            "모델이 정상/불량을 애매하게 판단하고 있어, "
            "새로운 결함 유형 후보일 가능성이 있습니다."
        )
        is_new = True
    else:
        level_msg = (
            "모델 입장에서는 기존에 학습된 결함 유형과 다른 패턴을 보여, "
            "새로운 결함 유형일 가능성이 큽니다."
        )
        is_new = True

    # 오른쪽 패널용 요약
    summary_for_panel = (
        f"모델 기준 정상 확률은 {p_good * 100:.1f}%, "
        f"불량(비정상) 확률은 {p_defect * 100:.1f}%입니다. "
        f"{level_msg}"
    )

    # 채팅창에 찍어줄 한 줄 요약
    if is_new:
        chat_summary = (
            "3단계 분석 결과: 이 샘플은 기존 결함과는 다른 "
            "새로운 결함 유형일 가능성이 큰 것으로 판단됩니다. "
            "가운데 챗 패널에서 이 결함의 이름을 자유롭게 정해 주세요."
        )
    else:
        chat_summary = (
            "3단계 분석 결과: 이 샘플은 기존 정상/결함 패턴과 "
            "유사한 편으로 보입니다. 필요하다면 '기존 결함과 유사함'으로 "
            "표시해 둘 수 있습니다."
        )

    return {
        "ok": True,
        "category": category,
        "filename": file.filename,
        "pred_label": pred_label,
        "p_good": p_good,
        "p_defect": p_defect,
        "summary": summary_for_panel,
        "chat_summary": chat_summary,
        "is_new_defect_candidate": is_new,
    }


__all__ = ["run_adaptive"]



