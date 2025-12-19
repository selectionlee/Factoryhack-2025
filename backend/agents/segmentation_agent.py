# backend/agents/segmentation_agent.py
from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import cv2
from PIL import Image

import torch
from fastapi import UploadFile
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# 🔹 LLM 사용해서 한글 설명 만들기
from .llm_agent import chat_with_openai


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEG_MODEL_DIR = PROJECT_ROOT / "segformer_finetuned_patches"

# DS-MVTec 원본 데이터(영문 설명 txt) 위치
DS_ROOT = PROJECT_ROOT / "DS-MVTec"

STATIC_ROOT = PROJECT_ROOT / "static"
OVERLAY_DIR = STATIC_ROOT / "seg_overlay"
MASK_DIR = STATIC_ROOT / "seg_mask"
UPLOAD_DIR = PROJECT_ROOT / "uploads_seg"

for d in [STATIC_ROOT, OVERLAY_DIR, MASK_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_seg_model = None
_seg_processor = None


def _load_segformer_once():
    global _seg_model, _seg_processor
    if _seg_model is not None:
        return _seg_model, _seg_processor

    _seg_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_DIR)
    _seg_model = SegformerForSemanticSegmentation.from_pretrained(
        SEG_MODEL_DIR
    ).to(DEVICE)
    _seg_model.eval()
    print("[SegFormer] loaded")
    return _seg_model, _seg_processor


def _overlay_mask(img_np, mask_np):
    red = np.zeros_like(img_np)
    red[:, :, 0] = 255

    blended = img_np.copy()
    blended[mask_np == 1] = (
        0.5 * img_np[mask_np == 1] + 0.5 * red[mask_np == 1]
    ).astype(np.uint8)
    return blended


def _run_segformer_on_image_path(image_path, save_vis_path=None, save_mask_path=None):
    model, processor = _load_segformer_once()

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    enc = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**enc)

    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(h, w)]
    )[0]
    seg_np = seg.cpu().numpy()
    pred_mask = (seg_np == 1).astype(np.uint8)

    img_np = np.array(image)
    blended = _overlay_mask(img_np, pred_mask)

    if save_vis_path:
        cv2.imwrite(str(save_vis_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    if save_mask_path:
        # 프론트엔드는 순수 마스크만 쓰고 싶어서 흑/백 대신 보라색 영역으로 저장
        mask_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        mask_rgb[pred_mask == 1] = (255, 0, 255)
        cv2.imwrite(str(save_mask_path), mask_rgb)

    return pred_mask


# ========================== 설명 매핑 부분 ==========================

def _read_text_with_fallback(path: Path) -> str | None:
    """여러 인코딩을 시도해서 txt를 읽는다."""
    if not path.exists():
        return None

    encodings = ["utf-8", "cp949", "euc-kr", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue

    # 그래도 안 되면 바이너리로 읽고 대충 디코드
    try:
        raw = path.read_bytes()
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _load_english_desc(product_type: str, defect_class: str, filename: str) -> str | None:
    """
    예: product_type='leather', defect_class='fold', filename='000.png' 일 때
    backend/DS-MVTec/leather/image/fold/000.txt 를 읽어온다.
    """
    stem = Path(filename).stem  # "000.png" -> "000"

    defect_dir = DS_ROOT / product_type / "image" / defect_class
    txt_path = defect_dir / f"{stem}.txt"

    # 1) 같은 번호 txt 먼저 시도
    if txt_path.exists():
        txt = _read_text_with_fallback(txt_path)
        if txt:
            print(f"[SEG] loaded desc txt: {txt_path}")
            return txt

    # 2) 안 되면 해당 defect 폴더에서 첫 번째 txt 아무거나 사용
    if defect_dir.exists():
        for p in sorted(defect_dir.glob("*.txt")):
            txt = _read_text_with_fallback(p)
            if txt:
                print(f"[SEG] fallback desc txt: {p}")
                return txt

    print(f"[SEG] no txt found for {product_type}/{defect_class}/{filename}")
    return None


def _make_korean_description(product_type: str, defect_class: str, filename: str) -> str:
    """
    txt 내용을 LLM으로 한국어 설명으로 변환.
    txt가 없으면 defect_class 만 가지고 대략적인 설명 생성.
    """
    eng = _load_english_desc(product_type, defect_class, filename)

    if eng:
        prompt = f"""
다음은 DS-MVTec 데이터셋에서 {product_type} 제품의 '{defect_class}' 결함 샘플에 대한 영어 설명입니다.

[영어 설명]
{eng}

위 내용을 바탕으로 작업자에게 보여줄 한국어 설명을 2~3문장으로 써 주세요.

- 제품이 어떤 재질/형태인지
- 결함이 이미지의 어디에, 어떤 모양/특징으로 나타나는지
- 너무 길지 않게 자연스럽게 요약해 주세요.
"""
    else:
        # txt 파일이 없을 때는 클래스 정보만으로 간단히 생성
        prompt = f"""
당신은 제조 라인의 품질 검사 도우미입니다.

'{product_type}' 제품에서 '{defect_class}' 라는 이름의 결함 클래스가 검출되었습니다.
이 결함이 어떤 느낌의 불량인지, 그리고 작업자가 어디를 주의해서 봐야 하는지
2~3문장 정도의 한국어 설명으로 알려주세요.

이미지 좌표는 주어지지 않았으니, 위치는 "이미지의 중앙 부분", "일부 영역" 등
일반적인 표현으로 설명해 주세요.
"""

    try:
        return chat_with_openai(prompt)
    except Exception as e:
        print("[SEG] LLM description error:", e)
        # 최악의 경우에도 빈 문자열 대신 짧은 기본 문구 반환
        return f"{product_type} 제품에서 '{defect_class}' 유형의 결함이 검출되었습니다. " \
               f"마스크가 표시된 영역을 중심으로 육안 검사를 진행해 주세요."


# ========================== FastAPI에서 호출 ==========================

async def segment_defect(product_type: str, pred_class: str, file: UploadFile):
    """
    - 업로드된 이미지를 저장
    - SegFormer로 마스크, overlay 생성
    - 해당 이미지 이름(000.png)에 맞는 txt를 찾아 LLM으로 한글 설명 생성
    """
    suffix = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex

    input_path = UPLOAD_DIR / f"input_{uid}{suffix}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    overlay_path = OVERLAY_DIR / f"overlay_{uid}.png"
    mask_path = MASK_DIR / f"mask_{uid}.png"

    _ = _run_segformer_on_image_path(
        image_path=input_path,
        save_vis_path=overlay_path,
        save_mask_path=mask_path,
    )

    # 🔹 파일 이름까지 이용해서 딱 맞는 txt → 한글 설명 (fallback 포함)
    description = _make_korean_description(
        product_type=product_type,
        defect_class=pred_class,
        filename=file.filename,
    )

    return {
        "ok": True,
        "product_type": product_type,
        "pred_class": pred_class,
        "overlay_url": f"/static/seg_overlay/{overlay_path.name}",
        "mask_url": f"/static/seg_mask/{mask_path.name}",
        "description": description,  # ← 프론트로 전달 (반드시 문자열)
    }





