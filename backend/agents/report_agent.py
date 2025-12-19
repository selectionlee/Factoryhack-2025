# backend/agents/report_agent.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
REPORT_DIR = STATIC_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 폰트 설정 (NanumGothic.ttf 가 있으면 사용, 없으면 기본 Helvetica) ----
FONT_DIR = PROJECT_ROOT / "fonts"
FONT_PATH = FONT_DIR / "NanumGothic.ttf"

if FONT_PATH.exists():
    pdfmetrics.registerFont(TTFont("NanumGothic", str(FONT_PATH)))
    BASE_FONT = "NanumGothic"
else:
    BASE_FONT = "Helvetica"


@dataclass
class DefectReportData:
    product_type: str
    file_name: str
    predicted_defect: Optional[str] = None
    stage1_summary: Optional[str] = None
    segmentation_summary: Optional[str] = None
    adapt_summary: Optional[str] = None
    llm_description: Optional[str] = None
    # ✅ 이미지 경로(URL) 추가
    orig_image_url: Optional[str] = None
    mask_image_url: Optional[str] = None


def _url_to_path(url: Optional[str]) -> Optional[Path]:
    """
    /static/seg_mask/xxx.png 같은 URL을 실제 파일 경로로 바꿔준다.
    """
    if not url:
        return None

    # 예: "/static/seg_mask/mask_xxx.png"
    if url.startswith("/static/"):
        rel = url[len("/static/") :]
        path = STATIC_DIR / rel
    else:
        path = PROJECT_ROOT / url.lstrip("/")

    return path if path.exists() else None


def _draw_multiline_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_chars: int = 60,
    line_height: float = 14,
):
    """
    아주 단순하게 '글자 수' 기준으로 줄바꿈.
    (원래 segmentation_summary 에 쓰던 방식과 거의 동일)
    """
    if not text:
        return y

    for i in range(0, len(text), max_chars):
        c.drawString(x, y, text[i : i + max_chars])
        y -= line_height
    return y


def create_defect_report(data: DefectReportData) -> Path:
    """
    DefectReportData를 받아 A4 한 장짜리 PDF 리포트를 생성하고 경로를 반환.
    (라운드 박스 / 회색 배경 제거, 보고서 스타일)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_DIR / f"defect_report_{timestamp}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    # --- 전체 흰색 배경 ---
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # 기본 마진
    margin_x = 50
    margin_y = 60
    content_width = width - margin_x * 2  # 필요하면 나중에 사용
    # content_height = height - margin_y * 2

    c.setFillColor(colors.black)
    c.setFont(BASE_FONT, 18)

    y = height - margin_y
    c.drawCentredString(width / 2, y, "Defect Inspection Report")
    y -= 35

    # ---- 메타 정보 ----
    c.setFont(BASE_FONT, 10)
    gen_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin_x, y, f"Generated at: {gen_str}")
    y -= 14
    c.drawString(margin_x, y, f"Product Type : {data.product_type}")
    y -= 14
    c.drawString(margin_x, y, f"Image File Name : {data.file_name}")
    y -= 14
    if data.predicted_defect:
        c.drawString(margin_x, y, f"Predicted Defect: {data.predicted_defect}")
    y -= 24

    max_chars = 60  # 한 줄에 대략 이 정도면 A4 폭 안에 안전하게 들어감

    # ---- Stage 1 ----
    c.setFont(BASE_FONT, 12)
    c.drawString(margin_x, y, "Stage 1 - Classification Summary")
    y -= 16
    c.setFont(BASE_FONT, 10)
    stage1_text = data.stage1_summary or "요약 정보가 없습니다."
    y = _draw_multiline_text(c, stage1_text, margin_x + 12, y, max_chars=max_chars)
    y -= 10

    # ---- Stage 2 ----
    c.setFont(BASE_FONT, 12)
    c.drawString(margin_x, y, "Stage 2 - Segmentation Summary")
    y -= 16
    c.setFont(BASE_FONT, 10)
    seg_text = data.segmentation_summary or "요약 정보가 없습니다."
    y = _draw_multiline_text(c, seg_text, margin_x + 12, y, max_chars=max_chars)
    y -= 10

    # ---- Stage 3 ----
    c.setFont(BASE_FONT, 12)
    c.drawString(margin_x, y, "Stage 3 - Adaptive Learning Summary")
    y -= 16
    c.setFont(BASE_FONT, 10)
    adapt_text = data.adapt_summary or "3단계 적응학습 요약 정보가 없습니다."
    y = _draw_multiline_text(c, adapt_text, margin_x + 12, y, max_chars=max_chars)

    # ---- 이미지 블록 (페이지 아래쪽에 2개) ----
    orig_path = _url_to_path(data.orig_image_url)
    mask_path = _url_to_path(data.mask_image_url)

    img_block_y = margin_y + 40
    img_max_w = 180
    img_max_h = 140
    gap_x = 40

    def _draw_one_image(x_left: float, label: str, img_path: Optional[Path]):
        nonlocal c
        c.setFont(BASE_FONT, 10)
        c.setFillColor(colors.black)
        c.drawString(x_left, img_block_y + img_max_h + 16, label)

        if not img_path:
            c.setFont(BASE_FONT, 9)
            c.drawString(x_left, img_block_y + img_max_h, "(이미지 없음)")
            return

        try:
            img_reader = ImageReader(str(img_path))
            iw, ih = img_reader.getSize()
            scale = min(img_max_w / iw, img_max_h / ih)
            dw = iw * scale
            dh = ih * scale
            c.drawImage(
                img_reader,
                x_left,
                img_block_y,
                width=dw,
                height=dh,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception as e:
            c.setFont(BASE_FONT, 9)
            c.drawString(x_left, img_block_y + img_max_h, f"(이미지 로드 실패: {e})")

    img_x1 = margin_x
    img_x2 = margin_x + img_max_w + gap_x

    _draw_one_image(img_x1, "Original / Overlay Image", orig_path)
    _draw_one_image(img_x2, "Defect Mask", mask_path)

    c.showPage()
    c.save()

    return pdf_path




