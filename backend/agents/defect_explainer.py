# backend/main.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- 에이전트 함수들 ----
from agents.classifier_agent import classify_image
from agents.segmentation_agent import run_segmentation
from agents.adaptive_agent import run_adaptive
from agents.chat_agent import chat_with_llm

# ============================================================
# FastAPI 기본 설정
# ============================================================
app = FastAPI(title="Factory Q Agent Backend")

# CORS 허용 (프론트: http://localhost:3000 기준)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 정적 파일 (세그멘테이션 마스크 등)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ============================================================
# 1단계: 분류 엔드포인트
# ============================================================


@app.post("/api/classify")
async def api_classify(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """
    1단계: 정상/불량 및 클래스 분류
    frontend: runStage1 에서 호출
    """
    result = await classify_image(category=category, file=file)
    return result


# ============================================================
# 2단계: 위치 + 마스크 엔드포인트
# ============================================================


@app.post("/api/segment")
async def api_segment(
    product_type: str = Form(...),
    pred_class: str = Form(""),
    file: UploadFile = File(...),
):
    """
    2단계: SegFormer(or Deeplab 등) 기반 위치/마스크 생성
    frontend: runStage2 에서 호출
    """
    result = await run_segmentation(
        product_type=product_type,
        pred_class=pred_class,
        file=file,
    )
    return result


# ============================================================
# 3단계: 제외 결함 기반 적응학습 엔드포인트
# ============================================================


@app.post("/api/adapt")
async def api_adapt(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """
    3단계: 제외 결함 기반 적응학습 분석
    frontend: runStage3 에서 호출
    """
    result = await run_adaptive(category=category, file=file)
    return result


# ============================================================
# Chat LLM 엔드포인트
# ============================================================


class ChatPayload(BaseModel):
    message: str


@app.post("/api/chat")
async def api_chat(payload: ChatPayload):
    """
    LLM 대화 엔드포인트
    frontend: handleSend, 3단계 chip 클릭 시 등에서 사용
    """
    reply = await chat_with_llm(payload.message)
    return {"reply": reply}


# ============================================================
# 🔥 여기에는 PDF / defects 등록 관련 엔드포인트를 두지 않는다.
#    나중에 보고서 기능을 다시 붙이고 싶으면,
#    /api/defects/register, /api/defects/report 등을
#    새 파일(예: report_agent.py)에 깔끔히 분리해서 추가하면 됨.
# ============================================================

