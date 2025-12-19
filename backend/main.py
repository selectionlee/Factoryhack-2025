# backend/main.py
from __future__ import annotations

import uuid
import shutil
import re  # ✅ "등록해줘" 의도 감지용
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from agents.chat_agent_graph import chat_agent_graph
from agents.workflow_graph import pipeline_graph

from agents.llm_agent import chat_with_openai
from agents.classifier_agent import classify_image
from agents.segmentation_agent import segment_defect
from agents.adaptive_agent import run_adaptive
from agents.report_agent import DefectReportData, create_defect_report

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_UPLOAD_DIR = PROJECT_ROOT / "uploads"
PIPELINE_UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Factory Q Backend")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- static mount ----
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---- 통합 실행용 세션 저장소 (메모리) ----
#   session_id -> { category, img_path, orig_filename }
_pipeline_sessions: Dict[str, dict] = {}


# ============================================================
# 1단계 분류 /api/classify
# ============================================================
@app.post("/api/classify")
async def api_classify(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    result = await classify_image(category, file)
    return result


# ============================================================
# 2단계 위치+마스크 /api/segment
# ============================================================
@app.post("/api/segment")
async def api_segment(
    file: UploadFile = File(...),
    product_type: str = Form(...),
    pred_class: str = Form(""),
):
    result = await segment_defect(
        product_type=product_type,
        pred_class=pred_class,
        file=file,
    )
    return result


# ============================================================
# 3단계 적응학습 /api/adapt
# ============================================================
@app.post("/api/adapt")
async def api_adapt(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    result = await run_adaptive(category=category, file=file)
    return result


# ============================================================
# LangGraph 통합 파이프라인 (기존 one-shot) /api/pipeline
#   - 필요하면 그대로 두고, 새 세션 기반은 /api/pipeline/init 사용
# ============================================================
@app.post("/api/pipeline")
async def api_pipeline(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """
    1단계(분류) → 2단계(위치+마스크) → 3단계(적응학습)을
    LangGraph 파이프라인으로 한 번에 실행.
    """
    suffix = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex
    saved_path = PIPELINE_UPLOAD_DIR / f"pipe_{uid}{suffix}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    init_state = {
        "category": category,
        "img_path": str(saved_path),
        "orig_filename": file.filename,
    }

    final_state = await pipeline_graph.ainvoke(init_state)
    return final_state


# ============================================================
# 통합 실행용 세션 초기화 /api/pipeline/init
#   - 통합실행 버튼 눌렀을 때 한 번만 호출
#   - 이미지 + category를 저장하고 session_id를 돌려줌
# ============================================================
class PipelineInitResponse(BaseModel):
    session_id: str
    category: str
    img_path: str
    orig_filename: str


@app.post("/api/pipeline/init", response_model=PipelineInitResponse)
async def api_pipeline_init(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """
    통합 실행 모드 시작용 API.
    - 이미지와 category를 서버에 저장하고
    - 이후 채팅에서 사용할 session_id를 반환.
    """
    suffix = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex
    saved_path = PIPELINE_UPLOAD_DIR / f"session_{uid}{suffix}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    session_id = uid
    _pipeline_sessions[session_id] = {
        "category": category.lower(),
        "img_path": str(saved_path),
        "orig_filename": file.filename,
    }

    return PipelineInitResponse(
        session_id=session_id,
        category=category.lower(),
        img_path=str(saved_path),
        orig_filename=file.filename,
    )


# ============================================================
# 텍스트 전용 LLM Chat /api/chat
# ============================================================
class ChatPayload(BaseModel):
    message: str


@app.post("/api/chat")
async def api_chat(payload: ChatPayload):
    reply = await run_in_threadpool(chat_with_openai, payload.message)
    if isinstance(reply, dict) and "reply" in reply:
        return reply
    return {"reply": str(reply)}


# ============================================================
# LangGraph 기반 품질 검사 에이전트 채팅 /api/agent-chat
#   - 통합 실행 모드 + 적응학습 모드 둘 다에서 사용
#   - "등록해줘" 가 들어오면 LangGraph 안 돌리고 바로 응답
#   - 통합실행: 먼저 /api/pipeline/init → 이후 여기서 session_id 사용
# ============================================================
@app.post("/api/agent-chat")
async def api_agent_chat(
    message: str = Form(...),
    category: str = Form(""),                   # 세션 없을 때만 필요
    file: Optional[UploadFile] = File(None),    # 세션 없을 때만 필요
    session_id: Optional[str] = Form(None),     # 통합실행 세션 ID
):
    """
    - message : 사용자의 질문/요청
    - category : capsule / tile / leather 등 (session_id 없을 때만 필요)
    - file : 검사할 이미지 (세션 없을 때만 필요)
    - session_id : /api/pipeline/init 에서 받은 세션 ID
    """

    msg = message.strip()

    # ------------------ 0) 등록 의도 처리 (LangGraph 호출 X) ------------------
    if "등록" in msg:
        # 예: "crack 유형으로 등록해줘"
        m = re.search(r"([a-zA-Z0-9_가-힣]+)\s*유형으로\s*등록", msg)
        if m:
            label = m.group(1)
        else:
            label = "현재 결함"

        reply_text = f"네, '{label}' 결함 유형으로 등록하겠습니다."

        return {
            "reply": reply_text,
            "category": category or None,
            "predicted_defect": None,
            "cls_result": None,
            "seg_result": None,
            "adapt_result": None,
            "session_id": session_id,  # 프론트가 이미 들고 있을 수 있으니 그대로 반환
        }

    # ------------------ 1) 이미지 정보 결정 (session_id 우선) ------------------
    effective_category: str
    img_path: str
    orig_filename: str

    session = None
    if session_id:
        session = _pipeline_sessions.get(session_id)

    if session is not None:
        # ✅ 통합 실행 세션이 이미 있는 경우 → 저장된 이미지/카테고리 사용
        effective_category = session["category"]
        img_path = session["img_path"]
        orig_filename = session["orig_filename"]
    else:
        # 🔸 세션이 없으면 이번 요청에서 파일/카테고리로 새로 구성
        if file is None:
            raise HTTPException(
                status_code=400,
                detail="session_id가 없으면 file을 함께 보내야 합니다.",
            )
        if not category:
            raise HTTPException(
                status_code=400,
                detail="session_id가 없으면 category도 함께 보내야 합니다.",
            )

        suffix = Path(file.filename).suffix or ".png"
        uid = uuid.uuid4().hex
        saved_path = PIPELINE_UPLOAD_DIR / f"chat_{uid}{suffix}"

        with saved_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        effective_category = category.lower()
        img_path = str(saved_path)
        orig_filename = file.filename

        # 🔹 이 경우에도 새 세션으로 저장해 두면, 이후 채팅에서 session_id 사용 가능
        new_session_id = uid
        _pipeline_sessions[new_session_id] = {
            "category": effective_category,
            "img_path": img_path,
            "orig_filename": orig_filename,
        }
        session_id = new_session_id  # 응답으로 돌려주기 위해 갱신

    # ------------------ 2) LangGraph 전체 파이프라인 수행 ------------------
    init_state = {
        "category": effective_category,
        "img_path": img_path,
        "orig_filename": orig_filename,
        "user_message": msg,
    }

    final_state = await chat_agent_graph.ainvoke(init_state)

    return {
        "reply": final_state.get("final_answer", ""),
        "category": final_state.get("category", effective_category),
        "predicted_defect": final_state.get("predicted_defect"),
        "cls_result": final_state.get("cls_result"),
        "seg_result": final_state.get("seg_result"),
        "adapt_result": final_state.get("adapt_result"),
        "session_id": session_id,  # 🔹 프론트가 계속 들고 다니도록
    }


# ============================================================
# 결함 보고서용 엔드포인트
# ============================================================
class DefectRegisterPayload(BaseModel):
    product_type: str
    file_name: str
    predicted_defect: Optional[str] = None
    stage1_summary: Optional[str] = None
    segmentation_summary: Optional[str] = None
    adapt_summary: Optional[str] = None
    llm_description: Optional[str] = None
    orig_image_url: Optional[str] = None
    mask_image_url: Optional[str] = None


_last_report_data: Optional[DefectReportData] = None


@app.post("/api/defects/register")
async def api_defects_register(payload: DefectRegisterPayload):
    global _last_report_data

    _last_report_data = DefectReportData(
        product_type=payload.product_type,
        file_name=payload.file_name,
        predicted_defect=payload.predicted_defect,
        stage1_summary=payload.stage1_summary,
        segmentation_summary=payload.segmentation_summary,
        adapt_summary=payload.adapt_summary,
        llm_description=payload.llm_description,
        orig_image_url=payload.orig_image_url,
        mask_image_url=payload.mask_image_url,
    )

    return {"ok": True}


@app.get("/api/defects/report")
async def api_defects_report():
    if _last_report_data is None:
        raise HTTPException(
            status_code=400,
            detail="등록된 결함 요약 데이터가 없습니다. 먼저 /api/defects/register 를 호출해 주세요.",
        )

    pdf_path = create_defect_report(_last_report_data)
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )












