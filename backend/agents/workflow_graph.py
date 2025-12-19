# backend/agents/workflow_graph.py
from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Optional

from fastapi import UploadFile
from langgraph.graph import StateGraph, END

from agents.classifier_agent import classify_image
from agents.segmentation_agent import segment_defect
from agents.adaptive_agent import run_adaptive


class PipelineState(TypedDict, total=False):
    """
    LangGraph 파이프라인에서 공유할 상태.
    - category: 제품 타입 (capsule / tile / leather ...)
    - img_path: 서버에 저장된 이미지 경로 (str)
    - orig_filename: 업로드 시 원래 파일 이름
    - cls_result / seg_result / adapt_result: 각 단계의 원본 결과 dict
    - predicted_defect: 1단계에서 나온 결함 라벨 (seg/adapt가 참고)
    """
    category: str
    img_path: str
    orig_filename: str

    cls_result: dict
    seg_result: dict
    adapt_result: dict

    predicted_defect: str


def _make_upload_file(path: Path, filename: str) -> UploadFile:
    """
    저장된 이미지 경로를 FastAPI UploadFile 형태로 감싸주는 헬퍼.
    각 노드에서 기존 함수(classify_image, segment_defect, run_adaptive)를
    그대로 재사용하기 위해 사용한다.
    """
    f = path.open("rb")
    return UploadFile(filename=filename, file=f)


# ---------------------- Node 1: 분류 ----------------------
async def classify_node(state: PipelineState) -> PipelineState:
    img_path = Path(state["img_path"])
    upload = _make_upload_file(img_path, state["orig_filename"])

    result = await classify_image(state["category"], upload)
    state["cls_result"] = result
    state["predicted_defect"] = result.get("predicted_defect", "")

    return state


# ---------------------- Node 2: 위치+마스크 ----------------------
async def segment_node(state: PipelineState) -> PipelineState:
    img_path = Path(state["img_path"])
    upload = _make_upload_file(img_path, state["orig_filename"])

    result = await segment_defect(
        product_type=state["category"],
        pred_class=state.get("predicted_defect", ""),
        file=upload,
    )
    state["seg_result"] = result
    return state


# ---------------------- Node 3: 적응학습 ----------------------
async def adapt_node(state: PipelineState) -> PipelineState:
    img_path = Path(state["img_path"])
    upload = _make_upload_file(img_path, state["orig_filename"])

    result = await run_adaptive(
        category=state["category"],
        file=upload,
    )
    state["adapt_result"] = result
    return state


# ---------------------- LangGraph 그래프 구성 ----------------------
def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("classify", classify_node)
    graph.add_node("segment", segment_node)
    graph.add_node("adapt", adapt_node)

    # entry: classify → segment → adapt → END
    graph.set_entry_point("classify")
    graph.add_edge("classify", "segment")
    graph.add_edge("segment", "adapt")
    graph.add_edge("adapt", END)

    return graph.compile()


# 컴파일된 그래프 객체 (FastAPI에서 import해서 사용)
pipeline_graph = build_pipeline_graph()
