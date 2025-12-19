# backend/agents/chat_agent_graph.py
from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Optional

from fastapi import UploadFile
from langgraph.graph import StateGraph, END
from starlette.concurrency import run_in_threadpool

from agents.workflow_graph import (
    classify_node,
    segment_node,
    adapt_node,
)
from agents.llm_agent import chat_with_openai


class ChatState(TypedDict, total=False):
    """
    채팅 에이전트에서 사용하는 상태.
    - category / img_path / orig_filename : 이미지 관련 정보
    - cls_result / seg_result / adapt_result : 각 단계 결과
    - predicted_defect : 분류 단계에서 나온 결함 라벨
    - user_message : 사용자가 보낸 질문/요청
    - final_answer : LLM이 생성한 최종 답변
    """
    category: str
    img_path: str
    orig_filename: str

    cls_result: dict
    seg_result: dict
    adapt_result: dict

    predicted_defect: str

    user_message: str
    final_answer: str


# ---------------------- LLM 응답 노드 ----------------------
async def llm_answer_node(state: ChatState) -> ChatState:
    """
    분류/세그/적응학습 결과를 요약해서 LLM에게 전달하고,
    사용자의 질문(user_message)에 맞게 최종 답변을 생성.
    """
    user_msg = state.get("user_message", "")

    # 결과 요약 텍스트 구성
    summary_parts = []

    if "cls_result" in state:
        summary_parts.append(f"[1단계 분류 결과]\n{state['cls_result']}")
    if "seg_result" in state:
        summary_parts.append(f"[2단계 세그멘테이션 결과]\n{state['seg_result']}")
    if "adapt_result" in state:
        summary_parts.append(f"[3단계 적응학습 결과]\n{state['adapt_result']}")

    summary_text = "\n\n".join(summary_parts) if summary_parts else "아직 검사 결과는 없습니다."

    prompt = f"""
너는 공장 품질 검사 보조 에이전트야.
아래는 이미지에 대해 실행된 검사 결과 요약이야.

=== 시스템 요약 ===
{summary_text}

=== 사용자 질문 ===
{user_msg}

위 정보를 바탕으로,
1) 사용자가 이해하기 쉬운 자연어로 결함(또는 정상 여부)을 설명하고,
2) 세그멘테이션/적응학습이 수행되었다면 그 의미를 함께 설명하고,
3) 추가로 사용자가 어떤 액션을 취하면 좋을지 간단히 제안해줘.

가능하면 한국어로 답변해.
"""

    # 기존 chat_with_openai는 sync 함수이므로 스레드풀에서 실행
    reply = await run_in_threadpool(chat_with_openai, prompt)

    if isinstance(reply, dict) and "reply" in reply:
        answer = str(reply["reply"])
    else:
        answer = str(reply)

    state["final_answer"] = answer
    return state


# ---------------------- LangGraph 그래프 구성 ----------------------
def build_chat_graph():
    graph = StateGraph(ChatState)

    # 기존 workflow_graph의 노드 재사용
    graph.add_node("classify", classify_node)
    graph.add_node("segment", segment_node)
    graph.add_node("adapt", adapt_node)
    graph.add_node("llm_answer", llm_answer_node)

    # entry: classify → segment → adapt → llm_answer → END
    graph.set_entry_point("classify")
    graph.add_edge("classify", "segment")
    graph.add_edge("segment", "adapt")
    graph.add_edge("adapt", "llm_answer")
    graph.add_edge("llm_answer", END)

    return graph.compile()


chat_agent_graph = build_chat_graph()
