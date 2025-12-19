# backend/agents/llm_agent.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """
당신은 제조 라인의 품질 검사 도우미 챗봇입니다.
- 항상 한국어로 답변하세요.
- 작업자에게 친절하고 이해하기 쉽게 설명하세요.
- 그냥 일상적인 질문도 자연스럽게 받아주세요.
"""


def chat_with_openai(message: str) -> str:
    """
    입력 받은 message 하나만 가지고 LLM에게 물어보고,
    답변 텍스트만 돌려주는 아주 단순한 함수
    """
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",  # 계정에 따라 gpt-4o, gpt-4 등으로 바꿔도 됨
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=500,
            temperature=0.4,
        )

        return resp["choices"][0]["message"]["content"].strip()

    except Exception as e:
        # 프론트에서 오류 내용을 바로 볼 수 있게 반환
        return f"⚠️ LLM 오류 발생: {type(e).__name__} - {e}"

