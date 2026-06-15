from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.graph import chat

router = APIRouter(prefix="/chat", tags=["agent"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is Harness Engineering?"])
    session_id: str = Field(default="default", examples=["session1"])


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@router.post("/", response_model=ChatResponse)
async def ask_agent(payload: ChatRequest) -> ChatResponse:
    """Send a question to the production RAG agent.

    Pipeline: retrieve → filter irrelevant chunks → generate →
              verify grounding → evaluate quality → retry if needed.
    """
    answer = chat(payload.question, payload.session_id)
    return ChatResponse(answer=answer, session_id=payload.session_id)
