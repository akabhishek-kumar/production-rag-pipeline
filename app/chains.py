"""LCEL chains for the production RAG pipeline.

Structured output note:
  grade_chain uses .with_structured_output() — works fine on Groq for int/str fields.
  relevance_chain and hallucination_chain use JsonOutputParser instead because
  Groq's llama-3.1-8b-instant fails tool_use for boolean fields in structured output.
  JsonOutputParser asks the LLM to return raw JSON and parses it as a Python dict.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from app.config import settings

llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
)


# ── 1. RAG chain ──────────────────────────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain enough information, "
        "say 'I don't have enough information to answer that.'\n\nContext:\n{context}",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()


# ── 2. Grade chain (structured output — works fine for int/str) ───────────────
class GradeResult(BaseModel):
    score: int = Field(description="Relevance and accuracy score 1-10", ge=1, le=10)
    reasoning: str = Field(description="One sentence explanation")

grade_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict quality evaluator. Rate the answer 1-10 for relevance "
        "and accuracy. 7+ means the answer fully addresses the question.",
    ),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nProvide score and reasoning."),
])
grade_chain = grade_prompt | llm.with_structured_output(GradeResult)


# ── 3. Rewrite chain ──────────────────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the question to be more specific and self-contained, resolving "
        "any pronouns using conversation history. Output ONLY the rewritten question.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Original question: {question}\n\nRewritten question:"),
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# ── 4. Relevance chain (JsonOutputParser — avoids Groq boolean tool_use bug) ──
# Returns a dict: {"is_relevant": true/false, "reason": "..."}
relevance_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        'You are a document relevance filter. Decide if the document helps answer '
        'the question. Respond ONLY with valid JSON, no explanation outside it.\n'
        'Format: {{"is_relevant": true, "reason": "one sentence"}}',
    ),
    ("human", "Question: {question}\n\nDocument: {document}"),
])
relevance_chain = relevance_prompt | llm | JsonOutputParser()


# ── 5. Hallucination chain (JsonOutputParser — same reason) ──────────────────
# Returns a dict: {"grounded": true/false, "explanation": "..."}
hallucination_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        'You are a hallucination detector. Determine if every claim in the answer '
        'is directly supported by the context. If the answer contains ANY information '
        'not in the context, it is NOT grounded. '
        'Respond ONLY with valid JSON, no explanation outside it.\n'
        'Format: {{"grounded": true, "explanation": "one sentence"}}',
    ),
    (
        "human",
        "Context:\n{context}\n\nAnswer: {answer}\n\nIs the answer fully grounded?",
    ),
])
hallucination_chain = hallucination_prompt | llm | JsonOutputParser()
