from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.chains import (
    GradeResult,
    grade_chain,
    hallucination_chain,
    rag_chain,
    relevance_chain,
    rewrite_chain,
)
from app.config import settings
from app.vectorstore import load_vectorstore, get_retriever


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: list[str]
    filtered_docs: list[str]
    current_question: str
    quality_score: int
    answer_grounded: bool
    retry_count: int


_vectorstore = load_vectorstore()
_retriever = get_retriever(_vectorstore)


def retrieve_node(state: AgentState) -> dict:
    docs = _retriever.invoke(state["current_question"])
    return {"retrieved_docs": [doc.page_content for doc in docs]}


def filter_docs_node(state: AgentState) -> dict:
    """Grade each chunk for relevance — drop irrelevant ones before generation."""
    question = state["current_question"]
    relevant = []
    for doc in state["retrieved_docs"]:
        # relevance_chain returns a dict (JsonOutputParser)
        result: dict = relevance_chain.invoke({"question": question, "document": doc})
        if result.get("is_relevant", False):
            relevant.append(doc)
        else:
            print(f"[filter] dropped: {doc[:60]}...")
    print(f"[filter] kept {len(relevant)}/{len(state['retrieved_docs'])} chunks")
    return {"filtered_docs": relevant}


def no_info_node(state: AgentState) -> dict:
    msg = (
        "I don't have enough information in my knowledge base to answer that. "
        "Try rephrasing, or add relevant documents and re-run ingest.py."
    )
    return {"messages": [AIMessage(content=msg)], "quality_score": 0}


def generate_node(state: AgentState) -> dict:
    context = "\n\n---\n\n".join(state["filtered_docs"])
    history = state["messages"][:-1]
    answer = rag_chain.invoke({
        "question": state["current_question"],
        "context": context,
        "history": history,
    })
    return {"messages": [AIMessage(content=answer)]}


def verify_answer_node(state: AgentState) -> dict:
    """Hallucination check — is the answer grounded in retrieved context?"""
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    context = "\n\n---\n\n".join(state["filtered_docs"])
    # hallucination_chain returns a dict (JsonOutputParser)
    result: dict = hallucination_chain.invoke({
        "context": context,
        "answer": last_ai.content,
    })
    grounded = result.get("grounded", False)
    print(f"[verify] grounded={grounded} — {result.get('explanation', '')}")
    return {"answer_grounded": grounded}


def evaluate_node(state: AgentState) -> dict:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    result: GradeResult = grade_chain.invoke({
        "question": state["current_question"],
        "answer": last_ai.content,
    })
    print(f"[evaluate] score={result.score}/10 — {result.reasoning}")
    return {"quality_score": result.score}


def rewrite_node(state: AgentState) -> dict:
    rewritten = rewrite_chain.invoke({
        "question": state["current_question"],
        "history": state["messages"],
    })
    print(f"[rewrite] '{state['current_question']}' -> '{rewritten}'")
    return {"current_question": rewritten, "retry_count": state["retry_count"] + 1}


def route_after_filter(state: AgentState) -> Literal["generate", "no_info"]:
    return "generate" if state["filtered_docs"] else "no_info"


def route_after_verify(state: AgentState) -> Literal["evaluate", "rewrite", "__end__"]:
    if state["answer_grounded"]:
        return "evaluate"
    if state["retry_count"] >= settings.max_retries:
        print("[route] not grounded but max retries reached -> END")
        return "__end__"
    print("[route] not grounded -> rewrite")
    return "rewrite"


def route_after_evaluate(state: AgentState) -> Literal["rewrite", "__end__"]:
    if state["quality_score"] >= settings.grade_threshold:
        print(f"[route] score {state['quality_score']} >= {settings.grade_threshold} -> END")
        return "__end__"
    if state["retry_count"] >= settings.max_retries:
        print("[route] max retries reached -> END anyway")
        return "__end__"
    print(f"[route] score {state['quality_score']} too low -> rewrite")
    return "rewrite"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve",      retrieve_node)
    builder.add_node("filter_docs",   filter_docs_node)
    builder.add_node("no_info",       no_info_node)
    builder.add_node("generate",      generate_node)
    builder.add_node("verify_answer", verify_answer_node)
    builder.add_node("evaluate",      evaluate_node)
    builder.add_node("rewrite",       rewrite_node)

    builder.add_edge(START,           "retrieve")
    builder.add_edge("retrieve",      "filter_docs")
    builder.add_edge("no_info",       END)
    builder.add_edge("generate",      "verify_answer")
    builder.add_edge("rewrite",       "retrieve")

    builder.add_conditional_edges(
        "filter_docs", route_after_filter,
        {"generate": "generate", "no_info": "no_info"},
    )
    builder.add_conditional_edges(
        "verify_answer", route_after_verify,
        {"evaluate": "evaluate", "rewrite": "rewrite", "__end__": END},
    )
    builder.add_conditional_edges(
        "evaluate", route_after_evaluate,
        {"__end__": END, "rewrite": "rewrite"},
    )

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


def chat(question: str, session_id: str) -> str:
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.recursion_limit,
    }
    final = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "current_question": question,
            "quality_score": 0,
            "retry_count": 0,
            "retrieved_docs": [],
            "filtered_docs": [],
            "answer_grounded": False,
        },
        config=config,
    )
    last_ai = next(m for m in reversed(final["messages"]) if isinstance(m, AIMessage))
    return last_ai.content
