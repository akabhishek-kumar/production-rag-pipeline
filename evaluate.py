"""Evaluation script — measures RAG pipeline quality.

Run with: python evaluate.py

What it measures:
  1. Context Relevance  — are retrieved chunks actually about the question?
  2. Answer Faithfulness — does the answer stick to the retrieved context?
  3. Answer Quality      — does the answer fully address the question?

These are the three core RAGAS metrics, implemented here using the same
LLM chains used in production (grade_chain, relevance_chain, hallucination_chain).

Usage: add your own test cases to EVAL_DATASET below and run the script.
"""

from app.chains import GradeResult, RelevanceResult, HallucinationResult
from app.chains import grade_chain, relevance_chain, hallucination_chain
from app.vectorstore import load_vectorstore, get_retriever
from app.config import settings

# ── Evaluation dataset ────────────────────────────────────────────────────────
# Add question + expected_keywords pairs. The script checks whether the
# generated answer contains the expected keywords (basic recall check).

EVAL_DATASET = [
    {
        "question": "What is a UiPath Coded Agent?",
        "expected_keywords": ["LangGraph", "UiPath SDK", "code-first"],
    },
    {
        "question": "What does MCP stand for and what does it do?",
        "expected_keywords": ["Model Context Protocol", "tools", "bind_tools"],
    },
    {
        "question": "What are the five components of Harness Engineering?",
        "expected_keywords": ["Tool Registry", "Model Management", "Guardrails"],
    },
    {
        "question": "What is the difference between ingestion and query phase in RAG?",
        "expected_keywords": ["chunk", "embed", "retriev"],
    },
]


def evaluate_single(question: str, expected_keywords: list[str]) -> dict:
    """Run one question through retrieval + evaluation chains and score it."""
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)

    # Retrieve
    docs = retriever.invoke(question)
    doc_texts = [doc.page_content for doc in docs]
    context = "\n\n---\n\n".join(doc_texts)

    # 1. Context Relevance — what % of retrieved chunks are relevant?
    relevance_scores = []
    for doc in doc_texts:
        result: RelevanceResult = relevance_chain.invoke({
            "question": question,
            "document": doc,
        })
        relevance_scores.append(result.is_relevant)
    context_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    # Generate answer using only relevant chunks
    relevant_context = "\n\n---\n\n".join(
        doc for doc, rel in zip(doc_texts, relevance_scores) if rel
    )
    if not relevant_context:
        return {
            "question": question,
            "context_relevance": 0.0,
            "answer_faithfulness": False,
            "answer_quality": 0,
            "keyword_recall": 0.0,
            "answer": "No relevant context found.",
        }

    from app.chains import rag_chain
    answer = rag_chain.invoke({
        "question": question,
        "context": relevant_context,
        "history": [],
    })

    # 2. Answer Faithfulness — hallucination check
    hall_result: HallucinationResult = hallucination_chain.invoke({
        "context": relevant_context,
        "answer": answer,
    })

    # 3. Answer Quality — grade 1-10
    grade_result: GradeResult = grade_chain.invoke({
        "question": question,
        "answer": answer,
    })

    # 4. Keyword Recall — simple check for expected terms
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_recall = found / len(expected_keywords) if expected_keywords else 1.0

    return {
        "question": question,
        "context_relevance": round(context_relevance, 2),
        "answer_faithfulness": hall_result.grounded,
        "answer_quality": grade_result.score,
        "keyword_recall": round(keyword_recall, 2),
        "answer": answer,
    }


def main():
    print("\nRAG Pipeline Evaluation")
    print("=" * 60)

    results = []
    for i, item in enumerate(EVAL_DATASET, 1):
        print(f"\n[{i}/{len(EVAL_DATASET)}] {item['question']}")
        result = evaluate_single(item["question"], item["expected_keywords"])
        results.append(result)
        print(f"  Context Relevance  : {result['context_relevance']:.0%}")
        print(f"  Answer Faithfulness: {'PASS' if result['answer_faithfulness'] else 'FAIL'}")
        print(f"  Answer Quality     : {result['answer_quality']}/10")
        print(f"  Keyword Recall     : {result['keyword_recall']:.0%}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Avg Context Relevance  : {sum(r['context_relevance'] for r in results)/len(results):.0%}")
    print(f"  Avg Answer Quality     : {sum(r['answer_quality'] for r in results)/len(results):.1f}/10")
    print(f"  Faithfulness Pass Rate : {sum(r['answer_faithfulness'] for r in results)/len(results):.0%}")
    print(f"  Avg Keyword Recall     : {sum(r['keyword_recall'] for r in results)/len(results):.0%}")


if __name__ == "__main__":
    main()
