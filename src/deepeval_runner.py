"""
DeepEval — RAG pipeline evaluation using open-source metrics.
Apache 2.0 license. No API key needed for local metrics.

Evaluates:
  - Answer Relevancy  : does the answer address the question?
  - Faithfulness      : is the answer grounded in the retrieved context?
  - Contextual Recall : does the context contain enough info to answer?
"""

import os
from typing import Optional


def _check_deepeval():
    try:
        import deepeval
        return True
    except ImportError:
        return False


def evaluate_rag_response(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    expected_answer: Optional[str] = None,
    use_local: bool = True,
) -> dict:
    """
    Evaluate a RAG response using DeepEval metrics.

    Args:
        question: The user's original question.
        answer: The LLM's generated answer.
        retrieved_contexts: List of context chunks retrieved by RAG.
        expected_answer: Optional ground-truth answer for correctness check.
        use_local: If True, use local non-LLM metrics (fast, no API).

    Returns:
        dict with metric scores and pass/fail flags.
    """
    if not _check_deepeval():
        raise ImportError("Run: pip install deepeval")

    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieved_contexts,
        expected_output=expected_answer or answer,
    )

    results = {}

    metrics = [
        ("answer_relevancy", AnswerRelevancyMetric(threshold=0.5, model="local" if use_local else None)),
        ("faithfulness",     FaithfulnessMetric(threshold=0.5,     model="local" if use_local else None)),
    ]

    if expected_answer:
        metrics.append(
            ("contextual_recall", ContextualRecallMetric(threshold=0.5, model="local" if use_local else None))
        )

    for metric_name, metric in metrics:
        try:
            metric.measure(test_case)
            results[metric_name] = {
                "score": round(float(metric.score), 3),
                "passed": metric.is_successful(),
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            results[metric_name] = {"score": None, "passed": None, "reason": str(e)}

    return results


def batch_evaluate(test_cases: list[dict], use_local: bool = True) -> list[dict]:
    """
    Evaluate multiple RAG test cases.
    Each dict: {question, answer, contexts, expected_answer (optional)}
    """
    all_results = []
    for tc in test_cases:
        result = evaluate_rag_response(
            question=tc["question"],
            answer=tc["answer"],
            retrieved_contexts=tc.get("contexts", []),
            expected_answer=tc.get("expected_answer"),
            use_local=use_local,
        )
        all_results.append({**tc, "metrics": result})
    return all_results


def format_eval_report(results: dict) -> str:
    """Format evaluation results as a readable string."""
    lines = ["📊 RAG Evaluation Report", "=" * 40]
    for metric, data in results.items():
        score = data.get("score")
        passed = data.get("passed")
        reason = data.get("reason", "")
        status = "✅" if passed else ("❌" if passed is False else "⚠️")
        score_str = f"{score:.3f}" if score is not None else "N/A"
        lines.append(f"{status} {metric.replace('_', ' ').title()}: {score_str}")
        if reason:
            lines.append(f"   ↳ {reason[:120]}")
    return "\n".join(lines)
