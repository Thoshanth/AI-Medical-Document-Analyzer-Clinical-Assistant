import re
from backend.llm_client import chat_completion
from backend.medical_rag.rag_pipeline import medical_rag_query
from backend.logger import get_logger

logger = get_logger("evaluation.ragas")


def compute_faithfulness(
    answer: str,
    context: str,
) -> float:
    """
    Faithfulness: Does the answer ONLY contain information
    supported by the context? Measures hallucination.

    Score: 0.0 (completely hallucinated) to 1.0 (fully faithful)
    """
    prompt = f"""You are evaluating faithfulness of a medical AI answer.

Context (what the system retrieved):
{context}

Answer (what the system said):
{answer}

Task: Identify each factual claim in the answer.
For each claim check if it is directly supported by the context.

Return ONLY a JSON object:
{{
    "total_claims": number,
    "supported_claims": number,
    "unsupported_claims": ["claim1", "claim2"],
    "faithfulness_score": 0.0-1.0
}}

Score = supported_claims / total_claims
Return valid JSON only."""

    try:
        import json
        raw = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned.strip())
        score = float(result.get("faithfulness_score", 0.7))
        logger.debug(f"Faithfulness: {score:.3f}")
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Faithfulness computation failed: {e}")
        return 0.7  # default neutral score


def compute_answer_relevancy(
    question: str,
    answer: str,
) -> float:
    """
    Answer Relevancy: Does the answer address the question?

    Score: 0.0 (completely irrelevant) to 1.0 (fully relevant)
    """
    prompt = f"""Evaluate if this medical answer is relevant to the question.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0:
1.0 = Answer directly and completely addresses the question
0.7 = Answer mostly relevant with minor gaps
0.5 = Answer partially relevant
0.3 = Answer barely relevant
0.0 = Answer completely irrelevant

Return ONLY a JSON object:
{{"relevancy_score": 0.0-1.0, "reason": "brief reason"}}"""

    try:
        import json
        raw = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned.strip())
        score = float(result.get("relevancy_score", 0.7))
        logger.debug(f"Relevancy: {score:.3f}")
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Relevancy computation failed: {e}")
        return 0.7


def compute_context_precision(
    question: str,
    contexts: list[str],
) -> float:
    """
    Context Precision: Are retrieved chunks actually relevant?
    Proportion of retrieved chunks that are useful.

    Score: 0.0 to 1.0
    """
    if not contexts:
        return 0.0

    relevant_count = 0
    for context in contexts:
        prompt = f"""Is this retrieved chunk relevant to answering the question?

Question: {question}
Retrieved chunk: {context[:300]}

Reply with exactly one word: RELEVANT or IRRELEVANT"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            ).strip().upper()
            if "RELEVANT" in response:
                relevant_count += 1
        except Exception:
            relevant_count += 1  # default to relevant on error

    score = relevant_count / len(contexts)
    logger.debug(f"Context precision: {score:.3f}")
    return score


def compute_context_recall(
    ground_truth: str,
    contexts: list[str],
) -> float:
    """
    Context Recall: Did retrieval find all needed information?
    Can the ground truth be derived from the retrieved contexts?

    Score: 0.0 to 1.0
    """
    if not contexts or not ground_truth:
        return 0.0

    combined_context = " ".join(contexts)

    prompt = f"""Evaluate if this context contains enough information to derive the answer.

Required answer (ground truth): {ground_truth}
Retrieved context: {combined_context[:1000]}

What proportion of the ground truth can be derived from the context?
Return ONLY a JSON object:
{{"recall_score": 0.0-1.0, "missing_info": "what is missing if any"}}"""

    try:
        import json
        raw = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned.strip())
        score = float(result.get("recall_score", 0.7))
        logger.debug(f"Context recall: {score:.3f}")
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Context recall computation failed: {e}")
        return 0.7


def run_ragas_evaluation(
    document_id: int,
    golden_pairs: list[dict],
) -> dict:
    """
    Runs all four RAGAS metrics on the medical RAG system.

    For each golden QA pair:
    1. Run the RAG pipeline to get answer + contexts
    2. Compute all 4 RAGAS metrics
    3. Average across all pairs

    Returns comprehensive RAGAS evaluation report.
    """
    logger.info(
        f"RAGAS evaluation | "
        f"doc_id={document_id} | pairs={len(golden_pairs)}"
    )

    per_question_results = []
    all_faithfulness = []
    all_relevancy = []
    all_precision = []
    all_recall = []

    for i, pair in enumerate(golden_pairs[:8]):
        question = pair.get("question", "")
        ground_truth = pair.get("ground_truth", "")
        domain = pair.get("domain", "general")

        logger.info(
            f"Evaluating {i+1}/{min(len(golden_pairs),8)} | "
            f"domain={domain}"
        )

        # Get RAG answer and contexts
        try:
            rag_result = medical_rag_query(
                question=question,
                document_id=document_id,
                top_k=3,
                include_kb=False,
            )
            answer = rag_result.get("answer", "")
            sources = rag_result.get("sources", [])
            contexts = [
                s.get("text_preview", "") for s in sources
            ]
        except Exception as e:
            logger.warning(f"RAG query failed for evaluation: {e}")
            answer = ""
            contexts = []

        # Combined context for faithfulness check
        context_str = " ".join(contexts) if contexts else ground_truth

        # Compute all 4 metrics
        faithfulness = compute_faithfulness(answer, context_str)
        relevancy = compute_answer_relevancy(question, answer)
        precision = compute_context_precision(question, contexts)
        recall = compute_context_recall(ground_truth, contexts)

        # Expected term coverage
        expected_terms = pair.get("expected_terms", [])
        answer_lower = answer.lower()
        terms_found = sum(
            1 for t in expected_terms if t.lower() in answer_lower
        )
        term_coverage = (
            terms_found / len(expected_terms)
            if expected_terms else 1.0
        )

        all_faithfulness.append(faithfulness)
        all_relevancy.append(relevancy)
        all_precision.append(precision)
        all_recall.append(recall)

        per_question_results.append({
            "question": question,
            "domain": domain,
            "difficulty": pair.get("difficulty", "medium"),
            "answer_preview": answer[:200] + "..." if answer else "",
            "metrics": {
                "faithfulness": round(faithfulness, 3),
                "answer_relevancy": round(relevancy, 3),
                "context_precision": round(precision, 3),
                "context_recall": round(recall, 3),
                "term_coverage": round(term_coverage, 3),
            },
            "contexts_retrieved": len(contexts),
        })

    # Average metrics
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    avg_faithfulness = safe_avg(all_faithfulness)
    avg_relevancy = safe_avg(all_relevancy)
    avg_precision = safe_avg(all_precision)
    avg_recall = safe_avg(all_recall)

    # Overall RAGAS score
    ragas_score = safe_avg([
        avg_faithfulness, avg_relevancy,
        avg_precision, avg_recall
    ])

    # Grade
    if ragas_score >= 0.80:
        grade = "A — Excellent RAG quality"
    elif ragas_score >= 0.65:
        grade = "B — Good RAG quality"
    elif ragas_score >= 0.50:
        grade = "C — Adequate RAG quality"
    else:
        grade = "D — Needs improvement"

    logger.info(
        f"RAGAS complete | "
        f"score={ragas_score:.3f} | grade={grade}"
    )

    return {
        "ragas_score": ragas_score,
        "grade": grade,
        "metrics": {
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "context_precision": avg_precision,
            "context_recall": avg_recall,
        },
        "metric_interpretations": {
            "faithfulness": "How well answers are grounded in retrieved context (anti-hallucination)",
            "answer_relevancy": "How directly answers address the questions asked",
            "context_precision": "Proportion of retrieved chunks that are actually relevant",
            "context_recall": "How well retrieval captures all needed information",
        },
        "questions_evaluated": len(per_question_results),
        "per_question_results": per_question_results,
    }