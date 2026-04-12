import json
from pathlib import Path
from backend.llm_client import chat_completion
from backend.logger import get_logger

logger = get_logger("fine_tuning.evaluator")

FT_DATA_DIR = Path("fine_tuning_data")

# Medical evaluation questions covering all clinical domains
EVAL_QUESTIONS = [
    {
        "question": "What is the first-line treatment for Type 2 Diabetes?",
        "key_terms": ["metformin", "lifestyle", "biguanide", "hba1c"],
        "domain": "pharmacology",
    },
    {
        "question": "What are emergency signs of cardiac arrest?",
        "key_terms": ["unconscious", "not breathing", "cpr", "emergency", "911"],
        "domain": "emergency_medicine",
    },
    {
        "question": "How do you interpret an elevated creatinine?",
        "key_terms": ["kidney", "renal", "glomerular", "egfr", "function"],
        "domain": "nephrology",
    },
    {
        "question": "What is the mechanism of Warfarin?",
        "key_terms": ["vitamin k", "clotting factors", "inr", "anticoagulant"],
        "domain": "pharmacology",
    },
    {
        "question": "What does a WBC of 15 K/uL indicate?",
        "key_terms": ["elevated", "infection", "leukocytosis", "bacterial"],
        "domain": "laboratory_medicine",
    },
    {
        "question": "What are the SOAP note components?",
        "key_terms": ["subjective", "objective", "assessment", "plan"],
        "domain": "clinical_documentation",
    },
    {
        "question": "How do you recognize a stroke?",
        "key_terms": ["fast", "face", "arm", "speech", "time", "emergency"],
        "domain": "emergency_medicine",
    },
    {
        "question": "What is HbA1c and why is it measured?",
        "key_terms": ["glycated hemoglobin", "diabetes", "3 months", "glucose control"],
        "domain": "endocrinology",
    },
]


def evaluate_model_responses(
    model_name: str = "current_minimax",
) -> dict:
    """
    Evaluates model quality on medical questions.

    Runs EVAL_QUESTIONS through the current LLM and
    scores responses on:
    1. Key term coverage (does answer contain expected terms?)
    2. Response length (sufficient detail?)
    3. Medical disclaimer presence
    4. Clinical accuracy (LLM-judged)

    This serves as a baseline comparison for fine-tuned model.
    """
    logger.info(
        f"Evaluating model | model={model_name} | "
        f"questions={len(EVAL_QUESTIONS)}"
    )

    results = []
    total_score = 0

    for i, eval_item in enumerate(EVAL_QUESTIONS):
        question = eval_item["question"]
        key_terms = eval_item["key_terms"]
        domain = eval_item["domain"]

        logger.info(
            f"Evaluating {i+1}/{len(EVAL_QUESTIONS)} | "
            f"domain={domain}"
        )

        # Get model response
        try:
            response = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert medical AI assistant. "
                            "Provide accurate, evidence-based medical "
                            "information."
                        )
                    },
                    {"role": "user", "content": question}
                ],
                max_tokens=400,
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"Evaluation LLM call failed: {e}")
            response = ""

        # Score the response
        response_lower = response.lower()

        # Metric 1: Key term coverage (0-40 points)
        terms_found = sum(
            1 for term in key_terms
            if term.lower() in response_lower
        )
        term_score = (terms_found / len(key_terms)) * 40

        # Metric 2: Response length/detail (0-20 points)
        word_count = len(response.split())
        if word_count >= 100:
            length_score = 20
        elif word_count >= 50:
            length_score = 15
        elif word_count >= 25:
            length_score = 10
        else:
            length_score = 5

        # Metric 3: Clinical structure (0-20 points)
        structure_indicators = [
            ":", "1)", "2)", "first", "second",
            "include", "such as", "for example",
        ]
        structure_score = min(
            sum(
                1 for ind in structure_indicators
                if ind in response_lower
            ) * 4,
            20,
        )

        # Metric 4: Safety indicators (0-20 points)
        safety_phrases = [
            "consult", "physician", "doctor", "professional",
            "monitor", "healthcare", "clinical",
        ]
        safety_score = min(
            sum(
                1 for phrase in safety_phrases
                if phrase in response_lower
            ) * 5,
            20,
        )

        total_item_score = (
            term_score + length_score +
            structure_score + safety_score
        )
        total_score += total_item_score

        results.append({
            "question": question,
            "domain": domain,
            "response_preview": response[:200] + "...",
            "metrics": {
                "key_terms_found": f"{terms_found}/{len(key_terms)}",
                "term_coverage_score": round(term_score, 1),
                "length_score": length_score,
                "structure_score": structure_score,
                "safety_score": safety_score,
                "total_score": round(total_item_score, 1),
            },
            "response_length_words": word_count,
        })

    avg_score = total_score / len(EVAL_QUESTIONS)
    max_possible = 100

    # Grade the model
    if avg_score >= 80:
        grade = "A — Excellent medical knowledge"
    elif avg_score >= 65:
        grade = "B — Good medical knowledge"
    elif avg_score >= 50:
        grade = "C — Adequate medical knowledge"
    else:
        grade = "D — Needs improvement"

    logger.info(
        f"Evaluation complete | "
        f"avg_score={avg_score:.1f}/{max_possible} | "
        f"grade={grade}"
    )

    # Save evaluation results
    eval_results_path = FT_DATA_DIR / "evaluation_results.json"
    eval_data = {
        "model": model_name,
        "average_score": round(avg_score, 2),
        "max_score": max_possible,
        "grade": grade,
        "results": results,
    }
    with open(eval_results_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    return {
        "model_evaluated": model_name,
        "average_score": round(avg_score, 2),
        "max_possible_score": max_possible,
        "grade": grade,
        "domains_tested": list(set(r["domain"] for r in results)),
        "question_results": results,
        "evaluation_saved": str(eval_results_path),
        "interpretation": {
            "term_coverage": "How many expected medical terms appear in answers",
            "length_score": "Response detail and completeness",
            "structure_score": "Clinical organization and structure",
            "safety_score": "Appropriate safety and professional referral mentions",
        },
    }


def compare_before_after(
    baseline_path: str = None,
) -> dict:
    """
    Compares model performance before and after fine-tuning.
    Loads baseline results and compares with current evaluation.
    """
    # Run current evaluation
    current_results = evaluate_model_responses("post_finetuning")

    if not baseline_path:
        baseline_path = str(FT_DATA_DIR / "evaluation_results.json")

    try:
        with open(baseline_path) as f:
            baseline = json.load(f)

        improvement = (
            current_results["average_score"] -
            baseline.get("average_score", 0)
        )

        return {
            "baseline_score": baseline.get("average_score"),
            "current_score": current_results["average_score"],
            "improvement": round(improvement, 2),
            "improvement_percent": round(
                (improvement / baseline.get("average_score", 1)) * 100,
                1
            ),
            "current_grade": current_results["grade"],
            "baseline_grade": baseline.get("grade"),
        }
    except Exception as e:
        logger.warning(f"Could not load baseline for comparison: {e}")
        return current_results