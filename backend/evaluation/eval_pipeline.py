import json
from pathlib import Path
from datetime import datetime
from backend.evaluation.golden_dataset import build_golden_dataset
from backend.evaluation.ragas_evaluator import run_ragas_evaluation
from backend.evaluation.medical_metrics import (
    evaluate_emergency_detection,
    evaluate_disclaimer_compliance,
    evaluate_clinical_accuracy,
    evaluate_medication_safety_language,
)
from backend.medical_rag.rag_pipeline import medical_rag_query
from backend.logger import get_logger

logger = get_logger("evaluation.pipeline")

EVAL_DIR = Path("evaluation_results")
EVAL_DIR.mkdir(exist_ok=True)


def run_full_evaluation(document_id: int) -> dict:
    """
    Complete medical AI evaluation pipeline.

    Phase 1 — Build evaluation dataset
    Phase 2 — RAGAS metrics (faithfulness, relevancy, precision, recall)
    Phase 3 — Medical metrics (emergency detection, accuracy, safety)
    Phase 4 — Generate overall report with pass/fail per metric

    Returns comprehensive evaluation report with grades.
    """
    logger.info(f"Full evaluation | doc_id={document_id}")
    eval_time = datetime.utcnow().isoformat()

    # Phase 1: Build golden dataset
    logger.info("Phase 1: Building golden dataset")
    dataset_info = build_golden_dataset(
        document_id=document_id,
        include_static=True,
    )

    golden_path = (
        EVAL_DIR /
        f"golden_dataset_doc{document_id}.json"
    )
    with open(golden_path) as f:
        golden_pairs = json.load(f)

    # Phase 2: RAGAS evaluation
    logger.info("Phase 2: Running RAGAS evaluation")
    ragas_results = {}
    try:
        ragas_results = run_ragas_evaluation(document_id, golden_pairs[:6])
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        ragas_results = {"error": str(e), "ragas_score": 0}

    # Phase 3: Collect system answers for medical metrics
    logger.info("Phase 3: Collecting system answers")
    system_answers = []
    for pair in golden_pairs[:5]:
        try:
            result = medical_rag_query(
                question=pair["question"],
                document_id=document_id,
                top_k=3,
                include_kb=False,
            )
            system_answers.append(result.get("answer", ""))
        except Exception as e:
            logger.warning(f"Answer collection failed: {e}")
            system_answers.append("")

    # Phase 3a: Emergency detection
    logger.info("Phase 3a: Emergency detection evaluation")
    emergency_results = evaluate_emergency_detection()

    # Phase 3b: Disclaimer compliance
    logger.info("Phase 3b: Disclaimer compliance")
    disclaimer_results = evaluate_disclaimer_compliance(
        system_answers
    )

    # Phase 3c: Clinical accuracy
    logger.info("Phase 3c: Clinical accuracy")
    accuracy_results = {}
    if system_answers and golden_pairs:
        accuracy_results = evaluate_clinical_accuracy(
            golden_pairs[:len(system_answers)],
            system_answers,
        )

    # Phase 3d: Medication safety language
    logger.info("Phase 3d: Medication safety language")
    med_safety_results = evaluate_medication_safety_language(
        system_answers
    )

    # Phase 4: Overall report
    logger.info("Phase 4: Generating overall report")

    # Calculate overall score
    scores = []
    if ragas_results.get("ragas_score"):
        scores.append(ragas_results["ragas_score"])
    if emergency_results.get("accuracy"):
        scores.append(emergency_results["accuracy"])
    if accuracy_results.get("average_clinical_accuracy"):
        scores.append(accuracy_results["average_clinical_accuracy"])

    overall_score = (
        round(sum(scores) / len(scores), 3) if scores else 0
    )

    # Pass/fail assessment
    passing_criteria = {
        "ragas_score": ragas_results.get("ragas_score", 0) >= 0.60,
        "emergency_sensitivity": emergency_results.get("sensitivity", 0) >= 0.95,
        "disclaimer_compliance": disclaimer_results.get("pass", False),
        "clinical_accuracy": accuracy_results.get("pass", False),
        "medication_safety": med_safety_results.get("pass", False),
    }

    all_passing = all(passing_criteria.values())

    # Overall grade
    if overall_score >= 0.85:
        grade = "A — Production Ready"
    elif overall_score >= 0.70:
        grade = "B — Near Production Ready"
    elif overall_score >= 0.55:
        grade = "C — Needs Improvement"
    else:
        grade = "D — Significant Issues"

    report = {
        "document_id": document_id,
        "evaluated_at": eval_time,
        "overall_score": overall_score,
        "overall_grade": grade,
        "production_ready": all_passing,
        "passing_criteria": passing_criteria,
        "dataset_info": dataset_info,
        "ragas_metrics": ragas_results,
        "medical_metrics": {
            "emergency_detection": emergency_results,
            "disclaimer_compliance": disclaimer_results,
            "clinical_accuracy": accuracy_results,
            "medication_safety_language": med_safety_results,
        },
        "recommendations": _generate_recommendations(
            ragas_results, emergency_results,
            disclaimer_results, accuracy_results,
        ),
    }

    # Save report
    report_path = EVAL_DIR / f"eval_report_doc{document_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    report["saved_to"] = str(report_path)
    logger.info(
        f"Evaluation complete | "
        f"score={overall_score} | "
        f"grade={grade} | "
        f"production_ready={all_passing}"
    )

    return report


def _generate_recommendations(
    ragas: dict,
    emergency: dict,
    disclaimer: dict,
    accuracy: dict,
) -> list[str]:
    """Generates actionable improvement recommendations."""
    recommendations = []

    if ragas.get("ragas_score", 1) < 0.70:
        if ragas.get("metrics", {}).get("faithfulness", 1) < 0.70:
            recommendations.append(
                "Improve faithfulness: Add stricter context grounding prompts "
                "to prevent hallucination"
            )
        if ragas.get("metrics", {}).get("context_recall", 1) < 0.70:
            recommendations.append(
                "Improve context recall: Increase top_k retrieval or "
                "improve chunking strategy"
            )
        if ragas.get("metrics", {}).get("context_precision", 1) < 0.70:
            recommendations.append(
                "Improve context precision: Add metadata filtering to "
                "exclude irrelevant chunks"
            )

    if emergency.get("sensitivity", 1) < 0.95:
        recommendations.append(
            "CRITICAL: Emergency detection sensitivity below 0.95. "
            "Add more emergency keyword patterns immediately."
        )

    if not disclaimer.get("pass", True):
        recommendations.append(
            "Disclaimer compliance failure: Ensure all responses "
            "include medical disclaimer injection"
        )

    if accuracy.get("average_clinical_accuracy", 1) < 0.70:
        recommendations.append(
            "Low clinical accuracy: Consider fine-tuning (Stage 10) "
            "on more medical QA pairs"
        )

    if not recommendations:
        recommendations.append(
            "All metrics passing. System is performing well. "
            "Continue monitoring with regular evaluation runs."
        )

    return recommendations


def get_saved_results(document_id: int) -> dict | None:
    """Retrieves previously saved evaluation results."""
    path = EVAL_DIR / f"eval_report_doc{document_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)