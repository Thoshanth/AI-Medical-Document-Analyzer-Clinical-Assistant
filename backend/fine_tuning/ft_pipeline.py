import json
from pathlib import Path
from datetime import datetime
from backend.fine_tuning.dataset_generator import build_training_dataset
from backend.fine_tuning.trainer import run_fine_tuning
from backend.fine_tuning.evaluator import evaluate_model_responses
from backend.logger import get_logger

logger = get_logger("fine_tuning.pipeline")

FT_DATA_DIR = Path("fine_tuning_data")
STATUS_FILE = FT_DATA_DIR / "training_status.json"


def update_status(status: dict):
    """Saves training status to disk."""
    status["updated_at"] = datetime.utcnow().isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def get_training_status() -> dict:
    """Returns current training status."""
    if not STATUS_FILE.exists():
        return {
            "status": "not_started",
            "message": "No fine-tuning has been run yet.",
        }
    with open(STATUS_FILE) as f:
        return json.load(f)


def run_full_pipeline(
    document_ids: list[int] = None,
    num_epochs: int = 3,
    use_simulation: bool = True,
) -> dict:
    """
    Full fine-tuning pipeline:
    1. Generate dataset from templates + KB + documents
    2. Run QLoRA fine-tuning (or simulation)
    3. Evaluate model quality
    4. Save results

    use_simulation=True: No GPU needed (default for learning)
    use_simulation=False: Actual training (requires GPU)
    """
    logger.info(
        f"Fine-tuning pipeline | "
        f"docs={document_ids} | "
        f"epochs={num_epochs} | "
        f"simulation={use_simulation}"
    )

    update_status({
        "status": "generating_dataset",
        "message": "Building medical QA training dataset..."
    })

    # Step 1: Generate dataset
    dataset_info = build_training_dataset(
        document_ids=document_ids,
        include_templates=True,
        include_kb=True,
    )

    logger.info(f"Dataset ready | {dataset_info}")

    update_status({
        "status": "training",
        "message": f"Running QLoRA fine-tuning on {dataset_info['train_pairs']} examples...",
        "dataset": dataset_info,
    })

    # Step 2: Fine-tune
    training_result = run_fine_tuning(
        train_path=dataset_info["train_path"],
        num_epochs=num_epochs,
        use_simulation=use_simulation,
    )

    logger.info(f"Training complete | status={training_result.get('status')}")

    update_status({
        "status": "evaluating",
        "message": "Evaluating model medical knowledge...",
        "training": training_result,
    })

    # Step 3: Evaluate
    eval_results = evaluate_model_responses(
        model_name="minimax_m2.5_baseline"
    )

    # Final status
    final_status = {
        "status": "complete",
        "message": "Fine-tuning pipeline complete",
        "dataset": dataset_info,
        "training": training_result,
        "evaluation": {
            "average_score": eval_results["average_score"],
            "grade": eval_results["grade"],
            "model": eval_results["model_evaluated"],
        },
    }
    update_status(final_status)

    return final_status