import json
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("fine_tuning.trainer")

FT_DATA_DIR = Path("fine_tuning_data")
ADAPTERS_DIR = FT_DATA_DIR / "adapters"
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BioMistral/BioMistral-7B"
MAX_SEQ_LENGTH = 2048


def format_alpaca_prompt(example: dict) -> str:
    """
    Formats a QA pair into Alpaca instruction format.
    This is the standard fine-tuning prompt template.
    """
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def run_fine_tuning(
    train_path: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    use_simulation: bool = True,
) -> dict:
    """
    Runs QLoRA fine-tuning on BioMistral-7B.

    use_simulation=True: Simulates training without GPU
    (shows what would happen, generates training metrics)

    use_simulation=False: Actual fine-tuning with Unsloth
    (requires GPU with 8GB+ VRAM)
    """
    if use_simulation:
        return _simulate_fine_tuning(
            train_path, num_epochs, learning_rate
        )
    else:
        return _run_actual_fine_tuning(
            train_path, num_epochs, learning_rate, batch_size
        )


def _simulate_fine_tuning(
    train_path: str,
    num_epochs: int,
    learning_rate: float,
) -> dict:
    """
    Simulates fine-tuning process without GPU.
    Shows training configuration and expected metrics.
    Perfect for demonstrating understanding of fine-tuning
    without requiring expensive hardware.
    """
    import time
    import random

    logger.info("Running fine-tuning SIMULATION (no GPU required)")

    # Load dataset
    with open(train_path) as f:
        train_data = json.load(f)

    num_samples = len(train_data)
    steps_per_epoch = num_samples // 4
    total_steps = steps_per_epoch * num_epochs

    logger.info(
        f"Simulation config | "
        f"samples={num_samples} | "
        f"epochs={num_epochs} | "
        f"steps={total_steps}"
    )

    # Simulate training progress
    training_log = []
    loss = 2.5
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        for step in range(1, steps_per_epoch + 1):
            # Simulate loss decrease with noise
            loss = loss * 0.97 + random.uniform(-0.02, 0.01)
            loss = max(loss, 0.3)

            if step % (steps_per_epoch // 3) == 0:
                training_log.append({
                    "epoch": epoch,
                    "step": (epoch - 1) * steps_per_epoch + step,
                    "loss": round(loss, 4),
                    "learning_rate": learning_rate,
                })
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Step {step}/{steps_per_epoch} | "
                    f"Loss: {loss:.4f}"
                )

        time.sleep(0.5)  # Simulate processing time

    elapsed = round(time.time() - start_time, 2)
    final_loss = training_log[-1]["loss"] if training_log else 0.45

    # Save simulated adapter info
    adapter_path = ADAPTERS_DIR / "simulated_adapter"
    adapter_path.mkdir(exist_ok=True)

    config = {
        "model_name": MODEL_NAME,
        "fine_tuning_type": "QLoRA (4-bit quantization)",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": [
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        "training_samples": num_samples,
        "num_epochs": num_epochs,
        "batch_size": 4,
        "learning_rate": learning_rate,
        "max_seq_length": MAX_SEQ_LENGTH,
        "final_loss": final_loss,
        "simulation": True,
    }

    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Simulation complete | "
        f"final_loss={final_loss:.4f} | "
        f"time={elapsed}s"
    )

    return {
        "status": "simulation_complete",
        "model": MODEL_NAME,
        "fine_tuning_type": "QLoRA (4-bit quantization + LoRA adapters)",
        "training_samples": num_samples,
        "num_epochs": num_epochs,
        "final_loss": final_loss,
        "initial_loss": 2.5,
        "loss_reduction": round(
            ((2.5 - final_loss) / 2.5) * 100, 1
        ),
        "training_log": training_log,
        "adapter_saved": str(adapter_path),
        "simulation_time_seconds": elapsed,
        "note": (
            "This is a simulation. For actual fine-tuning: "
            "run with use_simulation=False on a GPU with 8GB+ VRAM. "
            "Install: pip install unsloth bitsandbytes"
        ),
        "what_would_happen": {
            "step_1": f"Load {MODEL_NAME} in 4-bit quantization (QLoRA)",
            "step_2": "Attach LoRA adapters to attention layers",
            "step_3": f"Train for {num_epochs} epochs on {num_samples} medical QA pairs",
            "step_4": "Save LoRA adapter weights (~50MB vs 14GB full model)",
            "step_5": "Fine-tuned model answers medical questions like a specialist",
        },
    }


def _run_actual_fine_tuning(
    train_path: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
) -> dict:
    """
    Actual QLoRA fine-tuning using Unsloth.
    Requires GPU with 8GB+ VRAM.
    """
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        logger.error(
            "Unsloth not installed. Run: pip install unsloth bitsandbytes"
        )
        return {
            "status": "failed",
            "error": "Unsloth not installed. Use use_simulation=True or install unsloth.",
        }

    logger.info(f"Loading model | {MODEL_NAME}")

    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and format dataset
    with open(train_path) as f:
        train_data = json.load(f)

    formatted = [
        {"text": format_alpaca_prompt(ex)} for ex in train_data
    ]
    dataset = Dataset.from_list(formatted)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(ADAPTERS_DIR / "biomistral_medical"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    logger.info("Starting actual fine-tuning")
    trainer.train()

    # Save adapter weights
    adapter_path = ADAPTERS_DIR / "biomistral_medical_final"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    logger.info(f"Fine-tuning complete | adapter saved: {adapter_path}")

    return {
        "status": "complete",
        "model": MODEL_NAME,
        "adapter_path": str(adapter_path),
        "training_samples": len(train_data),
        "num_epochs": num_epochs,
    }