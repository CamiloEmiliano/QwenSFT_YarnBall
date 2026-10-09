"""
Supervised Fine-Tuning (SFT) Engine for Qwen2.5-7B-Instruct.

Trains a unified financial intelligence model on the YarnBall SFT dataset spanning:
- Task A: SEC OpenCypher Triples DSL
- Task B: Breaking News Event Edges
- Task C: Text-to-Cypher
- Task D: Contagion Reasoning (<think>)
- Task E: Portfolio Hedging Recommendations (<think>)

Enforces completion-only loss masking: cross-entropy loss is computed strictly
over completion tokens (after '<|im_start|>assistant\\n').
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import yaml
from dotenv import load_dotenv

# Deep Learning & Hugging Face imports (graceful fallback for local test runners)

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
except ImportError:
    torch = None
    load_dataset = None
    LoraConfig = None
    TaskType = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainerCallback = None
    DataCollatorForCompletionOnlyLM = None
    SFTConfig = None
    SFTTrainer = None


# Load local environment if present
load_dotenv()

# Automatically sync Hugging Face token if using YarnBall environment naming
if "HF_TOKEN" not in os.environ and "HUGGINGFACE_FULL_ACCESS_TOKEN_01" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_FULL_ACCESS_TOKEN_01"]


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_attn_implementation() -> str:
    """Check if FlashAttention-2 is available, otherwise fall back to SDPA."""
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability()[0]
        if major_cc >= 8:  # Ampere (A100, RTX 30xx/40xx), Hopper (H100), Ada (A6000)
            try:
                import flash_attn  # noqa: F401
                print("FlashAttention-2 detected and enabled.")
                return "flash_attention_2"
            except ImportError:
                print("PyTorch SDPA (Scaled Dot-Product Attention) enabled (flash_attn package not installed).")
                return "sdpa"
    return "sdpa"


def format_dataset_to_chatml(batch: dict) -> dict:
    """Format prompt-completion pairs into Qwen2.5 ChatML format."""
    formatted_texts = []
    # Dataset schema uses 'target_completion' (with fallback to 'completion')
    completions = batch.get("target_completion") or batch.get("completion") or []
    for p, c in zip(batch["prompt"], completions):
        # Qwen2.5 native ChatML conversation format
        chatml_text = (
            f"<|im_start|>user\n{p.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{c.strip()}<|im_end|>"
        )
        formatted_texts.append(chatml_text)
    return {"text": formatted_texts}


class TrainingDynamicsCallback(TrainerCallback if TrainerCallback is not None else object):
    """
    Ripped and adapted from AllenAI's selection_utils.py:log_training_dynamics.
    Records per-sample sequence probabilities and correctness across training epochs.
    Computes sequence probability: exp(-mean_token_loss) over non-masked target tokens (label != -100).
    """

    def __init__(
        self,
        output_path: str = "artifacts/training_dynamics.jsonl",
        correctness_threshold: float = 0.70,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.correctness_threshold = correctness_threshold
        # Clear previous run log if starting fresh
        if self.output_path.exists():
            self.output_path.unlink()

    def on_epoch_end(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        """Calculates per-sample target token log-likelihood on holdout/train probe."""
        if eval_dataloader is None or model is None or torch is None:
            return

        model.eval()
        epoch = int(round(state.epoch)) if state.epoch is not None else 1
        records = []
        batch_sample_offset = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                guids = batch.get("guid") or batch.get("sample_id")
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)

                batch_size = input_ids.size(0)
                if guids is None:
                    guids = [f"sample_{batch_sample_offset + i}" for i in range(batch_size)]
                    batch_sample_offset += batch_size

                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits  # [B, T, V]

                # Compute per-sample loss over non-masked target tokens (label != -100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())

                mask = (shift_labels != -100).float()
                token_counts = mask.sum(dim=1).clamp(min=1.0)
                per_sample_loss = (loss * mask).sum(dim=1) / token_counts

                for guid, sample_loss in zip(guids, per_sample_loss.cpu().tolist()):
                    seq_prob = math.exp(-sample_loss)
                    is_correct = seq_prob >= self.correctness_threshold
                    records.append({
                        "epoch": epoch,
                        "guid": str(guid),
                        "seq_prob": round(seq_prob, 5),
                        "is_correct": is_correct,
                    })

        with open(self.output_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"[TrainingDynamics] Logged dynamics for {len(records)} samples at epoch {epoch} -> {self.output_path}")
        model.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train YarnBall Qwen2.5-7B LoRA SFT Model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--base-model-name", default=None, help="Base model name or path (overrides config)")
    parser.add_argument("--dataset-id", default=None, help="Hugging Face Dataset ID (overrides config)")
    parser.add_argument("--dataset-revision", default=None, help="Dataset git revision/tag on Hub")
    parser.add_argument("--output-dir", default=None, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Run 2-step verification run on 20 samples")
    parser.add_argument("--no-push", action="store_true", help="Do not push weights to Hugging Face Hub")
    return parser.parse_args()


def main():
    if torch is None or AutoModelForCausalLM is None:
        raise ImportError(
            "Deep learning dependencies (torch, transformers, trl, peft) are not installed in this environment. "
            "Install them via 'pip install -r requirements.txt' or run inside the Vast.ai GPU container."
        )

    args = parse_args()
    cfg = load_yaml_config(args.config)


    # Resolve settings from config + CLI args
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})

    base_model_name = args.base_model_name or model_cfg.get("base_model_name", "Qwen/Qwen2.5-7B-Instruct")
    dataset_id = args.dataset_id or ds_cfg.get("repo_id", "CamiloEmiliano/yarnball-sft")
    dataset_revision = args.dataset_revision or ds_cfg.get("revision", None)
    output_dir = args.output_dir or train_cfg.get("output_dir", "./outputs/yarnball-qwen-7b-lora")
    hub_model_id = train_cfg.get("hub_model_id", "CamiloEmiliano/yarnball-qwen-7b-lora")
    push_to_hub = (not args.no_push) and train_cfg.get("push_to_hub", True)

    epochs = args.epochs or train_cfg.get("num_train_epochs", 3)
    batch_size = args.batch_size or train_cfg.get("per_device_train_batch_size", 4)
    lr = args.lr or float(train_cfg.get("learning_rate", 2e-4))
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    print("\n" + "=" * 70)
    print("YARNBALL SFT TRAINING ENGINE - QWEN2.5-7B-INSTRUCT")
    print("=" * 70)
    print(f"Base Model:       {base_model_name}")
    print(f"Dataset ID:       {dataset_id} (Revision: {dataset_revision or 'main'})")
    print(f"Output Directory: {output_dir}")
    print(f"Push to Hub:      {push_to_hub} ({hub_model_id})")
    print(f"Dry Run Mode:     {args.dry_run}")
    print("=" * 70 + "\n")

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset from Hugging Face Hub
    print(f"Loading dataset from Hugging Face Hub: {dataset_id}...")
    dataset = load_dataset(dataset_id, revision=dataset_revision)

    train_data = dataset[ds_cfg.get("split_train", "train")]
    val_data = dataset[ds_cfg.get("split_val", "validation")]

    if args.dry_run:
        print("[DRY-RUN] Subsampling 20 training records and 5 validation records for smoke test...")
        train_data = train_data.select(range(min(20, len(train_data))))
        val_data = val_data.select(range(min(5, len(val_data))))
        epochs = 1

    # Format into ChatML texts
    print("Formatting prompts into Qwen ChatML...")
    train_data = train_data.map(format_dataset_to_chatml, batched=True)
    val_data = val_data.map(format_dataset_to_chatml, batched=True)

    # 3. Setup Completion-Only Data Collator
    # Masks prompt tokens with label -100 so loss is computed strictly after the delimiter
    response_template = "<|im_start|>assistant\n"
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_token_ids,
        tokenizer=tokenizer,
    )

    # 4. Determine Precision & Attention Implementation
    attn_impl = get_attn_implementation()
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading base model in {torch_dtype} with {attn_impl}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    # 5. Configure LoRA Adapter
    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 6. SFT Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs if not args.dry_run else 1,
        max_steps=5 if args.dry_run else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,  # Keep samples distinct for completion-only loss masking
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=1 if args.dry_run else train_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=2 if args.dry_run else train_cfg.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=5 if args.dry_run else train_cfg.get("save_steps", 200),
        save_total_limit=2,
        push_to_hub=push_to_hub and (not args.dry_run),
        hub_model_id=hub_model_id,
        hub_private_repo=train_cfg.get("hub_private_repo", True),
        report_to=train_cfg.get("report_to", "none"),
        gradient_checkpointing=True,
    )

    # 7. Initialize SFTTrainer with TrainingDynamicsCallback
    print("Initializing SFTTrainer...")
    dynamics_output_path = os.path.join(output_dir, "training_dynamics.jsonl")
    dynamics_callback = TrainingDynamicsCallback(output_path=dynamics_output_path)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[dynamics_callback],
    )

    # 8. Train Execution
    print("\nStarting SFT Training Run...")
    train_result = trainer.train()

    print("\nTraining complete! Evaluating on validation holdout...")
    eval_result = trainer.evaluate()
    print("Evaluation Results:", eval_result)

    # 9. Save and Push
    if not args.dry_run:
        print(f"\nSaving final model adapter to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        if push_to_hub:
            print(f"Pushing LoRA adapter weights to Hugging Face Hub: {hub_model_id}...")
            trainer.push_to_hub(commit_message="Add trained YarnBall Qwen2.5-7B LoRA adapter")
            tokenizer.push_to_hub(hub_model_id)

    print("\n" + "=" * 70)
    print("YARNBALL SFT RUN COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
