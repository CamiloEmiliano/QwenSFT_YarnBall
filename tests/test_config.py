"""Tests for training configuration and hyperparameter integrity."""

from pathlib import Path
import pytest
import yaml

from train import load_yaml_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def test_config_file_exists():
    """Verify that config.yaml exists at repository root."""
    assert CONFIG_PATH.is_file(), f"config.yaml not found at {CONFIG_PATH}"


def test_load_yaml_config_returns_dict():
    """Verify that load_yaml_config returns a valid dictionary."""
    cfg = load_yaml_config(str(CONFIG_PATH))
    assert isinstance(cfg, dict)
    assert len(cfg) > 0


def test_model_configuration():
    """Verify base model parameters match specification."""
    cfg = load_yaml_config(str(CONFIG_PATH))
    model = cfg.get("model", {})
    assert model.get("base_model_name") == "Qwen/Qwen2.5-7B-Instruct"
    assert model.get("torch_dtype") == "bfloat16"
    assert model.get("attn_implementation") in ("flash_attention_2", "sdpa")


def test_lora_hyperparameters():
    """Verify LoRA rank, alpha, dropout, and target projections."""
    cfg = load_yaml_config(str(CONFIG_PATH))
    lora = cfg.get("lora", {})

    assert lora.get("r") == 32, "LoRA rank must be 32"
    assert lora.get("lora_alpha") == 64, "LoRA alpha must be 64 (scaling factor 2.0)"
    assert lora.get("lora_dropout") == 0.05
    assert lora.get("task_type") == "CAUSAL_LM"

    target_modules = lora.get("target_modules", [])
    expected_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(target_modules) == expected_modules, "All 7 linear projection layers must be targeted"


def test_training_hyperparameters():
    """Verify batch sizes, sequence length, and precision settings."""
    cfg = load_yaml_config(str(CONFIG_PATH))
    training = cfg.get("training", {})

    assert training.get("num_train_epochs") == 3
    assert training.get("per_device_train_batch_size") == 4
    assert training.get("gradient_accumulation_steps") == 4
    effective_batch = training.get("per_device_train_batch_size") * training.get("gradient_accumulation_steps")
    assert effective_batch == 16, "Effective global batch size must be 16"

    assert training.get("max_seq_length") == 2048, "Sequence length must be clamped to 2,048"
    assert training.get("bf16") is True, "Precision must default to bfloat16"
    assert training.get("lr_scheduler_type") == "cosine"
    assert float(training.get("learning_rate")) == 2.0e-4


def test_dataset_and_hub_targets():
    """Verify private dataset and model hub identifiers."""
    cfg = load_yaml_config(str(CONFIG_PATH))
    ds = cfg.get("dataset", {})
    training = cfg.get("training", {})

    assert ds.get("repo_id") == "CamiloEmiliano/yarnball-sft"
    assert training.get("hub_model_id") == "CamiloEmiliano/yarnball-qwen-7b-lora"
    assert training.get("hub_private_repo") is True
