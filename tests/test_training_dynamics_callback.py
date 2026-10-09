"""
Unit tests for TrainingDynamicsCallback in QwenSFT_YarnBall.
Verifies that per-sample sequence probabilities exp(-loss) and correctness are logged across epochs.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from train import TrainingDynamicsCallback


def test_callback_initialization(tmp_path: Path):
    """Verify callback initialization and previous log cleanup."""
    log_path = tmp_path / "training_dynamics.jsonl"
    log_path.write_text('{"epoch": 1, "guid": "old"}\n', encoding="utf-8")
    assert log_path.exists()

    callback = TrainingDynamicsCallback(output_path=str(log_path), correctness_threshold=0.75)
    # File should be cleared on init
    assert not log_path.exists()
    assert callback.correctness_threshold == 0.75


def test_callback_on_epoch_end_with_mocked_tensors(tmp_path: Path):
    """Verify loss masking, sequence probability exp(-loss), and JSONL emission."""
    try:
        import torch
    except ImportError:
        pytest.skip("Torch not installed in test environment")

    log_path = tmp_path / "dynamics.jsonl"
    callback = TrainingDynamicsCallback(output_path=str(log_path), correctness_threshold=0.70)

    # Mock Trainer state
    state = MagicMock()
    state.epoch = 1.0

    # Mock Model
    model = MagicMock()
    model.device = torch.device("cpu")

    # Construct synthetic batch: B=2, T=5, V=10
    # Sample 0 has low loss (high target logit), Sample 1 has higher loss
    vocab_size = 10
    seq_len = 5
    batch_size = 2

    # input_ids: [2, 5]
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)

    # labels: [2, 5] with prompt tokens masked as -100
    # Sample 0: first 2 tokens prompt (-100), last 3 target tokens (label 3)
    # Sample 1: first 2 tokens prompt (-100), last 3 target tokens (label 7)
    labels = torch.tensor([
        [-100, -100, 3, 3, 3],
        [-100, -100, 7, 7, 7],
    ], dtype=torch.long)

    # Logits: [2, 5, 10]
    logits = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32)
    # For sample 0, give label 3 very high logits (near zero loss -> seq_prob near 1.0)
    logits[0, :, 3] = 10.0
    # For sample 1, give label 0 high logits instead of 7 (high loss on target -> low seq_prob)
    logits[1, :, 0] = 10.0

    outputs = MagicMock()
    outputs.logits = logits
    model.return_value = outputs

    batch = {
        "guid": ["sample_aapl_01", "sample_msft_02"],
        "input_ids": input_ids,
        "labels": labels,
    }
    eval_dataloader = [batch]

    callback.on_epoch_end(args=MagicMock(), state=state, control=MagicMock(), model=model, eval_dataloader=eval_dataloader)

    assert log_path.is_file()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2

    rec0 = lines[0]
    assert rec0["epoch"] == 1
    assert rec0["guid"] == "sample_aapl_01"
    assert rec0["seq_prob"] > 0.90
    assert rec0["is_correct"] is True

    rec1 = lines[1]
    assert rec1["epoch"] == 1
    assert rec1["guid"] == "sample_msft_02"
    assert rec1["seq_prob"] < 0.10
    assert rec1["is_correct"] is False


def test_callback_handles_empty_or_none():
    """Verify callback does not raise errors when model or dataloader is None."""
    callback = TrainingDynamicsCallback()
    # Should safely return without exception
    callback.on_epoch_end(args=MagicMock(), state=MagicMock(), control=MagicMock(), model=None, eval_dataloader=None)
