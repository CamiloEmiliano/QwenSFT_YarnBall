"""Tests for Qwen ChatML dataset prompt-completion formatting."""

import pytest
from train import format_dataset_to_chatml


def test_format_chatml_with_target_completion():
    """Verify that dataset records with 'target_completion' key format cleanly into ChatML."""
    batch = {
        "prompt": [
            "<|extract_sec_graph|>\nDocument: SEC 10-K\nText: Apple depends on TSMC.\n\nTriples:"
        ],
        "target_completion": [
            "(:Company {name: 'Apple'})-[:SUPPLIES_TO]->(:Company {name: 'TSMC'})"
        ],
    }

    result = format_dataset_to_chatml(batch)
    assert "text" in result
    assert len(result["text"]) == 1

    formatted = result["text"][0]
    # Check exact ChatML structure
    assert formatted.startswith("<|im_start|>user\n")
    assert "<|im_end|>\n<|im_start|>assistant\n" in formatted
    assert formatted.endswith("<|im_end|>")
    assert "Apple depends on TSMC." in formatted
    assert "(:Company {name: 'Apple'})" in formatted


def test_format_chatml_with_completion_fallback():
    """Verify backward compatibility fallback when key is 'completion' instead of 'target_completion'."""
    batch = {
        "prompt": [
            "What is the capital requirement for a bank holding company?"
        ],
        "completion": [
            "Under Basel III guidelines, Tier 1 common equity must be at least 4.5%."
        ],
    }

    result = format_dataset_to_chatml(batch)
    assert "text" in result
    assert len(result["text"]) == 1
    assert "Basel III guidelines" in result["text"][0]


def test_format_chatml_strips_outer_whitespace():
    """Verify that extraneous leading/trailing whitespace in prompts/completions is trimmed."""
    batch = {
        "prompt": [
            "   \n  Prompt with whitespace   \n\n "
        ],
        "target_completion": [
            "  \n Response with whitespace  \n "
        ],
    }

    result = format_dataset_to_chatml(batch)
    formatted = result["text"][0]

    # Ensure no double leading newlines inside user tag
    assert "<|im_start|>user\nPrompt with whitespace<|im_end|>" in formatted
    assert "<|im_start|>assistant\nResponse with whitespace<|im_end|>" in formatted


def test_format_chatml_multi_batch():
    """Verify batched transformation across multiple records."""
    batch = {
        "prompt": ["Prompt 1", "Prompt 2", "Prompt 3"],
        "target_completion": ["Completion 1", "Completion 2", "Completion 3"],
    }

    result = format_dataset_to_chatml(batch)
    assert len(result["text"]) == 3
    for i in range(3):
        assert f"Prompt {i+1}" in result["text"][i]
        assert f"Completion {i+1}" in result["text"][i]
