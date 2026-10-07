"""Tests for security hygiene, ignore rules, and argument safety."""

from pathlib import Path
import pytest

from train import parse_args

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_gitignore_ignores_sensitive_artifacts():
    """Verify that .gitignore properly ignores credentials and model weights."""
    gitignore_path = REPO_ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore file must exist"

    content = gitignore_path.read_text(encoding="utf-8")
    lines = {line.strip() for line in content.splitlines() if line.strip()}

    assert ".env" in lines, ".env must be ignored"
    assert "outputs/" in lines or "outputs" in lines, "outputs/ must be ignored"
    assert "wandb/" in lines or "wandb" in lines, "wandb/ must be ignored"


def test_config_contains_no_secrets():
    """Verify that config.yaml does not contain plaintext access tokens."""
    config_path = REPO_ROOT / "config.yaml"
    content = config_path.read_text(encoding="utf-8")

    forbidden_patterns = ["hf_", "vast_", "Bearer ", "token:", "secret", "password"]
    for pat in forbidden_patterns:
        assert pat.lower() not in content.lower(), f"Potential secret pattern '{pat}' detected in config.yaml"


def test_train_argparser_defaults():
    """Verify that default CLI arguments for train.py are safe."""
    import sys
    from unittest.mock import patch

    with patch("sys.argv", ["train.py"]):
        args = parse_args()
        assert args.config == "config.yaml"
        assert args.dry_run is False
        assert args.no_push is False
        assert args.base_model_name is None
        assert args.dataset_revision is None
