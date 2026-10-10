"""
Tests for Training Report & Plotting Generator in QwenSFT_YarnBall.
"""

import json
from pathlib import Path
import pytest

from report import (
    TrainingReportGenerator,
    compute_cartography_coordinates,
    generate_training_report,
    PureSVGPlotter,
)


def test_compute_cartography_coordinates():
    """Verify coordinate math (confidence, variability, correctness, regions)."""
    records = [
        {"epoch": 1, "guid": "sample_01", "seq_prob": 0.90, "is_correct": True},
        {"epoch": 2, "guid": "sample_01", "seq_prob": 0.95, "is_correct": True},
        {"epoch": 1, "guid": "sample_02", "seq_prob": 0.10, "is_correct": False},
        {"epoch": 2, "guid": "sample_02", "seq_prob": 0.85, "is_correct": True},
        {"epoch": 1, "guid": "sample_03", "seq_prob": 0.20, "is_correct": False},
        {"epoch": 2, "guid": "sample_03", "seq_prob": 0.25, "is_correct": False},
    ]

    coords = compute_cartography_coordinates(records)
    assert len(coords) == 3

    # sample_01: high confidence, low variability -> Easy-to-Learn
    s1 = coords["sample_01"]
    assert s1["confidence"] > 0.90
    assert s1["variability"] < 0.10
    assert s1["correctness"] == 1.0
    assert s1["region"] == "Easy-to-Learn"

    # sample_02: huge swing (0.10 to 0.85) -> high variability -> Ambiguous
    s2 = coords["sample_02"]
    assert s2["variability"] > 0.30
    assert s2["region"] == "Ambiguous"

    # sample_03: consistently low prob -> Hard-to-Learn
    s3 = coords["sample_03"]
    assert s3["confidence"] < 0.30
    assert s3["variability"] < 0.10
    assert s3["correctness"] == 0.0
    assert s3["region"] == "Hard-to-Learn"


def test_pure_svg_plotter(tmp_path: Path):
    """Verify that pure SVG plotter produces well-formed SVG files."""
    loss_svg = tmp_path / "loss.svg"
    PureSVGPlotter.plot_loss_curves(
        train_steps=[10, 20, 30],
        train_losses=[2.5, 1.8, 1.2],
        eval_steps=[15, 30],
        eval_losses=[2.0, 1.4],
        output_path=loss_svg,
    )
    assert loss_svg.is_file()
    svg_content = loss_svg.read_text(encoding="utf-8")
    assert "<svg" in svg_content
    assert "</svg>" in svg_content
    assert "Training Loss" in svg_content

    lr_svg = tmp_path / "lr.svg"
    PureSVGPlotter.plot_lr_schedule(
        steps=[10, 20, 30],
        lrs=[1e-5, 2e-4, 5e-5],
        output_path=lr_svg,
    )
    assert lr_svg.is_file()
    assert "<svg" in lr_svg.read_text(encoding="utf-8")

    ppl_svg = tmp_path / "ppl.svg"
    PureSVGPlotter.plot_perplexity_curves(
        eval_steps=[15, 30],
        eval_perplexities=[7.389, 4.055],
        output_path=ppl_svg,
    )
    assert ppl_svg.is_file()
    ppl_text = ppl_svg.read_text(encoding="utf-8")
    assert "<svg" in ppl_text
    assert "Validation Perplexity" in ppl_text


def test_training_report_generator_end_to_end(tmp_path: Path):
    """Verify end-to-end report generation with mock trainer_state.json and dynamics."""
    run_dir = tmp_path / "mock_run"
    run_dir.mkdir()

    # Create synthetic trainer_state.json
    state = {
        "log_history": [
            {"step": 10, "loss": 2.50, "learning_rate": 5e-5, "grad_norm": 1.5, "epoch": 0.5},
            {"step": 20, "loss": 1.80, "learning_rate": 2e-4, "grad_norm": 1.2, "epoch": 1.0},
            {"step": 20, "eval_loss": 1.75, "epoch": 1.0},
            {"step": 30, "loss": 1.20, "learning_rate": 1e-4, "grad_norm": 0.9, "epoch": 1.5},
            {"step": 40, "loss": 0.95, "learning_rate": 2e-5, "grad_norm": 0.8, "epoch": 2.0},
            {"step": 40, "eval_loss": 1.10, "epoch": 2.0},
            {"train_runtime": 120.5, "train_samples_per_second": 15.2, "step": 40},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

    # Create synthetic training_dynamics.jsonl
    dynamics = [
        {"epoch": 1, "guid": "sample_a", "seq_prob": 0.80, "is_correct": True},
        {"epoch": 2, "guid": "sample_a", "seq_prob": 0.92, "is_correct": True},
        {"epoch": 1, "guid": "sample_b", "seq_prob": 0.30, "is_correct": False},
        {"epoch": 2, "guid": "sample_b", "seq_prob": 0.75, "is_correct": True},
    ]
    dynamics_lines = "\n".join(json.dumps(d) for d in dynamics)
    (run_dir / "training_dynamics.jsonl").write_text(dynamics_lines, encoding="utf-8")

    # Generate report
    report_output_dir = run_dir / "report"
    result = generate_training_report(run_dir=run_dir, output_dir=report_output_dir)

    md_file = Path(result["markdown_report"])
    html_file = Path(result["html_report"])

    assert md_file.is_file()
    assert html_file.is_file()

    md_text = md_file.read_text(encoding="utf-8")
    assert "YarnBall SFT Training Report" in md_text
    assert "Loss Reduction" in md_text
    assert "62.00%" in md_text  # (2.50 - 0.95) / 2.50 = 62%
    assert "Validation Perplexity" in md_text

    html_text = html_file.read_text(encoding="utf-8")
    assert "YarnBall SFT Training Report" in html_text
    assert "Training Loss" in html_text
    assert "Validation Perplexity" in html_text

    # Verify plots directory
    plots_dir = report_output_dir / "plots"
    assert plots_dir.is_dir()
    assert (plots_dir / "loss_curves.svg").is_file()
    assert (plots_dir / "perplexity_curves.svg").is_file()
    assert (plots_dir / "learning_rate_schedule.svg").is_file()
    assert (plots_dir / "cartography_scatter.svg").is_file()
