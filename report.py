"""
Training Report & Progress Visualization Engine for QwenSFT_YarnBall.

Parses Hugging Face Trainer state (`trainer_state.json` / `log_history`) and
Training Dynamics logs (`training_dynamics.jsonl`), calculates convergence
statistics and validation perplexities (exp(loss)), creates publication-quality
plots (Matplotlib with zero-dependency SVG fallback), and compiles comprehensive
Markdown and HTML reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# 1. Training Dynamics & Cartography Coordinate Math
# ==============================================================================

def compute_cartography_coordinates(
    dynamics_records: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Computes per-sample confidence, variability, and correctness from sequence probabilities.
    
    Coordinates:
    - Confidence: mean(seq_prob) across epochs
    - Variability: std(seq_prob) across epochs
    - Correctness: fraction of epochs where is_correct is True
    """
    by_guid: Dict[str, List[Dict[str, Any]]] = {}
    for r in dynamics_records:
        guid = str(r.get("guid", ""))
        if guid:
            by_guid.setdefault(guid, []).append(r)

    coordinates: Dict[str, Dict[str, Any]] = {}
    for guid, samples in by_guid.items():
        probs = [float(s.get("seq_prob", 0.0)) for s in samples]
        corrects = [1.0 if s.get("is_correct") else 0.0 for s in samples]
        n = len(probs)
        if n == 0:
            continue

        mean_prob = sum(probs) / n
        variance = sum((p - mean_prob) ** 2 for p in probs) / n if n > 1 else 0.0
        std_prob = math.sqrt(variance)
        correctness = sum(corrects) / n

        # Swayamdipta region categorization
        if std_prob >= 0.15:
            region = "Ambiguous"
        elif mean_prob >= 0.70:
            region = "Easy-to-Learn"
        else:
            region = "Hard-to-Learn"

        coordinates[guid] = {
            "guid": guid,
            "confidence": round(mean_prob, 4),
            "variability": round(std_prob, 4),
            "correctness": round(correctness, 4),
            "region": region,
            "epochs_tracked": n,
        }

    return coordinates


# ==============================================================================
# 2. Pure-Python SVG Plotting Engine (Zero Extra Dependencies)
# ==============================================================================

class PureSVGPlotter:
    """Fallback vector plotter creating self-contained SVG graphics."""

    @staticmethod
    def _svg_wrapper(width: int, height: int, content: str, title: str = "") -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#131722; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <style>
    .grid {{ stroke: #2a2e39; stroke-width: 1; stroke-dasharray: 4,4; }}
    .axis {{ stroke: #787b86; stroke-width: 1.5; }}
    .axis-label {{ fill: #b2b5be; font-size: 11px; }}
    .chart-title {{ fill: #f0f3fa; font-size: 15px; font-weight: 600; }}
    .legend-text {{ fill: #d1d4dc; font-size: 11px; }}
  </style>
  <rect width="{width}" height="{height}" fill="#131722" rx="8"/>
  {f'<text x="{width/2}" y="28" text-anchor="middle" class="chart-title">{title}</text>' if title else ''}
  {content}
</svg>"""

    @classmethod
    def plot_loss_curves(
        cls,
        train_steps: List[int],
        train_losses: List[float],
        eval_steps: List[int],
        eval_losses: List[float],
        output_path: Path,
        title: str = "Training and Validation Loss Trajectory",
    ) -> None:
        W, H = 800, 420
        pad_l, pad_r, pad_t, pad_b = 70, 40, 50, 60
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b

        all_steps = (train_steps or []) + (eval_steps or [])
        all_losses = (train_losses or []) + (eval_losses or [])

        if not all_steps or not all_losses:
            output_path.write_text(cls._svg_wrapper(W, H, "<text x='400' y='210' fill='#888' text-anchor='middle'>No loss data available</text>", title), encoding="utf-8")
            return

        min_x, max_x = min(all_steps), max(all_steps) or 1
        if min_x == max_x:
            max_x += 1
        min_y = 0.0
        max_y = max(all_losses) * 1.15 or 1.0

        def to_screen(x: float, y: float) -> Tuple[float, float]:
            sx = pad_l + ((x - min_x) / (max_x - min_x)) * plot_w
            sy = pad_t + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h
            return sx, sy

        content = []
        # Grid lines (5 horizontal)
        for i in range(5):
            y_val = min_y + (i / 4.0) * (max_y - min_y)
            _, sy = to_screen(min_x, y_val)
            content.append(f'<line x1="{pad_l}" y1="{sy}" x2="{W - pad_r}" y2="{sy}" class="grid"/>')
            content.append(f'<text x="{pad_l - 10}" y="{sy + 4}" text-anchor="end" class="axis-label">{y_val:.2f}</text>')

        # Grid lines (5 vertical)
        for i in range(5):
            x_val = min_x + (i / 4.0) * (max_x - min_x)
            sx, _ = to_screen(x_val, min_y)
            content.append(f'<line x1="{sx}" y1="{pad_t}" x2="{sx}" y2="{H - pad_b}" class="grid"/>')
            content.append(f'<text x="{sx}" y="{H - pad_b + 20}" text-anchor="middle" class="axis-label">{int(round(x_val))}</text>')

        # Axes
        content.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{H - pad_b}" class="axis"/>')
        content.append(f'<line x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" y2="{H - pad_b}" class="axis"/>')

        # Axis labels
        content.append(f'<text x="{W/2}" y="{H - 15}" text-anchor="middle" class="axis-label">Optimization Step</text>')
        content.append(f'<text x="20" y="{H/2}" transform="rotate(-90 20 {H/2})" text-anchor="middle" class="axis-label">Cross-Entropy Loss</text>')

        # Train loss path
        if len(train_steps) > 1 and len(train_losses) == len(train_steps):
            pts = [f"{to_screen(x, y)[0]:.1f},{to_screen(x, y)[1]:.1f}" for x, y in zip(train_steps, train_losses)]
            content.append(f'<polyline fill="none" stroke="#2962ff" stroke-width="2.5" points="{" ".join(pts)}"/>')

        # Eval loss markers
        if eval_steps and len(eval_steps) == len(eval_losses):
            for x, y in zip(eval_steps, eval_losses):
                sx, sy = to_screen(x, y)
                content.append(f'<circle cx="{sx}" cy="{sy}" r="5" fill="#f7525f" stroke="#ffffff" stroke-width="1.5"/>')
            if len(eval_steps) > 1:
                e_pts = [f"{to_screen(x, y)[0]:.1f},{to_screen(x, y)[1]:.1f}" for x, y in zip(eval_steps, eval_losses)]
                content.append(f'<polyline fill="none" stroke="#f7525f" stroke-width="2" stroke-dasharray="4,4" points="{" ".join(e_pts)}"/>')

        # Legend
        content.append(f"""
        <g transform="translate({W - 220}, {pad_t + 10})">
          <line x1="0" y1="0" x2="20" y2="0" stroke="#2962ff" stroke-width="2.5"/>
          <text x="28" y="4" class="legend-text">Training Loss</text>
          <circle cx="10" cy="18" r="4.5" fill="#f7525f"/>
          <text x="28" y="22" class="legend-text">Validation Loss</text>
        </g>
        """)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cls._svg_wrapper(W, H, "".join(content), title), encoding="utf-8")

    @classmethod
    def plot_perplexity_curves(
        cls,
        eval_steps: List[int],
        eval_perplexities: List[float],
        output_path: Path,
        title: str = "Validation Perplexity Progression (exp(loss))",
    ) -> None:
        """Plots step-by-step validation perplexity trajectory with optimal checkpoint marker."""
        W, H = 800, 420
        pad_l, pad_r, pad_t, pad_b = 70, 40, 50, 60
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b

        if not eval_steps or not eval_perplexities:
            output_path.write_text(cls._svg_wrapper(W, H, "<text x='400' y='210' fill='#888' text-anchor='middle'>No perplexity data available</text>", title), encoding="utf-8")
            return

        min_x, max_x = min(eval_steps), max(eval_steps) or 1
        if min_x == max_x:
            max_x += 1
        min_y = max(1.0, min(eval_perplexities) * 0.85)
        max_y = max(eval_perplexities) * 1.15 or 2.0

        def to_screen(x: float, y: float) -> Tuple[float, float]:
            sx = pad_l + ((x - min_x) / (max_x - min_x)) * plot_w
            sy = pad_t + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h
            return sx, sy

        content = []
        for i in range(5):
            y_val = min_y + (i / 4.0) * (max_y - min_y)
            _, sy = to_screen(min_x, y_val)
            content.append(f'<line x1="{pad_l}" y1="{sy}" x2="{W - pad_r}" y2="{sy}" class="grid"/>')
            content.append(f'<text x="{pad_l - 10}" y="{sy + 4}" text-anchor="end" class="axis-label">{y_val:.2f}</text>')

        for i in range(5):
            x_val = min_x + (i / 4.0) * (max_x - min_x)
            sx, _ = to_screen(x_val, min_y)
            content.append(f'<line x1="{sx}" y1="{pad_t}" x2="{sx}" y2="{H - pad_b}" class="grid"/>')
            content.append(f'<text x="{sx}" y="{H - pad_b + 20}" text-anchor="middle" class="axis-label">{int(round(x_val))}</text>')

        content.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{H - pad_b}" class="axis"/>')
        content.append(f'<line x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" y2="{H - pad_b}" class="axis"/>')
        content.append(f'<text x="{W/2}" y="{H - 15}" text-anchor="middle" class="axis-label">Optimization Step</text>')
        content.append(f'<text x="20" y="{H/2}" transform="rotate(-90 20 {H/2})" text-anchor="middle" class="axis-label">Perplexity (PPL)</text>')

        if len(eval_steps) > 1:
            pts = [f"{to_screen(x, y)[0]:.1f},{to_screen(x, y)[1]:.1f}" for x, y in zip(eval_steps, eval_perplexities)]
            content.append(f'<polyline fill="none" stroke="#e040fb" stroke-width="2.5" stroke-dasharray="3,3" points="{" ".join(pts)}"/>')

        best_ppl = min(eval_perplexities)
        for x, y in zip(eval_steps, eval_perplexities):
            sx, sy = to_screen(x, y)
            is_best = (y == best_ppl)
            r = 7 if is_best else 5
            fill = "#00e676" if is_best else "#e040fb"
            content.append(f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>')
            content.append(f'<text x="{sx}" y="{sy - 10}" text-anchor="middle" font-size="10" fill="#d1d4dc">{y:.2f}</text>')

        content.append(f"""
        <g transform="translate({W - 250}, {pad_t + 10})">
          <line x1="0" y1="0" x2="20" y2="0" stroke="#e040fb" stroke-width="2.5" stroke-dasharray="3,3"/>
          <circle cx="10" cy="0" r="4.5" fill="#e040fb"/>
          <text x="28" y="4" class="legend-text">Validation Perplexity</text>
          <circle cx="10" cy="18" r="5.5" fill="#00e676"/>
          <text x="28" y="22" class="legend-text">Optimal Checkpoint</text>
        </g>
        """)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cls._svg_wrapper(W, H, "".join(content), title), encoding="utf-8")

    @classmethod
    def plot_lr_schedule(
        cls,
        steps: List[int],
        lrs: List[float],
        output_path: Path,
        title: str = "Learning Rate Schedule",
    ) -> None:
        W, H = 800, 360
        pad_l, pad_r, pad_t, pad_b = 80, 40, 50, 60
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b

        if not steps or not lrs:
            output_path.write_text(cls._svg_wrapper(W, H, "<text x='400' y='180' fill='#888' text-anchor='middle'>No LR data available</text>", title), encoding="utf-8")
            return

        min_x, max_x = min(steps), max(steps) or 1
        min_y, max_y = 0.0, (max(lrs) * 1.15) if max(lrs) > 0 else 1e-4

        def to_screen(x: float, y: float) -> Tuple[float, float]:
            sx = pad_l + ((x - min_x) / (max_x - min_x)) * plot_w
            sy = pad_t + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h
            return sx, sy

        content = []
        for i in range(4):
            y_val = min_y + (i / 3.0) * (max_y - min_y)
            _, sy = to_screen(min_x, y_val)
            content.append(f'<line x1="{pad_l}" y1="{sy}" x2="{W - pad_r}" y2="{sy}" class="grid"/>')
            content.append(f'<text x="{pad_l - 10}" y="{sy + 4}" text-anchor="end" class="axis-label">{y_val:.2e}</text>')

        for i in range(5):
            x_val = min_x + (i / 4.0) * (max_x - min_x)
            sx, _ = to_screen(x_val, min_y)
            content.append(f'<line x1="{sx}" y1="{pad_t}" x2="{sx}" y2="{H - pad_b}" class="grid"/>')
            content.append(f'<text x="{sx}" y="{H - pad_b + 20}" text-anchor="middle" class="axis-label">{int(round(x_val))}</text>')

        content.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{H - pad_b}" class="axis"/>')
        content.append(f'<line x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" y2="{H - pad_b}" class="axis"/>')

        if len(steps) > 1:
            pts = [f"{to_screen(x, y)[0]:.1f},{to_screen(x, y)[1]:.1f}" for x, y in zip(steps, lrs)]
            content.append(f'<polyline fill="none" stroke="#ff9800" stroke-width="2.5" points="{" ".join(pts)}"/>')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cls._svg_wrapper(W, H, "".join(content), title), encoding="utf-8")

    @classmethod
    def plot_cartography_scatter(
        cls,
        coordinates: Dict[str, Dict[str, Any]],
        output_path: Path,
        title: str = "Dataset Cartography Coordinate Map",
    ) -> None:
        W, H = 800, 500
        pad_l, pad_r, pad_t, pad_b = 60, 40, 50, 60
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b

        color_map = {
            "Easy-to-Learn": "#00e676",
            "Ambiguous": "#ffea00",
            "Hard-to-Learn": "#ff1744",
        }

        content = []
        for i in range(5):
            v = i / 4.0
            sy = pad_t + plot_h - (v * plot_h)
            content.append(f'<line x1="{pad_l}" y1="{sy}" x2="{W - pad_r}" y2="{sy}" class="grid"/>')
            content.append(f'<text x="{pad_l - 10}" y="{sy + 4}" text-anchor="end" class="axis-label">{v:.2f}</text>')

            sx = pad_l + (v * plot_w)
            content.append(f'<line x1="{sx}" y1="{pad_t}" x2="{sx}" y2="{H - pad_b}" class="grid"/>')
            content.append(f'<text x="{sx}" y="{H - pad_b + 20}" text-anchor="middle" class="axis-label">{(v * 0.5):.2f}</text>')

        content.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{H - pad_b}" class="axis"/>')
        content.append(f'<line x1="{pad_l}" y1="{H - pad_b}" x2="{W - pad_r}" y2="{H - pad_b}" class="axis"/>')
        content.append(f'<text x="{W/2}" y="{H - 15}" text-anchor="middle" class="axis-label">Variability (std)</text>')
        content.append(f'<text x="20" y="{H/2}" transform="rotate(-90 20 {H/2})" text-anchor="middle" class="axis-label">Confidence (mean seq_prob)</text>')

        for coord in coordinates.values():
            var = min(0.5, max(0.0, coord.get("variability", 0.0)))
            conf = min(1.0, max(0.0, coord.get("confidence", 0.0)))
            reg = coord.get("region", "Ambiguous")
            cx = pad_l + (var / 0.5) * plot_w
            cy = pad_t + plot_h - (conf * plot_h)
            col = color_map.get(reg, "#ffea00")
            content.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{col}" opacity="0.75"/>')

        content.append(f"""
        <g transform="translate({W - 200}, {pad_t + 10})">
          <circle cx="10" cy="0" r="5" fill="#00e676"/>
          <text x="24" y="4" class="legend-text">Easy-to-Learn</text>
          <circle cx="10" cy="18" r="5" fill="#ffea00"/>
          <text x="24" y="22" class="legend-text">Ambiguous</text>
          <circle cx="10" cy="36" r="5" fill="#ff1744"/>
          <text x="24" y="40" class="legend-text">Hard-to-Learn</text>
        </g>
        """)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cls._svg_wrapper(W, H, "".join(content), title), encoding="utf-8")


# ==============================================================================
# 3. Matplotlib High-Resolution Plotting Engine
# ==============================================================================

class MatplotlibPlotter:
    """Publication-grade Matplotlib plotting with dark finance theme and dual Perplexity axis."""

    @staticmethod
    def _apply_theme():
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": "#131722",
            "axes.facecolor": "#1e222d",
            "axes.edgecolor": "#363c4e",
            "axes.labelcolor": "#b2b5be",
            "xtick.color": "#787b86",
            "ytick.color": "#787b86",
            "grid.color": "#2a2e39",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "legend.facecolor": "#1e222d",
            "legend.edgecolor": "#363c4e",
            "font.family": "sans-serif",
        })

    @classmethod
    def plot_dashboard(
        cls,
        train_steps: List[int],
        train_losses: List[float],
        eval_steps: List[int],
        eval_losses: List[float],
        eval_perplexities: List[float],
        lr_steps: List[int],
        learning_rates: List[float],
        dynamics_by_epoch: Dict[int, Dict[str, float]],
        coordinates: Dict[str, Dict[str, Any]],
        output_path: Path,
    ) -> None:
        if not HAS_MATPLOTLIB:
            return

        cls._apply_theme()
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
        fig.suptitle("YarnBall Qwen2.5-7B SFT Training Dynamics & Convergence", fontsize=16, fontweight="bold", y=0.98, color="#f0f3fa")

        # -------------------------------------------------------------
        # Subplot 1: Train & Validation Loss + Dual Validation Perplexity
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        lines1, labels1 = [], []

        if train_steps and train_losses:
            l1 = ax1.plot(train_steps, train_losses, color="#2962ff", alpha=0.35, label="Raw Train Loss", linewidth=1)
            lines1.extend(l1)
            labels1.append("Raw Train Loss")
            if len(train_losses) >= 5:
                w = 5
                smoothed = [sum(train_losses[max(0, i - w + 1):i + 1]) / len(train_losses[max(0, i - w + 1):i + 1]) for i in range(len(train_losses))]
                l2 = ax1.plot(train_steps, smoothed, color="#2962ff", linewidth=2.2, label="Smoothed Train Loss")
                lines1.extend(l2)
                labels1.append("Smoothed Train Loss")

        if eval_steps and eval_losses:
            l3 = ax1.plot(eval_steps, eval_losses, color="#f7525f", marker="o", markersize=6, linewidth=2, linestyle="--", label="Validation Loss")
            lines1.extend(l3)
            labels1.append("Validation Loss")

        ax1.set_title("Loss & Validation Perplexity Trajectory", fontsize=13, fontweight="bold", color="#d1d4dc")
        ax1.set_xlabel("Optimization Step")
        ax1.set_ylabel("Cross-Entropy Loss", color="#b2b5be")
        ax1.grid(True)

        # Twin Y-Axis for Validation Perplexity
        if eval_steps and eval_perplexities:
            ax1_ppl = ax1.twinx()
            l4 = ax1_ppl.plot(eval_steps, eval_perplexities, color="#e040fb", marker="s", markersize=5, linewidth=1.8, linestyle=":", label="Validation Perplexity (PPL)")
            ax1_ppl.set_ylabel("Perplexity (PPL)", color="#e040fb", fontsize=11)
            ax1_ppl.tick_params(axis="y", labelcolor="#e040fb")
            lines1.extend(l4)
            labels1.append("Validation Perplexity (PPL)")

        ax1.legend(lines1, labels1, loc="upper right")

        # -------------------------------------------------------------
        # Subplot 2: Learning Rate & Optimization Schedule
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        if lr_steps and learning_rates:
            ax2.plot(lr_steps, learning_rates, color="#ff9800", linewidth=2.2, label="Learning Rate")
            ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
        ax2.set_title("Learning Rate Schedule (Warmup + Cosine)", fontsize=13, fontweight="bold", color="#d1d4dc")
        ax2.set_xlabel("Optimization Step")
        ax2.set_ylabel("Learning Rate")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        # -------------------------------------------------------------
        # Subplot 3: Per-Epoch Sequence Probabilities
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        epochs = sorted(dynamics_by_epoch.keys())
        if epochs:
            mean_probs = [dynamics_by_epoch[e].get("mean_prob", 0.0) for e in epochs]
            pass_rates = [dynamics_by_epoch[e].get("pass_rate", 0.0) for e in epochs]
            x_indices = range(len(epochs))
            bar_width = 0.35

            ax3.bar([x - bar_width/2 for x in x_indices], mean_probs, width=bar_width, color="#00bcd4", label="Mean Target Seq Prob")
            ax3.bar([x + bar_width/2 for x in x_indices], pass_rates, width=bar_width, color="#4caf50", label="Correctness Pass Rate")
            ax3.set_xticks(list(x_indices))
            ax3.set_xticklabels([f"Epoch {e}" for e in epochs])
            ax3.set_ylim(0.0, 1.05)
        ax3.set_title("Training Dynamics across Epochs", fontsize=13, fontweight="bold", color="#d1d4dc")
        ax3.set_xlabel("Training Epoch")
        ax3.set_ylabel("Probability / Accuracy Metric")
        ax3.grid(True)
        ax3.legend(loc="lower right")

        # -------------------------------------------------------------
        # Subplot 4: Dataset Cartography (Confidence vs. Variability)
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        if coordinates:
            color_map = {
                "Easy-to-Learn": "#00e676",
                "Ambiguous": "#ffea00",
                "Hard-to-Learn": "#ff1744",
            }
            for region, color in color_map.items():
                xs = [c["variability"] for c in coordinates.values() if c["region"] == region]
                ys = [c["confidence"] for c in coordinates.values() if c["region"] == region]
                if xs:
                    ax4.scatter(xs, ys, c=color, label=f"{region} (n={len(xs)})", alpha=0.65, edgecolors="none", s=25)

        ax4.set_title("Dataset Cartography Map", fontsize=13, fontweight="bold", color="#d1d4dc")
        ax4.set_xlabel("Variability (std)")
        ax4.set_ylabel("Confidence (mean seq_prob)")
        ax4.set_ylim(-0.05, 1.05)
        ax4.grid(True)
        ax4.legend(loc="lower left")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)


# ==============================================================================
# 4. Report Generator & Compiler
# ==============================================================================

class TrainingReportGenerator:
    """
    Compiles training metrics, perplexities, plots, and markdown/HTML reports.
    """

    def __init__(
        self,
        run_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.output_dir = Path(output_dir or (self.run_dir / "report")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def load_trainer_state(self) -> Dict[str, Any]:
        """Locates and loads trainer_state.json from run directory."""
        candidates = [
            self.run_dir / "trainer_state.json",
            self.run_dir.parent / "trainer_state.json",
        ]
        for p in candidates:
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    def load_training_dynamics(self) -> List[Dict[str, Any]]:
        """Locates and loads training_dynamics.jsonl from run or artifacts directory."""
        candidates = [
            self.run_dir / "training_dynamics.jsonl",
            self.run_dir.parent / "artifacts" / "training_dynamics.jsonl",
            Path("artifacts/training_dynamics.jsonl").resolve(),
        ]
        for p in candidates:
            if p.is_file():
                records = []
                try:
                    for line in p.read_text(encoding="utf-8").strip().split("\n"):
                        if line.strip():
                            records.append(json.loads(line))
                    return records
                except Exception:
                    pass
        return []

    def extract_time_series(
        self, log_history: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """Parses Hugging Face Trainer log_history into typed series with perplexity."""
        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []
        lr_steps, learning_rates = [], []
        grad_norm_steps, grad_norms = [], []

        for entry in log_history:
            step = entry.get("step")
            if step is None:
                continue

            # Training loss
            if "loss" in entry:
                train_steps.append(int(step))
                train_losses.append(float(entry["loss"]))

            # Learning rate
            if "learning_rate" in entry:
                lr_steps.append(int(step))
                learning_rates.append(float(entry["learning_rate"]))

            # Gradient norm
            if "grad_norm" in entry:
                grad_norm_steps.append(int(step))
                grad_norms.append(float(entry["grad_norm"]))

            # Validation loss
            if "eval_loss" in entry:
                eval_steps.append(int(step))
                eval_losses.append(float(entry["eval_loss"]))

        eval_perplexities = [round(math.exp(min(50.0, l)), 4) for l in eval_losses]

        return {
            "train_steps": train_steps,
            "train_losses": train_losses,
            "eval_steps": eval_steps,
            "eval_losses": eval_losses,
            "eval_perplexities": eval_perplexities,
            "lr_steps": lr_steps,
            "learning_rates": learning_rates,
            "grad_norm_steps": grad_norm_steps,
            "grad_norms": grad_norms,
        }

    def compute_summary_metrics(
        self,
        time_series: Dict[str, List[Any]],
        log_history: List[Dict[str, Any]],
        coordinates: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculates executive headline metrics."""
        train_losses = time_series["train_losses"]
        eval_losses = time_series["eval_losses"]
        eval_steps = time_series["eval_steps"]
        eval_perplexities = time_series["eval_perplexities"]
        learning_rates = time_series["learning_rates"]

        init_train_loss = train_losses[0] if train_losses else None
        final_train_loss = train_losses[-1] if train_losses else None
        train_loss_reduction = (
            ((init_train_loss - final_train_loss) / init_train_loss * 100.0)
            if (init_train_loss and final_train_loss)
            else 0.0
        )

        init_eval_loss = eval_losses[0] if eval_losses else None
        final_eval_loss = eval_losses[-1] if eval_losses else None
        best_eval_loss = min(eval_losses) if eval_losses else None
        best_eval_step = (
            eval_steps[eval_losses.index(best_eval_loss)]
            if best_eval_loss is not None
            else None
        )

        init_perplexity = eval_perplexities[0] if eval_perplexities else None
        final_perplexity = eval_perplexities[-1] if eval_perplexities else None
        best_perplexity = min(eval_perplexities) if eval_perplexities else None

        runtime_sec = None
        samples_per_sec = None
        for entry in reversed(log_history):
            if "train_runtime" in entry:
                runtime_sec = float(entry["train_runtime"])
            if "train_samples_per_second" in entry:
                samples_per_sec = float(entry["train_samples_per_second"])
            if runtime_sec is not None:
                break

        region_counts = {"Easy-to-Learn": 0, "Ambiguous": 0, "Hard-to-Learn": 0}
        for c in coordinates.values():
            reg = c.get("region", "Ambiguous")
            region_counts[reg] = region_counts.get(reg, 0) + 1

        total_tracked = len(coordinates)

        return {
            "initial_train_loss": round(init_train_loss, 4) if init_train_loss else "N/A",
            "final_train_loss": round(final_train_loss, 4) if final_train_loss else "N/A",
            "train_loss_reduction_pct": f"{train_loss_reduction:.2f}%" if init_train_loss else "N/A",
            "initial_eval_loss": round(init_eval_loss, 4) if init_eval_loss else "N/A",
            "final_eval_loss": round(final_eval_loss, 4) if final_eval_loss else "N/A",
            "best_eval_loss": round(best_eval_loss, 4) if best_eval_loss else "N/A",
            "best_eval_step": best_eval_step or "N/A",
            "initial_perplexity": round(init_perplexity, 2) if init_perplexity else "N/A",
            "final_perplexity": round(final_perplexity, 2) if final_perplexity else "N/A",
            "best_perplexity": round(best_perplexity, 2) if best_perplexity else "N/A",
            "peak_lr": f"{max(learning_rates):.2e}" if learning_rates else "N/A",
            "final_lr": f"{learning_rates[-1]:.2e}" if learning_rates else "N/A",
            "runtime_seconds": f"{runtime_sec:.1f}s" if runtime_sec else "N/A",
            "samples_per_second": f"{samples_per_sec:.2f}" if samples_per_sec else "N/A",
            "total_samples_tracked": total_tracked,
            "easy_samples": region_counts["Easy-to-Learn"],
            "ambiguous_samples": region_counts["Ambiguous"],
            "hard_samples": region_counts["Hard-to-Learn"],
        }

    def generate(self) -> Dict[str, Any]:
        """Executes full report generation, plots, and document outputs."""
        state = self.load_trainer_state()
        log_history = state.get("log_history", [])
        dynamics_records = self.load_training_dynamics()

        time_series = self.extract_time_series(log_history)
        coordinates = compute_cartography_coordinates(dynamics_records)

        dynamics_by_epoch: Dict[int, Dict[str, float]] = {}
        for r in dynamics_records:
            ep = int(r.get("epoch", 1))
            dynamics_by_epoch.setdefault(ep, {"probs": [], "corrects": []})
            dynamics_by_epoch[ep]["probs"].append(float(r.get("seq_prob", 0.0)))
            dynamics_by_epoch[ep]["corrects"].append(1.0 if r.get("is_correct") else 0.0)

        epoch_stats: Dict[int, Dict[str, float]] = {}
        for ep, d in dynamics_by_epoch.items():
            probs = d["probs"]
            corrects = d["corrects"]
            epoch_stats[ep] = {
                "mean_prob": sum(probs) / len(probs) if probs else 0.0,
                "pass_rate": sum(corrects) / len(corrects) if corrects else 0.0,
            }

        metrics = self.compute_summary_metrics(time_series, log_history, coordinates)

        # -------------------------------------------------------------
        # Generate Visualizations (Matplotlib + Pure SVG)
        # -------------------------------------------------------------
        loss_svg = self.plots_dir / "loss_curves.svg"
        ppl_svg = self.plots_dir / "perplexity_curves.svg"
        lr_svg = self.plots_dir / "learning_rate_schedule.svg"
        cart_svg = self.plots_dir / "cartography_scatter.svg"

        PureSVGPlotter.plot_loss_curves(
            time_series["train_steps"],
            time_series["train_losses"],
            time_series["eval_steps"],
            time_series["eval_losses"],
            loss_svg,
        )

        PureSVGPlotter.plot_perplexity_curves(
            time_series["eval_steps"],
            time_series["eval_perplexities"],
            ppl_svg,
        )

        PureSVGPlotter.plot_lr_schedule(
            time_series["lr_steps"],
            time_series["learning_rates"],
            lr_svg,
        )

        PureSVGPlotter.plot_cartography_scatter(
            coordinates,
            cart_svg,
        )

        dashboard_png = self.plots_dir / "training_dashboard.png"
        if HAS_MATPLOTLIB:
            MatplotlibPlotter.plot_dashboard(
                time_series["train_steps"],
                time_series["train_losses"],
                time_series["eval_steps"],
                time_series["eval_losses"],
                time_series["eval_perplexities"],
                time_series["lr_steps"],
                time_series["learning_rates"],
                epoch_stats,
                coordinates,
                dashboard_png,
            )

        # -------------------------------------------------------------
        # Generate Markdown Report
        # -------------------------------------------------------------
        md_path = self.output_dir / "training_report.md"
        self._write_markdown_report(md_path, metrics, time_series, epoch_stats)

        # -------------------------------------------------------------
        # Generate HTML Report
        # -------------------------------------------------------------
        html_path = self.output_dir / "training_report.html"
        self._write_html_report(html_path, metrics, time_series, epoch_stats)

        return {
            "output_dir": str(self.output_dir),
            "markdown_report": str(md_path),
            "html_report": str(html_path),
            "metrics": metrics,
            "plots": [
                str(loss_svg),
                str(ppl_svg),
                str(lr_svg),
                str(cart_svg),
                str(dashboard_png) if HAS_MATPLOTLIB else None,
            ],
        }

    def _write_markdown_report(
        self,
        path: Path,
        m: Dict[str, Any],
        time_series: Dict[str, List[Any]],
        epoch_stats: Dict[int, Dict[str, float]],
    ) -> None:
        model_name = self.config.get("model", {}).get("base_model_name", "Qwen/Qwen2.5-7B-Instruct")
        dataset_id = self.config.get("dataset", {}).get("repo_id", "CamiloEmiliano/yarnball-sft")
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        dashboard_rel = "plots/training_dashboard.png" if HAS_MATPLOTLIB else "plots/loss_curves.svg"

        # Perplexity evaluation rows
        eval_steps = time_series.get("eval_steps", [])
        eval_losses = time_series.get("eval_losses", [])
        eval_ppls = time_series.get("eval_perplexities", [])

        eval_rows = ""
        best_ppl = min(eval_ppls) if eval_ppls else 0.0
        for s, l, p in zip(eval_steps, eval_losses, eval_ppls):
            status = "Optimal Checkpoint" if p == best_ppl else "Converging"
            eval_rows += f"| Step {s} | `{l:.4f}` | `{p:.3f}` | {status} |\n"

        content = f"""# YarnBall SFT Training Report: {model_name}

**Dataset**: `{dataset_id}`  
**Generated**: `{now_utc}`  
**Run Directory**: `{self.run_dir}`  

---

## 1. Executive Headline Metrics

| Metric | Training Value | Validation Value | Target Specification |
| :--- | :--- | :--- | :--- |
| **Initial Loss** | `{m['initial_train_loss']}` | `{m['initial_eval_loss']}` | - |
| **Final Loss** | `{m['final_train_loss']}` | `{m['final_eval_loss']}` | Stable convergence |
| **Loss Reduction** | `{m['train_loss_reduction_pct']}` | - | > 40.0% |
| **Validation Perplexity** | - | `{m['final_perplexity']}` (Best: `{m['best_perplexity']}`) | Lower is better |
| **Optimal Checkpoint** | - | Step `{m['best_eval_step']}` | Lowest validation loss |
| **Peak Learning Rate** | `{m['peak_lr']}` | Final: `{m['final_lr']}` | Cosine decay |
| **Total Runtime / Speed** | `{m['runtime_seconds']}` | `{m['samples_per_second']} samples/sec` | Fast datacenter |

---

## 2. Visual Training Dashboard

![Training Dashboard]({dashboard_rel})

*Figure 1: Quad-panel training diagnostic showing (Top-Left) Cross-Entropy loss trajectory with dual-axis Validation Perplexity, (Top-Right) Learning rate warmup & cosine schedule, (Bottom-Left) Per-epoch target sequence probabilities, and (Bottom-Right) Dataset Cartography coordinates.*

---

## 3. Validation Perplexity Progression

Perplexity represents the model's effective branching factor over target completion tokens (PPL = exp(loss)). Lower perplexity indicates sharp certainty on Cypher grammar and `<think>` reasoning paths:

| Evaluation Step | Validation Loss | Validation Perplexity (PPL) | Status |
| :--- | :--- | :--- | :--- |
{eval_rows if eval_rows else "| Step 0 | N/A | N/A | None |\\n"}

![Validation Perplexity Trajectory](plots/perplexity_curves.svg)

---

## 4. Training Dynamics across Epochs

| Epoch | Mean Target Seq Probability | Correctness Pass Rate (>= 70%) | Status |
| :--- | :--- | :--- | :--- |
"""
        for ep in sorted(epoch_stats.keys()):
            mean_p = epoch_stats[ep]["mean_prob"]
            pass_r = epoch_stats[ep]["pass_rate"]
            status = "Nominal" if pass_r >= 0.75 else "Converging"
            content += f"| Epoch {ep} | `{mean_p:.4f}` | `{pass_r * 100:.2f}%` | {status} |\n"

        content += f"""
---

## 5. Dataset Cartography Breakdown

- **Total Probed Samples**: `{m['total_samples_tracked']}`
- **Easy-to-Learn** (High confidence, low variability): `{m['easy_samples']}`
- **Ambiguous** (High variability, optimal for active learning): `{m['ambiguous_samples']}`
- **Hard-to-Learn** (Consistently low confidence, candidate label noise): `{m['hard_samples']}`

### Individual Plots
- [Loss Curves SVG](plots/loss_curves.svg)
- [Validation Perplexity SVG](plots/perplexity_curves.svg)
- [Learning Rate Schedule SVG](plots/learning_rate_schedule.svg)
- [Cartography Coordinate Scatter SVG](plots/cartography_scatter.svg)
"""
        path.write_text(content, encoding="utf-8")

    def _write_html_report(
        self,
        path: Path,
        m: Dict[str, Any],
        time_series: Dict[str, List[Any]],
        epoch_stats: Dict[int, Dict[str, float]],
    ) -> None:
        model_name = self.config.get("model", {}).get("base_model_name", "Qwen/Qwen2.5-7B-Instruct")
        dataset_id = self.config.get("dataset", {}).get("repo_id", "CamiloEmiliano/yarnball-sft")
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        dashboard_img_src = "plots/training_dashboard.png" if HAS_MATPLOTLIB else "plots/loss_curves.svg"

        epoch_rows = "".join([
            f"<tr><td>Epoch {ep}</td><td><code>{epoch_stats[ep]['mean_prob']:.4f}</code></td><td><code>{epoch_stats[ep]['pass_rate'] * 100:.1f}%</code></td><td><span class='badge'>Converging</span></td></tr>"
            for ep in sorted(epoch_stats.keys())
        ])

        eval_steps = time_series.get("eval_steps", [])
        eval_losses = time_series.get("eval_losses", [])
        eval_ppls = time_series.get("eval_perplexities", [])
        best_ppl = min(eval_ppls) if eval_ppls else 0.0

        eval_table_rows = "".join([
            f"<tr><td>Step {s}</td><td><code>{l:.4f}</code></td><td><code style='color:#e040fb;'>{p:.3f}</code></td><td><span class='badge' style='background:{'rgba(63,185,80,0.2)' if p == best_ppl else 'rgba(110,118,129,0.2)'};color:{'#3fb950' if p == best_ppl else '#8b949e'}'>{'Optimal' if p == best_ppl else 'Converging'}</span></td></tr>"
            for s, l, p in zip(eval_steps, eval_losses, eval_ppls)
        ])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>YarnBall SFT Training Report - {model_name}</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --accent: #58a6ff;
      --purple: #e040fb;
      --success: #3fb950;
      --warning: #d29922;
      --danger: #f85149;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 30px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{ border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: var(--text-bright); margin: 0 0 10px 0; font-size: 24px; }}
    .meta {{ font-size: 13px; color: #8b949e; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }}
    .card-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; margin-bottom: 6px; }}
    .card-val {{ font-size: 22px; font-weight: bold; color: var(--text-bright); }}
    .card-sub {{ font-size: 12px; color: var(--accent); margin-top: 4px; }}
    .plot-box {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 25px; text-align: center; }}
    .plot-box img {{ max-width: 100%; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: #8b949e; text-transform: uppercase; font-size: 11px; }}
    code {{ background: rgba(110,118,129,0.2); padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
    .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>YarnBall SFT Training Report: {model_name}</h1>
      <div class="meta">
        Dataset: <strong>{dataset_id}</strong> &bull; Generated: {now_utc} &bull; Output: <code>{self.run_dir.name}</code>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Training Loss</div>
        <div class="card-val">{m['final_train_loss']}</div>
        <div class="card-sub">&darr; {m['train_loss_reduction_pct']} from {m['initial_train_loss']}</div>
      </div>
      <div class="card">
        <div class="card-title">Validation Perplexity</div>
        <div class="card-val" style="color:var(--purple);">{m['final_perplexity']}</div>
        <div class="card-sub">Best: {m['best_perplexity']} (Loss: {m['best_eval_loss']})</div>
      </div>
      <div class="card">
        <div class="card-title">Optimal Checkpoint</div>
        <div class="card-val">Step {m['best_eval_step']}</div>
        <div class="card-sub">Best Validation Step</div>
      </div>
      <div class="card">
        <div class="card-title">Throughput</div>
        <div class="card-val">{m['samples_per_second']}</div>
        <div class="card-sub">Total: {m['runtime_seconds']}</div>
      </div>
    </div>

    <div class="plot-box">
      <h3 style="margin-top:0; color:var(--text-bright);">Training &amp; Convergence Dynamics</h3>
      <img src="{dashboard_img_src}" alt="Training Dashboard">
    </div>

    <div class="card" style="margin-bottom:25px;">
      <h3 style="margin-top:0; color:var(--text-bright);">Validation Perplexity Trajectory</h3>
      <div style="text-align:center; margin-bottom:15px;">
        <img src="plots/perplexity_curves.svg" style="max-width:100%;" alt="Validation Perplexity Curve">
      </div>
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Validation Loss</th>
            <th>Validation Perplexity (PPL)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {eval_table_rows}
        </tbody>
      </table>
    </div>

    <div class="card" style="margin-bottom:25px;">
      <h3 style="margin-top:0; color:var(--text-bright);">Training Dynamics across Epochs</h3>
      <table>
        <thead>
          <tr>
            <th>Epoch</th>
            <th>Mean Target Seq Probability</th>
            <th>Correctness Pass Rate (>= 70%)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {epoch_rows}
        </tbody>
      </table>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Easy-to-Learn Samples</div>
        <div class="card-val" style="color:var(--success);">{m['easy_samples']}</div>
        <div class="card-sub">High Confidence / Low Variability</div>
      </div>
      <div class="card">
        <div class="card-title">Ambiguous Samples</div>
        <div class="card-val" style="color:var(--warning);">{m['ambiguous_samples']}</div>
        <div class="card-sub">Active Learning Target Partition</div>
      </div>
      <div class="card">
        <div class="card-title">Hard-to-Learn Samples</div>
        <div class="card-val" style="color:var(--danger);">{m['hard_samples']}</div>
        <div class="card-sub">Candidate Mislabeled Instances</div>
      </div>
    </div>
  </div>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")


# ==============================================================================
# 5. CLI & Quick Helper
# ==============================================================================

def generate_training_report(
    run_dir: str | Path = "./outputs/yarnball-qwen-7b-lora",
    output_dir: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Helper invoked directly by train.py or CLI."""
    generator = TrainingReportGenerator(run_dir=run_dir, output_dir=output_dir, config=config)
    return generator.generate()


def main():
    parser = argparse.ArgumentParser(description="Generate YarnBall SFT Training Report & Plots")
    parser.add_argument("--run-dir", default="./outputs/yarnball-qwen-7b-lora", help="Path to checkpoint/output directory containing trainer_state.json")
    parser.add_argument("--output-dir", default=None, help="Directory to save report artifacts and plots")
    parser.add_argument("--config", default="config.yaml", help="Path to training config.yaml")
    args = parser.parse_args()

    cfg = {}
    config_path = Path(args.config)
    if config_path.is_file():
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass

    print(f"Generating training report for: {args.run_dir}")
    result = generate_training_report(run_dir=args.run_dir, output_dir=args.output_dir, config=cfg)
    print(f"Report generated successfully!")
    print(f"  Markdown: {result['markdown_report']}")
    print(f"  HTML:     {result['html_report']}")


if __name__ == "__main__":
    main()
