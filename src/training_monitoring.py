"""Training history export and lightweight curve generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


CURVE_KEYS = {
    "loss": "train_loss",
    "learning_rate": "learning_rate",
    "grad_norm": "grad_norm",
    "eval_loss": "eval_loss",
}


def extract_curves(log_history: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, float]]]:
    curves: Dict[str, List[Dict[str, float]]] = {name: [] for name in CURVE_KEYS.values()}
    for item in log_history:
        step = item.get("step")
        if step is None:
            continue
        for raw_key, curve_name in CURVE_KEYS.items():
            value = item.get(raw_key)
            if value is None:
                continue
            try:
                curves[curve_name].append({"step": float(step), "value": float(value)})
            except (TypeError, ValueError):
                continue
    return {key: values for key, values in curves.items() if values}


def save_training_artifacts(log_history: Iterable[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = list(log_history)
    curves = extract_curves(history)

    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try_save_curve_pngs(curves, output_dir / "plots")


def try_save_curve_pngs(curves: Dict[str, List[Dict[str, float]]], plots_dir: Path) -> None:
    if not curves:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, points in curves.items():
        if not points:
            continue
        steps = [point["step"] for point in points]
        values = [point["value"] for point in points]
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.plot(steps, values, marker="o", linewidth=1.8)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{name}.png", dpi=160)
        plt.close(fig)
