"""Model presets for the Wolof LoRA demo."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "qwen": {
        "model": "Qwen/Qwen3-0.6B",
        "output_dir": "outputs/qwen3_0_6b_wolof_lora",
        "logging_dir": "outputs/tensorboard/qwen3_0_6b_wolof_lora",
    },
    "gemma": {
        "model": "google/gemma-4-E2B-it",
        "output_dir": "outputs/gemma_4_e2b_wolof_lora",
        "logging_dir": "outputs/tensorboard/gemma_4_e2b_wolof_lora",
    },
}


def normalize_model_choice(model_choice: str) -> str:
    choice = model_choice.strip().lower()
    aliases = {
        "qwen3": "qwen",
        "qwen3-0.6b": "qwen",
        "gemma4": "gemma",
        "gemma-4": "gemma",
        "gemma-4-e2b": "gemma",
        "google/gemma-4-e2b-it": "gemma",
    }
    choice = aliases.get(choice, choice)
    if choice not in MODEL_PRESETS:
        allowed = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model choice '{model_choice}'. Use one of: {allowed}.")
    return choice


def resolve_model_name(model_choice: str, model_override: Optional[str]) -> str:
    if model_override:
        return model_override
    return MODEL_PRESETS[normalize_model_choice(model_choice)]["model"]


def resolve_output_dir(model_choice: str, output_dir: str) -> str:
    if output_dir and output_dir != "auto":
        return output_dir
    return MODEL_PRESETS[normalize_model_choice(model_choice)]["output_dir"]


def resolve_logging_dir(model_choice: str, logging_dir: str) -> str:
    if logging_dir and logging_dir != "auto":
        return logging_dir
    return MODEL_PRESETS[normalize_model_choice(model_choice)]["logging_dir"]


def resolve_adapter_dir(model_choice: str, adapter: str, script_dir: Path) -> str:
    if adapter and adapter != "auto":
        return adapter

    output_dir = Path(resolve_output_dir(model_choice, "auto"))
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    best_adapter = output_dir / "best_adapter"
    if best_adapter.exists():
        return str(best_adapter)
    return str(output_dir)
