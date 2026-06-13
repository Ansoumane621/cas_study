"""Checkpoint selection, pruning, and clean adapter export utilities."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
CHECKPOINT_RESULTS_FILE = "checkpoint_eval_results.json"
BEST_ADAPTER_DIR = "best_adapter"
LATEST_ADAPTER_DIR = "latest_adapter"
HUB_EXPORT_DIR = "hub_upload_adapter"

ADAPTER_ASSET_NAMES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
}


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return -1
    return int(match.group(1))


def list_checkpoints(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    checkpoints = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and CHECKPOINT_RE.match(path.name)
    ]
    return sorted(checkpoints, key=checkpoint_step)


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def has_adapter_files(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def is_adapter_asset(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in ADAPTER_ASSET_NAMES:
        return True
    return path.name.startswith("tokenizer.") or path.name.endswith(".model")


def export_clean_adapter(source_dir: Path, export_dir: Path) -> Path:
    """Copy only the LoRA adapter/tokenizer files needed for inference or Hub upload."""
    source_dir = source_dir.resolve()
    export_dir = export_dir.resolve()
    if source_dir == export_dir:
        raise ValueError("Clean export directory must be different from source directory.")
    if not has_adapter_files(source_dir):
        raise FileNotFoundError(f"No LoRA adapter files found in {source_dir}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in source_dir.iterdir():
        if not is_adapter_asset(item):
            continue
        shutil.copy2(item, export_dir / item.name)
        copied += 1

    if copied == 0:
        raise FileNotFoundError(f"No uploadable adapter assets found in {source_dir}")
    return export_dir


def composite_metric(metrics: Dict[str, float]) -> float:
    """Small teaching-friendly score to rank checkpoints on held-out examples."""
    return (
        0.45 * metrics.get("token_f1", 0.0)
        + 0.35 * metrics.get("rouge_l", 0.0)
        + 0.20 * metrics.get("bleu", 0.0)
    )


def average_scores(scored_rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(scored_rows)
    keys = ["exact_match", "token_f1", "bleu", "rouge_l"]
    averages: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        averages[key] = sum(values) / len(values) if values else 0.0
    averages["composite"] = composite_metric(averages)
    return averages


def evaluate_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    eval_data: str,
    max_examples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    system_prompt: str,
) -> Dict[str, Any]:
    """Generate on a few held-out examples and return checkpoint-level metrics."""
    from evaluation import generate_predictions, load_eval_examples, score_prediction

    examples = load_eval_examples(eval_data)[:max_examples]
    generated_rows = generate_predictions(
        examples=examples,
        model_name=model_name,
        adapter=str(checkpoint_dir),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        system_prompt=system_prompt,
    )

    scored_rows = []
    for row in generated_rows:
        metrics = score_prediction(row["prediction"], row["reference"])
        scored_rows.append({**row, **metrics})

    averages = average_scores(scored_rows)
    return {
        "name": checkpoint_dir.name,
        "path": str(checkpoint_dir),
        "step": checkpoint_step(checkpoint_dir),
        "num_eval_examples": len(scored_rows),
        "metrics": averages,
        "examples": scored_rows,
    }


def rank_checkpoint_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        results,
        key=lambda row: (
            float(row.get("metrics", {}).get("composite", 0.0)),
            int(row.get("step", -1)),
        ),
        reverse=True,
    )


def prune_checkpoints(output_dir: Path, keep_paths: Iterable[Path]) -> List[str]:
    keep = {path.resolve() for path in keep_paths}
    removed = []
    for checkpoint in list_checkpoints(output_dir):
        if checkpoint.resolve() in keep:
            continue
        shutil.rmtree(checkpoint)
        removed.append(checkpoint.name)
    return removed


def write_checkpoint_report(
    output_dir: Path,
    results: List[Dict[str, Any]],
    kept: List[str],
    removed: List[str],
    best_source: Optional[Path],
    latest_source: Optional[Path],
) -> Path:
    report = {
        "best_source": str(best_source) if best_source else None,
        "latest_source": str(latest_source) if latest_source else None,
        "kept_checkpoints": kept,
        "removed_checkpoints": removed,
        "results": results,
    }
    report_path = output_dir / CHECKPOINT_RESULTS_FILE
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def load_checkpoint_report(output_dir: Path) -> Dict[str, Any]:
    report_path = output_dir / CHECKPOINT_RESULTS_FILE
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def evaluate_prune_and_export(
    output_dir: Path,
    model_name: str,
    eval_data: Path,
    keep_count: int,
    max_examples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    system_prompt: str,
) -> Dict[str, Any]:
    checkpoints = list_checkpoints(output_dir)
    latest = latest_checkpoint(output_dir)
    if not checkpoints:
        if has_adapter_files(output_dir):
            export_clean_adapter(output_dir, output_dir / LATEST_ADAPTER_DIR)
        return {"kept_checkpoints": [], "removed_checkpoints": [], "results": []}

    results: List[Dict[str, Any]] = []
    if eval_data.exists() and max_examples > 0:
        for checkpoint in checkpoints:
            results.append(
                evaluate_checkpoint(
                    checkpoint_dir=checkpoint,
                    model_name=model_name,
                    eval_data=str(eval_data),
                    max_examples=max_examples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    system_prompt=system_prompt,
                )
            )
        ranked = rank_checkpoint_results(results)
        kept_sources = [Path(row["path"]) for row in ranked[:keep_count]]
        best_source = Path(ranked[0]["path"]) if ranked else latest
    else:
        kept_sources = checkpoints[-keep_count:]
        best_source = latest

    removed = prune_checkpoints(output_dir, kept_sources)
    kept = [path.name for path in kept_sources]

    if best_source and best_source.exists():
        export_clean_adapter(best_source, output_dir / BEST_ADAPTER_DIR)
    if latest and latest.exists():
        export_clean_adapter(latest, output_dir / LATEST_ADAPTER_DIR)

    report_path = write_checkpoint_report(
        output_dir=output_dir,
        results=results,
        kept=kept,
        removed=removed,
        best_source=best_source,
        latest_source=latest,
    )
    return {
        "report_path": str(report_path),
        "best_source": str(best_source) if best_source else None,
        "latest_source": str(latest) if latest else None,
        "kept_checkpoints": kept,
        "removed_checkpoints": removed,
        "results": results,
    }


def select_adapter_source(adapter_dir: Path, selector: str) -> Tuple[Path, str]:
    """Select which adapter to export/upload: best, latest, final, or explicit path."""
    selector = selector.strip()
    if selector == "best":
        best_export = adapter_dir / BEST_ADAPTER_DIR
        if has_adapter_files(best_export):
            return best_export, "best_adapter"
        report = load_checkpoint_report(adapter_dir)
        best_source = report.get("best_source")
        if best_source and has_adapter_files(Path(best_source)):
            return Path(best_source), "best_checkpoint"
        latest = latest_checkpoint(adapter_dir)
        if latest and has_adapter_files(latest):
            return latest, "latest_checkpoint_fallback"
        return adapter_dir, "final_fallback"

    if selector == "latest":
        latest_export = adapter_dir / LATEST_ADAPTER_DIR
        if has_adapter_files(latest_export):
            return latest_export, "latest_adapter"
        latest = latest_checkpoint(adapter_dir)
        if latest and has_adapter_files(latest):
            return latest, "latest_checkpoint"
        return adapter_dir, "final_fallback"

    if selector == "final":
        return adapter_dir, "final_adapter"

    custom = Path(selector)
    if not custom.is_absolute():
        custom = adapter_dir / custom
    return custom, "custom_adapter"
