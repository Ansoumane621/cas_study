"""Download and prepare Wolof fine-tuning datasets by source family.

The pipeline keeps source families separate:
- aya: public multilingual instruction/translation data;
- soynade: Wolof orthography normalization data;
- synth: local synthetic classroom data.

It then creates chat-format files, deterministic train/validation/eval splits,
and a YAML config consumed by the assistant-only training script.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer


try:
    from data_utils import SYSTEM_PROMPT
except ImportError:
    from src.data_utils import SYSTEM_PROMPT


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
CONFIG_DIR = PROJECT_DIR / "configs"

SOURCE_FILES = {
    "aya": DATA_DIR / "wolof_aya.jsonl",
    "soynade": DATA_DIR / "wolof_soynade.jsonl",
    "synth": DATA_DIR / "wolof_synth.jsonl",
}
CHAT_FILES = {
    "aya": DATA_DIR / "chat_aya.json",
    "soynade": DATA_DIR / "chat_soynade.json",
    "synth": DATA_DIR / "chat_synth.json",
}

DEFAULT_COMBINED_FILE = DATA_DIR / "wolof_train_aya_soynade.jsonl"
DEFAULT_SYNTH_FILE = DATA_DIR / "syntetic_wolof_instruct_data.jsonl"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "wolof_training_config.yml"
DEFAULT_EVAL_JSONL = SPLITS_DIR / "eval_all.jsonl"

app = typer.Typer(
    help="Download, normalize, chat-template, split, and configure Wolof training data.",
    pretty_exceptions_show_locals=False,
)


@dataclass(frozen=True)
class SourceInfo:
    name: str
    raw_path: Path
    chat_path: Path
    default_category: str
    description: str


SOURCES = {
    "aya": SourceInfo(
        name="aya",
        raw_path=SOURCE_FILES["aya"],
        chat_path=CHAT_FILES["aya"],
        default_category="language",
        description="Aya Wolof instruction/translation examples",
    ),
    "soynade": SourceInfo(
        name="soynade",
        raw_path=SOURCE_FILES["soynade"],
        chat_path=CHAT_FILES["soynade"],
        default_category="orthography",
        description="Wolof non-standard orthography normalization examples",
    ),
    "synth": SourceInfo(
        name="synth",
        raw_path=SOURCE_FILES["synth"],
        chat_path=CHAT_FILES["synth"],
        default_category="synthetic",
        description="Local synthetic Senegal/Africa classroom instruction examples",
    ),
}


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = list(rows)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(data)


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalized_instruction_row(
    instruction: Any,
    output: Any,
    category: str,
    source: str,
    source_detail: str,
) -> Optional[Dict[str, str]]:
    instruction_text = clean_text(instruction)
    output_text = clean_text(output)
    if not instruction_text or not output_text:
        return None
    return {
        "instruction": instruction_text,
        "input": category,
        "output": output_text,
        "source": source,
        "source_detail": source_detail,
    }


def row_to_chat(row: Dict[str, str], source_name: str) -> Dict[str, Any]:
    category = clean_text(row.get("input")) or SOURCES[source_name].default_category
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean_text(row.get("instruction"))},
            {"role": "assistant", "content": clean_text(row.get("output"))},
        ],
        "metadata": {
            "source": source_name,
            "source_detail": clean_text(row.get("source_detail")) or source_name,
            "category": category,
            "variables": {
                "instruction": clean_text(row.get("instruction")),
                "input": category,
                "output": clean_text(row.get("output")),
                "source": source_name,
                "source_detail": clean_text(row.get("source_detail")) or source_name,
                "category": category,
            },
        },
    }


def load_chat(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return raw


def split_rows(
    rows: List[Dict[str, Any]],
    validation_ratio: float,
    eval_ratio: float,
    seed: int,
    min_validation: int,
    min_eval: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create deterministic train/validation/eval splits for one source."""
    if not rows:
        return {"train": [], "validation": [], "eval": []}

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    eval_count = ratio_count(n, eval_ratio, min_eval)
    validation_count = ratio_count(n, validation_ratio, min_validation)
    if eval_count + validation_count >= n:
        eval_count = max(1, min(eval_count, n // 5))
        validation_count = max(1, min(validation_count, n // 5))
    train_count = max(0, n - eval_count - validation_count)

    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "eval": shuffled[train_count + validation_count :],
    }


def ratio_count(total: int, ratio: float, minimum: int) -> int:
    if total <= 1 or ratio <= 0:
        return 0
    count = round(total * ratio)
    count = max(minimum, count)
    return min(count, max(1, total - 1))


def eval_jsonl_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    eval_rows = []
    for row in rows:
        messages = row.get("messages", [])
        if len(messages) < 3:
            continue
        metadata = row.get("metadata", {})
        eval_rows.append(
            {
                "instruction": clean_text(messages[-2].get("content")),
                "input": clean_text(metadata.get("category")),
                "reference": clean_text(messages[-1].get("content")),
                "prediction": "",
            }
        )
    return eval_rows


def write_config(
    config_path: Path,
    validation_ratio: float,
    eval_ratio: float,
    seed: int,
    split_counts: Dict[str, Dict[str, int]],
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Wolof LoRA training configuration.",
        "# Ratios are decimal fractions: 0.05 means 5%, not 0.05%.",
        f"seed: {seed}",
        "",
        "splits:",
        f"  validation_ratio: {validation_ratio}",
        f"  eval_ratio: {eval_ratio}",
        "  note: \"validation/eval are stratified per source family\"",
        "",
        "model:",
        "  model_choice: qwen",
        "  alternatives:",
        "    - qwen",
        "    - gemma",
        "",
        "training:",
        "  epochs: 3",
        "  batch_size: 1",
        "  grad_accum: 8",
        "  max_seq_length: 1024",
        "  learning_rate: 0.0001",
        "  warmup_steps: 50",
        "  weight_decay: 0.01",
        "  lora_r: 16",
        "  lora_alpha: 32",
        "  lora_dropout: 0.05",
        "  logging_steps: 10",
        "  save_steps: 100",
        "  save_total_limit: 5",
        "  eval_steps: 100",
        "",
        "data_sources:",
    ]
    for name, source in SOURCES.items():
        counts = split_counts.get(name, {})
        lines.extend(
            [
                f"  - name: {name}",
                f"    description: \"{source.description}\"",
                f"    raw_path: {relative_to_project(source.raw_path)}",
                f"    chat_path: {relative_to_project(source.chat_path)}",
                f"    default_category: {source.default_category}",
                f"    examples: {sum(counts.values())}",
                f"    train_examples: {counts.get('train', 0)}",
                f"    validation_examples: {counts.get('validation', 0)}",
                f"    eval_examples: {counts.get('eval', 0)}",
            ]
        )
    lines.extend(["", "training_data:"])
    for name in SOURCES:
        lines.extend(
            [
                f"  - name: {name}",
                f"    path: data/splits/chat_{name}_train.json",
                "    weight: 1.0",
            ]
        )
    lines.extend(["", "validation_data:"])
    for name in SOURCES:
        lines.extend(
            [
                f"  - name: {name}",
                f"    path: data/splits/chat_{name}_validation.json",
            ]
        )
    lines.extend(
        [
            "",
            "eval_data:",
            f"  combined_path: {relative_to_project(DEFAULT_EVAL_JSONL)}",
            "  sources:",
        ]
    )
    for name in SOURCES:
        lines.extend(
            [
                f"    - name: {name}",
                f"      path: data/splits/chat_{name}_eval.json",
            ]
        )
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_DIR))


def bootstrap_aya_soynade_from_combined(combined_path: Path) -> None:
    if not combined_path.exists():
        return
    if SOURCE_FILES["aya"].exists() and SOURCE_FILES["soynade"].exists():
        return

    typer.echo(f"Bootstrapping aya/soynade from existing file: {combined_path}")
    aya_rows = []
    soynade_rows = []
    for row in read_jsonl(combined_path):
        instruction = clean_text(row.get("instruction"))
        output = clean_text(row.get("output"))
        if not instruction or not output:
            continue
        if looks_like_soynade(instruction):
            source_name = "soynade"
            target = soynade_rows
            category = SOURCES["soynade"].default_category
        else:
            source_name = "aya"
            target = aya_rows
            category = SOURCES["aya"].default_category
        normalized = normalized_instruction_row(
            instruction=instruction,
            output=output,
            category=category,
            source=source_name,
            source_detail=f"bootstrap:{combined_path.name}",
        )
        if normalized:
            target.append(normalized)

    if aya_rows and not SOURCE_FILES["aya"].exists():
        write_jsonl(aya_rows, SOURCE_FILES["aya"])
        typer.echo(f"Saved bootstrapped aya rows: {len(aya_rows)}")
    if soynade_rows and not SOURCE_FILES["soynade"].exists():
        write_jsonl(soynade_rows, SOURCE_FILES["soynade"])
        typer.echo(f"Saved bootstrapped soynade rows: {len(soynade_rows)}")


def looks_like_soynade(instruction: str) -> bool:
    lowered = instruction.lower()
    markers = [
        "corrige l'orthographe",
        "non_standard",
        "<non_standard>",
        "beqil ma",
        "wolof standard",
        "standardisé",
    ]
    return any(marker in lowered for marker in markers)


def copy_synth_source(synth_input: Path) -> None:
    if SOURCE_FILES["synth"].exists():
        return
    if not synth_input.exists():
        typer.echo(f"WARNING: local synth file not found: {synth_input}")
        return

    rows = []
    for row in read_jsonl(synth_input):
        normalized = normalized_instruction_row(
            instruction=row.get("instruction"),
            output=row.get("output"),
            category=clean_text(row.get("input")) or SOURCES["synth"].default_category,
            source="synth",
            source_detail=synth_input.name,
        )
        if normalized:
            rows.append(normalized)
    count = write_jsonl(rows, SOURCE_FILES["synth"])
    typer.echo(f"Saved synth rows: {count}")


@app.command()
def download(
    max_aya_examples: Optional[int] = typer.Option(
        None,
        help="Optional cap for Aya examples; leave empty to stream all matching Wolof rows.",
    ),
    max_soynade_examples: Optional[int] = typer.Option(
        None,
        help="Optional cap for Soynade examples; leave empty to stream all rows.",
    ),
    synth_input: str = typer.Option(
        str(DEFAULT_SYNTH_FILE.relative_to(PROJECT_DIR)),
        help="Local synthetic JSONL file copied into the normalized synth source.",
    ),
) -> None:
    """Download public sources and normalize each source into its own JSONL file."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    typer.echo("Loading CohereLabs/aya_dataset Wolof rows...")
    aya_rows = []
    aya_ds = load_dataset("CohereLabs/aya_dataset", split="train", streaming=True)
    for row in aya_ds:
        if row.get("language") != "Wolof" and row.get("language_code") != "wol":
            continue
        normalized = normalized_instruction_row(
            instruction=row.get("inputs"),
            output=row.get("targets"),
            category=SOURCES["aya"].default_category,
            source="aya",
            source_detail="CohereLabs/aya_dataset",
        )
        if normalized:
            aya_rows.append(normalized)
        if max_aya_examples and len(aya_rows) >= max_aya_examples:
            break
    typer.echo(f"Saving aya source rows: {len(aya_rows)}")
    write_jsonl(aya_rows, SOURCE_FILES["aya"])

    typer.echo("Loading soynade-research/Wolof-Non-Standard-Orthography rows...")
    soynade_rows = []
    soynade_ds = load_dataset(
        "soynade-research/Wolof-Non-Standard-Orthography",
        split="train",
        streaming=True,
    )
    for row in soynade_ds:
        normalized = normalized_instruction_row(
            instruction=(
                "Corrige l'orthographe de cette phrase pour l'écrire en wolof standard :\n"
                f"{clean_text(row.get('non_standardized'))}"
            ),
            output=row.get("wo"),
            category=SOURCES["soynade"].default_category,
            source="soynade",
            source_detail="soynade-research/Wolof-Non-Standard-Orthography",
        )
        if normalized:
            soynade_rows.append(normalized)
        if max_soynade_examples and len(soynade_rows) >= max_soynade_examples:
            break
    typer.echo(f"Saving soynade source rows: {len(soynade_rows)}")
    write_jsonl(soynade_rows, SOURCE_FILES["soynade"])

    copy_synth_source(PROJECT_DIR / synth_input)


@app.command()
def prepare(
    validation_ratio: float = typer.Option(
        0.05,
        help="Validation split ratio per source. 0.05 means 5%.",
    ),
    eval_ratio: float = typer.Option(
        0.05,
        help="Evaluation split ratio per source. 0.05 means 5%.",
    ),
    seed: int = typer.Option(42, help="Deterministic split seed."),
    min_validation: int = typer.Option(1, help="Minimum validation examples per non-empty source."),
    min_eval: int = typer.Option(1, help="Minimum eval examples per non-empty source."),
    synth_input: str = typer.Option(
        str(DEFAULT_SYNTH_FILE.relative_to(PROJECT_DIR)),
        help="Local synthetic JSONL copied when data/wolof_synth.jsonl is missing.",
    ),
    config_path: str = typer.Option(
        str(DEFAULT_CONFIG_FILE.relative_to(PROJECT_DIR)),
        help="YAML config written for train_lora_assistant_only.py.",
    ),
) -> None:
    """Create chat files, train/validation/eval splits, and the YAML config."""
    if validation_ratio >= 1 or eval_ratio >= 1:
        raise ValueError("Ratios must be decimal fractions, for example 0.05 for 5%.")
    if validation_ratio + eval_ratio >= 0.5:
        raise ValueError("validation_ratio + eval_ratio is too large for small datasets.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap_aya_soynade_from_combined(DEFAULT_COMBINED_FILE)
    copy_synth_source(PROJECT_DIR / synth_input)

    split_counts: Dict[str, Dict[str, int]] = {}
    all_eval_rows: List[Dict[str, str]] = []

    for name, source in SOURCES.items():
        if not source.raw_path.exists():
            typer.echo(f"WARNING: missing source file for {name}: {source.raw_path}")
            split_counts[name] = {"train": 0, "validation": 0, "eval": 0}
            continue

        raw_rows = read_jsonl(source.raw_path)
        chat_rows = [row_to_chat(row, name) for row in raw_rows]
        write_json(chat_rows, source.chat_path)
        typer.echo(f"Saved chat file for {name}: {len(chat_rows)} rows -> {source.chat_path}")

        splits = split_rows(
            chat_rows,
            validation_ratio=validation_ratio,
            eval_ratio=eval_ratio,
            seed=seed,
            min_validation=min_validation,
            min_eval=min_eval,
        )
        split_counts[name] = {split_name: len(rows) for split_name, rows in splits.items()}
        for split_name, rows in splits.items():
            path = SPLITS_DIR / f"chat_{name}_{split_name}.json"
            write_json(rows, path)
            typer.echo(f"Saved {name} {split_name}: {len(rows)} rows -> {path}")
        all_eval_rows.extend(eval_jsonl_rows(splits["eval"]))

    write_jsonl(all_eval_rows, DEFAULT_EVAL_JSONL)
    typer.echo(f"Saved combined eval JSONL: {len(all_eval_rows)} rows -> {DEFAULT_EVAL_JSONL}")

    write_config(
        config_path=PROJECT_DIR / config_path,
        validation_ratio=validation_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        split_counts=split_counts,
    )
    typer.echo(f"Saved training config: {PROJECT_DIR / config_path}")


@app.command("all")
def run_all(
    max_aya_examples: Optional[int] = typer.Option(None, help="Optional cap for Aya examples."),
    max_soynade_examples: Optional[int] = typer.Option(None, help="Optional cap for Soynade examples."),
    validation_ratio: float = typer.Option(0.05, help="Validation ratio. 0.05 means 5%."),
    eval_ratio: float = typer.Option(0.05, help="Evaluation ratio. 0.05 means 5%."),
    seed: int = typer.Option(42, help="Deterministic split seed."),
) -> None:
    """Download sources, then prepare chat files, splits, and YAML config."""
    synth_input = str(DEFAULT_SYNTH_FILE.relative_to(PROJECT_DIR))
    config_path = str(DEFAULT_CONFIG_FILE.relative_to(PROJECT_DIR))
    download(
        max_aya_examples=max_aya_examples,
        max_soynade_examples=max_soynade_examples,
        synth_input=synth_input,
    )
    prepare(
        validation_ratio=validation_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        min_validation=1,
        min_eval=1,
        synth_input=synth_input,
        config_path=config_path,
    )


if __name__ == "__main__":
    app()
