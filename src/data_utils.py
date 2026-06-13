"""Data loading and chat-template conversion for Wolof instruction tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset


SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in Wolof and Senegalese/African "
    "contexts. Answer in clear Wolof. Keep the response useful, factual, and "
    "appropriate for the requested category. Do not write hidden reasoning. "
    "Never output <think> or </think> tags."
)


REQUIRED_FIELDS = ("instruction", "input", "output")
MINIMAL_REQUIRED_FIELDS = ("instruction", "output")


def load_instruction_examples(
    path: str | Path,
    default_input: str | None = None,
) -> List[Dict[str, str]]:
    """Load a JSONL file or a JSON list with instruction/input/output rows."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if data_path.suffix == ".jsonl":
        rows = []
        with data_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append(
                    _validate_example(
                        json.loads(line),
                        line_no=line_no,
                        default_input=default_input,
                    )
                )
        return rows

    if data_path.suffix == ".json":
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON input must be a list of examples.")
        return [
            _validate_example(row, line_no=i + 1, default_input=default_input)
            for i, row in enumerate(raw)
        ]

    raise ValueError("Dataset must be .jsonl or .json")


def _validate_example(
    example: Dict[str, Any],
    line_no: int,
    default_input: str | None = None,
) -> Dict[str, str]:
    missing_minimal = [field for field in MINIMAL_REQUIRED_FIELDS if field not in example]
    if missing_minimal:
        raise ValueError(f"Example {line_no} is missing fields: {missing_minimal}")

    if "input" not in example and default_input is not None:
        example = {**example, "input": default_input}

    missing = [field for field in REQUIRED_FIELDS if field not in example]
    if missing:
        raise ValueError(f"Example {line_no} is missing fields: {missing}")

    clean = {field: str(example.get(field, "")).strip() for field in REQUIRED_FIELDS}
    if not clean["instruction"]:
        raise ValueError(f"Example {line_no} has an empty instruction.")
    if not clean["output"]:
        raise ValueError(f"Example {line_no} has an empty output.")
    return clean


def example_to_messages(
    example: Dict[str, str],
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Convert one instruction/input/output row to chat messages.

    The dataset schema is:
    - instruction: the user question or task
    - input: the category/context, for example education, agriculture, health
    - output: the expected Wolof answer
    """
    category = example.get("input", "").strip()
    instruction = example["instruction"].strip()

    user_content = f"Context category: {category}\nInstruction: {instruction}"
    if not category:
        user_content = f"Instruction: {instruction}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"].strip()},
    ]


def messages_to_text(messages: List[Dict[str, str]], tokenizer) -> str:
    """Apply tokenizer chat template, with a fallback for tokenizers without one."""
    return render_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def render_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    """Render messages while disabling Qwen3 thinking mode when supported."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    return fallback_chat_template(messages, add_generation_prompt=add_generation_prompt)


def fallback_chat_template(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Simple ChatML-style fallback when the tokenizer has no chat template."""
    rendered = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        rendered.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "".join(rendered)


def build_sft_dataset(
    examples: Iterable[Dict[str, str]],
    tokenizer,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dataset:
    """Return a Hugging Face Dataset with text/messages columns for SFT."""
    rows = []
    for example in examples:
        messages = example_to_messages(example, system_prompt=system_prompt)
        rows.append(
            {
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "output": example["output"],
                "messages": messages,
                "text": messages_to_text(messages, tokenizer),
            }
        )
    return Dataset.from_list(rows)


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(rows: Iterable[Dict[str, Any]], path: str | Path, indent: int = 2) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
