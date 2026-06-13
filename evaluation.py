"""Evaluate Wolof LoRA outputs with metrics from Session 8."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer

from src.data_utils import SYSTEM_PROMPT
from src.model_registry import resolve_adapter_dir, resolve_model_name


SCRIPT_DIR = Path(__file__).resolve().parent
app = typer.Typer(
    help="Evaluate Wolof instruction-tuning outputs with classic LLM metrics.",
    pretty_exceptions_show_locals=False,
)


def resolve_relative_to_script(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate


def load_eval_examples(path: str) -> List[Dict[str, str]]:
    data_path = resolve_relative_to_script(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {data_path}")

    examples = []
    with data_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(validate_eval_example(row, line_no))
    return examples


def validate_eval_example(row: Dict[str, Any], line_no: int) -> Dict[str, str]:
    required = ("instruction", "input", "reference")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"Example {line_no} is missing required fields: {missing}")

    clean = {
        "instruction": str(row.get("instruction", "")).strip(),
        "input": str(row.get("input", "")).strip(),
        "reference": str(row.get("reference", "")).strip(),
        "prediction": str(row.get("prediction", "")).strip(),
    }
    if not clean["instruction"]:
        raise ValueError(f"Example {line_no} has an empty instruction.")
    if not clean["reference"]:
        raise ValueError(f"Example {line_no} has an empty reference.")
    return clean


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def normalized_text(text: str) -> str:
    return " ".join(tokenize(text))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalized_text(prediction) == normalized_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: List[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(prediction: str, reference: str, max_order: int = 4) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for order in range(1, max_order + 1):
        pred_ngrams = ngrams(pred_tokens, order)
        ref_ngrams = ngrams(ref_tokens, order)
        if not pred_ngrams:
            precisions.append(1.0)
            continue
        overlap = sum((pred_ngrams & ref_ngrams).values())
        precisions.append((overlap + 1) / (sum(pred_ngrams.values()) + 1))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    return brevity_penalty * geo_mean


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "bleu": bleu_score(prediction, reference),
        "rouge_l": rouge_l_f1(prediction, reference),
    }


def average_metric(rows: Iterable[Dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    if not values:
        return 0.0
    return sum(values) / len(values)


def load_lora_model(model_name: str, adapter: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inference_lora_qwen3_wolof import pick_device, pick_dtype

    device = pick_device()
    adapter_path = resolve_relative_to_script(adapter)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_predictions(
    examples: List[Dict[str, str]],
    model_name: str,
    adapter: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    system_prompt: str,
) -> List[Dict[str, str]]:
    from inference_lora_qwen3_wolof import build_messages, generate_answer

    model, tokenizer, device = load_lora_model(model_name, adapter)
    generated_rows = []
    for i, row in enumerate(examples, start=1):
        typer.echo(f"Generating prediction {i}/{len(examples)}...")
        messages = build_messages(
            instruction=row["instruction"],
            category=row["input"],
            system_prompt=system_prompt,
        )
        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated = dict(row)
        generated["prediction"] = prediction
        generated_rows.append(generated)
    return generated_rows


def sequence_perplexity(model, tokenizer, device: str, text: str) -> float:
    import torch

    encoded = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return float(torch.exp(loss).detach().cpu())


def add_perplexity(
    scored_rows: List[Dict[str, Any]],
    model_name: str,
    adapter: str,
    max_examples: int,
) -> None:
    model, tokenizer, device = load_lora_model(model_name, adapter)
    for i, row in enumerate(scored_rows[:max_examples], start=1):
        typer.echo(f"Computing reference perplexity {i}/{min(len(scored_rows), max_examples)}...")
        row["reference_ppl"] = sequence_perplexity(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=row["reference"],
        )


def print_report(scored_rows: List[Dict[str, Any]], show_examples: int) -> None:
    metric_keys = ["exact_match", "token_f1", "bleu", "rouge_l"]
    if any("reference_ppl" in row for row in scored_rows):
        metric_keys.append("reference_ppl")

    typer.echo("\n=== Aggregate metrics ===")
    for key in metric_keys:
        values = [row[key] for row in scored_rows if key in row]
        if values:
            typer.echo(f"{key:>15}: {sum(values) / len(values):.4f}")

    typer.echo("\n=== Example-level scores ===")
    for idx, row in enumerate(scored_rows[:show_examples], start=1):
        typer.echo(f"\n[{idx}] category={row['input']}")
        typer.echo(f"Q: {row['instruction']}")
        typer.echo(f"Reference:  {row['reference']}")
        typer.echo(f"Prediction: {row['prediction']}")
        typer.echo(
            "Scores: "
            f"EM={row['exact_match']:.2f}, "
            f"F1={row['token_f1']:.2f}, "
            f"BLEU={row['bleu']:.2f}, "
            f"ROUGE-L={row['rouge_l']:.2f}"
        )
        if "reference_ppl" in row:
            typer.echo(f"Reference PPL: {row['reference_ppl']:.2f}")


JUDGE_PROMPT_TEMPLATE = """You are an evaluator for a Wolof instruction-tuned model.
Score the model answer from 1 to 5 on:
- correctness: factual correctness against the reference,
- groundedness: whether the answer is supported by the expected answer,
- style: clear Wolof, concise, appropriate to the category,
- repetition: penalize repeated words or loops.

Return only JSON with keys correctness, groundedness, style, repetition, reason.

Category: {category}
Question: {instruction}
Reference answer: {reference}
Model answer: {prediction}
"""


def print_judge_prompt(row: Dict[str, Any]) -> None:
    typer.echo("\n=== LLM-as-a-Judge prompt example ===")
    typer.echo(
        JUDGE_PROMPT_TEMPLATE.format(
            category=row["input"],
            instruction=row["instruction"],
            reference=row["reference"],
            prediction=row["prediction"],
        )
    )


@app.command()
def main(
    data: str = typer.Option(
        "data/wolof_eval_examples.jsonl",
        help="Evaluation JSONL with instruction/input/reference and optional prediction.",
    ),
    generate: bool = typer.Option(
        False,
        "--generate",
        help="Generate predictions with the LoRA adapter instead of using prediction fields.",
    ),
    model_choice: str = typer.Option(
        "qwen",
        "--model-choice",
        "--model-family",
        help="Preset base model: qwen or gemma.",
    ),
    model: str | None = typer.Option(
        None,
        help="Optional explicit base model name or path. Overrides --model-choice.",
    ),
    adapter: str = typer.Option(
        "auto",
        help="LoRA adapter directory used with --generate or --perplexity.",
    ),
    max_new_tokens: int = typer.Option(160, help="Maximum generated tokens."),
    temperature: float = typer.Option(0.0, help="Use 0 for deterministic evaluation."),
    top_p: float = typer.Option(0.9, help="Nucleus sampling probability."),
    repetition_penalty: float = typer.Option(1.12, help="Penalty to reduce repeated phrases."),
    no_repeat_ngram_size: int = typer.Option(3, help="Block repeated n-grams of this size."),
    system_prompt: str = typer.Option(SYSTEM_PROMPT, help="System prompt."),
    perplexity: bool = typer.Option(False, "--perplexity", help="Compute reference perplexity."),
    perplexity_max_examples: int = typer.Option(5, help="Limit expensive perplexity examples."),
    show_examples: int = typer.Option(8, help="Number of scored examples to print."),
    print_judge: bool = typer.Option(
        False,
        "--print-judge-prompt",
        help="Print an LLM-as-a-Judge prompt for the first evaluated example.",
    ),
) -> None:
    resolved_model = resolve_model_name(model_choice, model)
    resolved_adapter = resolve_adapter_dir(model_choice, adapter, SCRIPT_DIR)
    examples = load_eval_examples(data)

    if generate:
        examples = generate_predictions(
            examples=examples,
            model_name=resolved_model,
            adapter=resolved_adapter,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            system_prompt=system_prompt,
        )

    missing_predictions = [i for i, row in enumerate(examples, start=1) if not row["prediction"]]
    if missing_predictions:
        raise ValueError(
            "Missing prediction fields for examples "
            f"{missing_predictions}. Use --generate or add predictions to the eval file."
        )

    scored_rows: List[Dict[str, Any]] = []
    for row in examples:
        metrics = score_prediction(row["prediction"], row["reference"])
        scored_rows.append({**row, **metrics})

    if perplexity:
        add_perplexity(
            scored_rows=scored_rows,
            model_name=resolved_model,
            adapter=resolved_adapter,
            max_examples=perplexity_max_examples,
        )

    print_report(scored_rows, show_examples=show_examples)
    if print_judge and scored_rows:
        print_judge_prompt(scored_rows[0])


if __name__ == "__main__":
    app()
