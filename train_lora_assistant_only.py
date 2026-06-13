"""Assistant-only LoRA training with -100 masking for chat instruction data."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import typer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.checkpoint_manager import evaluate_prune_and_export, latest_checkpoint
from src.data_utils import (
    SYSTEM_PROMPT,
    example_to_messages,
    load_instruction_examples,
    render_chat_template,
)
from src.model_registry import resolve_logging_dir, resolve_model_name, resolve_output_dir
from src.training_monitoring import save_training_artifacts


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

IGNORE_INDEX = -100
SCRIPT_DIR = Path(__file__).resolve().parent
app = typer.Typer(
    help="Train Qwen/Gemma LoRA with assistant-only loss masking.",
    pretty_exceptions_show_locals=False,
)


@dataclass
class EncodedStats:
    kept_examples: int
    skipped_examples: int
    avg_total_tokens: float
    avg_supervised_tokens: float
    max_total_tokens: int
    max_supervised_tokens: int


@dataclass
class AssistantOnlyDataCollator:
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class AssistantOnlyChatDataset(Dataset):
    """Tokenized chat dataset where only assistant answer tokens contribute to loss."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
    ) -> None:
        self.features: List[Dict[str, List[int]]] = []
        total_tokens = []
        supervised_tokens = []
        skipped = 0

        for row in rows:
            encoded = encode_assistant_only_example(
                messages=row["messages"],
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
            if encoded is None:
                skipped += 1
                continue
            self.features.append(encoded)
            total_tokens.append(len(encoded["input_ids"]))
            supervised_tokens.append(sum(label != IGNORE_INDEX for label in encoded["labels"]))

        if not self.features:
            raise ValueError("No trainable examples left after tokenization/truncation.")

        self.stats = EncodedStats(
            kept_examples=len(self.features),
            skipped_examples=skipped,
            avg_total_tokens=sum(total_tokens) / len(total_tokens),
            avg_supervised_tokens=sum(supervised_tokens) / len(supervised_tokens),
            max_total_tokens=max(total_tokens),
            max_supervised_tokens=max(supervised_tokens),
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]


def resolve_relative_to_script(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate


def tensorboard_is_available() -> bool:
    return (
        importlib.util.find_spec("tensorboard") is not None
        or importlib.util.find_spec("tensorboardX") is not None
    )


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def move_model_if_needed(model):
    if torch.cuda.is_available():
        return model
    if torch.backends.mps.is_available():
        return model.to("mps")
    return model


def clear_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def resolve_resume_checkpoint(value: Optional[str], output_dir: Path) -> Optional[Path]:
    if not value:
        return None

    if value.strip().lower() in {"latest", "auto"}:
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* directory found in {output_dir}")
        return checkpoint

    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        output_dir / raw,
        resolve_relative_to_script(value),
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Resume checkpoint not found. Checked: {checked}")


def load_training_rows(
    data_path: Path,
    default_category: str,
    system_prompt: str,
    max_train_examples: Optional[int],
) -> List[Dict[str, Any]]:
    """Load either messages+metadata JSON or instruction/input/output data."""
    import json

    default_input = default_category.strip() or None
    rows: List[Dict[str, Any]] = []

    if data_path.suffix == ".json":
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON training data must be a list.")
        if raw and isinstance(raw[0], dict) and "messages" in raw[0]:
            for item in raw:
                messages = validate_messages(item.get("messages", []))
                rows.append({"messages": messages, "metadata": item.get("metadata", {})})
            return maybe_limit_rows(rows, max_train_examples)

    examples = load_instruction_examples(data_path, default_input=default_input)
    for example in examples:
        rows.append(
            {
                "messages": example_to_messages(example, system_prompt=system_prompt),
                "metadata": {
                    "source": data_path.name,
                    "category": example.get("input", ""),
                },
            }
        )
    return maybe_limit_rows(rows, max_train_examples)


def load_training_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config. Install it with: python -m pip install PyYAML"
        ) from exc

    path = resolve_relative_to_script(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Training config must be a YAML mapping.")
    return loaded


def config_list(config_data: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = config_data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Config key {key!r} must be a list.")
    rows = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Each item in config key {key!r} must be a mapping.")
        if "path" not in item:
            raise ValueError(f"Each item in config key {key!r} must contain a path.")
        rows.append(item)
    return rows


def apply_config_hparams(config_data: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    training = config_data.get("training", {})
    if not isinstance(training, dict):
        return values
    merged = dict(values)
    for key in list(merged):
        if key in training:
            merged[key] = training[key]
    return merged


def load_rows_from_config_entries(
    entries: Sequence[Dict[str, Any]],
    system_prompt: str,
    fallback_default_category: str,
    max_train_examples: Optional[int],
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for entry in entries:
        path = resolve_relative_to_script(str(entry["path"]))
        name = str(entry.get("name", path.stem))
        default_category = str(entry.get("default_category", fallback_default_category))
        source_limit = entry.get("max_examples")
        rows = load_training_rows(
            data_path=path,
            default_category=default_category,
            system_prompt=system_prompt,
            max_train_examples=int(source_limit) if source_limit else None,
        )
        for row in rows:
            metadata = row.setdefault("metadata", {})
            metadata.setdefault("source", name)
            metadata.setdefault("config_path", str(path))
        typer.echo(f"Loaded {len(rows)} rows from config source {name}: {path}")
        all_rows.extend(rows)
    return maybe_limit_rows(all_rows, max_train_examples)


def maybe_limit_rows(rows: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def validate_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each messages-format row must contain a non-empty messages list.")

    clean: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object with role/content.")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        if not content:
            raise ValueError(f"Empty content for role {role!r}")
        clean.append({"role": role, "content": content})

    if clean[-1]["role"] != "assistant":
        raise ValueError("The final message must be the assistant answer.")
    return clean


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    size = min(len(left), len(right))
    for idx in range(size):
        if left[idx] != right[idx]:
            return idx
    return size


def encode_assistant_only_example(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int,
) -> Optional[Dict[str, List[int]]]:
    validate_messages(messages)
    prompt_messages = messages[:-1]

    prompt_text = render_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = render_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids:
        return None

    prefix_len = common_prefix_len(prompt_ids, full_ids)
    labels = list(full_ids)
    labels[:prefix_len] = [IGNORE_INDEX] * prefix_len

    if len(full_ids) > max_seq_length:
        overflow = len(full_ids) - max_seq_length
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]

    if not any(label != IGNORE_INDEX for label in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_training_arguments(
    output_dir: Path,
    logging_dir: Path,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    eval_steps: int,
    seed: int,
    tensorboard: bool,
    has_eval_dataset: bool,
) -> TrainingArguments:
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if tensorboard:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    cfg: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "seed": seed,
        "remove_unused_columns": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": "tensorboard" if tensorboard else "none",
    }

    optional_cfg = {
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    }
    for key, value in optional_cfg.items():
        if key in supported:
            cfg[key] = value
    if has_eval_dataset:
        if "eval_strategy" in supported:
            cfg["eval_strategy"] = "steps"
        elif "evaluation_strategy" in supported:
            cfg["evaluation_strategy"] = "steps"
        if "eval_steps" in supported:
            cfg["eval_steps"] = eval_steps
    return TrainingArguments(**cfg)


def build_trainer(
    model_obj,
    training_args: TrainingArguments,
    dataset: Dataset,
    validation_dataset: Optional[Dataset],
    collator: AssistantOnlyDataCollator,
    tokenizer,
) -> Trainer:
    """Build Trainer while supporting tokenizer/processing_class API variants."""
    supported = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model_obj,
        "args": training_args,
        "train_dataset": dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
    }

    if "tokenizer" in supported:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


@app.command()
def main(
    config: Optional[str] = typer.Option(
        None,
        help="YAML config with training_data, validation_data, eval_data, and hyperparameters.",
    ),
    use_config_hparams: bool = typer.Option(
        True,
        "--use-config-hparams/--no-use-config-hparams",
        help="When --config is set, use training hyperparameters from the YAML file.",
    ),
    model_choice: str = typer.Option(
        "qwen",
        "--model-choice",
        "--model-family",
        help="Preset base model: qwen or gemma.",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Optional explicit base model name or path. Overrides --model-choice.",
    ),
    data: str = typer.Option(
        "data/chat_template_wol_train.json",
        help="Messages+metadata JSON or instruction JSONL/JSON training data.",
    ),
    default_category: str = typer.Option(
        "",
        help="Category to inject when instruction data has no input/category field.",
    ),
    output_dir: str = typer.Option(
        "auto",
        help="Adapter output directory. Use auto to derive it from --model-choice.",
    ),
    max_train_examples: Optional[int] = typer.Option(
        None,
        help="Optional small subset for a quick pipeline test.",
    ),
    max_seq_length: int = typer.Option(1024, min=128, help="Maximum sequence length."),
    epochs: float = typer.Option(3.0, help="Number of training epochs."),
    batch_size: int = typer.Option(1, help="Per-device training batch size."),
    grad_accum: int = typer.Option(8, help="Gradient accumulation steps."),
    learning_rate: float = typer.Option(1e-4, help="Learning rate."),
    warmup_steps: int = typer.Option(50, min=0, help="Warmup steps."),
    weight_decay: float = typer.Option(0.01, min=0.0, help="AdamW weight decay."),
    lora_r: int = typer.Option(16, min=1, help="LoRA rank."),
    lora_alpha: int = typer.Option(32, min=1, help="LoRA alpha."),
    lora_dropout: float = typer.Option(0.05, min=0.0, help="LoRA dropout."),
    logging_dir: str = typer.Option(
        "auto",
        help="TensorBoard log directory. Use auto to derive it from --model-choice.",
    ),
    logging_steps: int = typer.Option(10, min=1, help="Training log frequency."),
    save_steps: int = typer.Option(100, min=1, help="Checkpoint save frequency."),
    save_total_limit: int = typer.Option(5, min=1, help="Maximum checkpoints kept."),
    eval_steps: int = typer.Option(100, min=1, help="Validation loss frequency when validation data exists."),
    tensorboard: bool = typer.Option(
        True,
        "--tensorboard/--no-tensorboard",
        help="Enable TensorBoard logging when installed.",
    ),
    gradient_checkpointing: bool = typer.Option(
        True,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Reduce memory by recomputing activations during backward pass.",
    ),
    eval_data: str = typer.Option(
        "data/wolof_eval_examples.jsonl",
        help="Held-out JSONL used to rank checkpoints after training.",
    ),
    checkpoint_eval: bool = typer.Option(
        True,
        "--checkpoint-eval/--no-checkpoint-eval",
        help="Evaluate checkpoints, export best_adapter/latest_adapter.",
    ),
    checkpoint_eval_max_examples: int = typer.Option(5, min=1, help="Checkpoint eval examples."),
    checkpoint_eval_max_new_tokens: int = typer.Option(96, min=16, help="Checkpoint eval tokens."),
    keep_checkpoints: int = typer.Option(5, min=1, help="Checkpoints kept after pruning."),
    resume_from_checkpoint: Optional[str] = typer.Option(
        None,
        "--resume-from-checkpoint",
        help="Resume Trainer state from a checkpoint path, checkpoint name, or 'latest'.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    system_prompt: str = typer.Option(SYSTEM_PROMPT, help="System prompt for instruction rows."),
) -> None:
    config_data = load_training_config(config)
    if config_data and use_config_hparams:
        merged_hparams = apply_config_hparams(
            config_data,
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "max_seq_length": max_seq_length,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "eval_steps": eval_steps,
            },
        )
        epochs = float(merged_hparams["epochs"])
        batch_size = int(merged_hparams["batch_size"])
        grad_accum = int(merged_hparams["grad_accum"])
        max_seq_length = int(merged_hparams["max_seq_length"])
        learning_rate = float(merged_hparams["learning_rate"])
        warmup_steps = int(merged_hparams["warmup_steps"])
        weight_decay = float(merged_hparams["weight_decay"])
        lora_r = int(merged_hparams["lora_r"])
        lora_alpha = int(merged_hparams["lora_alpha"])
        lora_dropout = float(merged_hparams["lora_dropout"])
        logging_steps = int(merged_hparams["logging_steps"])
        save_steps = int(merged_hparams["save_steps"])
        save_total_limit = int(merged_hparams["save_total_limit"])
        eval_steps = int(merged_hparams["eval_steps"])

    resolved_model = resolve_model_name(model_choice, model)
    output_dir_path = resolve_relative_to_script(resolve_output_dir(model_choice, output_dir))
    logging_dir_path = resolve_relative_to_script(resolve_logging_dir(model_choice, logging_dir))
    data_path = resolve_relative_to_script(data)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logging_dir_path.mkdir(parents=True, exist_ok=True)

    if tensorboard and not tensorboard_is_available():
        typer.echo(
            "TensorBoard is not installed. Continuing with --no-tensorboard. "
            "Install it with: python -m pip install tensorboard",
            err=True,
        )
        tensorboard = False

    typer.echo("Assistant-only training enabled.")
    typer.echo("Labels are -100 for system/user/padding tokens; loss is only on assistant outputs.")
    if config_data and config:
        typer.echo(f"Training config: {resolve_relative_to_script(config)}")
    typer.echo(f"Model choice: {model_choice}")
    typer.echo(f"Base model: {resolved_model}")
    if config_data:
        typer.echo("Training data: YAML training_data entries")
    else:
        typer.echo(f"Training data: {data_path}")
    typer.echo(f"Output directory: {output_dir_path}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_entries = config_list(config_data, "training_data") if config_data else []
    if training_entries:
        rows = load_rows_from_config_entries(
            entries=training_entries,
            system_prompt=system_prompt,
            fallback_default_category=default_category,
            max_train_examples=max_train_examples,
        )
    else:
        rows = load_training_rows(
            data_path=data_path,
            default_category=default_category,
            system_prompt=system_prompt,
            max_train_examples=max_train_examples,
        )
    dataset = AssistantOnlyChatDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    stats = dataset.stats
    typer.echo(
        "Tokenized examples: "
        f"kept={stats.kept_examples}, skipped={stats.skipped_examples}, "
        f"avg_total={stats.avg_total_tokens:.1f}, "
        f"avg_supervised={stats.avg_supervised_tokens:.1f}, "
        f"max_total={stats.max_total_tokens}, "
        f"max_supervised={stats.max_supervised_tokens}"
    )
    validation_entries = config_list(config_data, "validation_data") if config_data else []
    validation_dataset = None
    if validation_entries:
        validation_rows = load_rows_from_config_entries(
            entries=validation_entries,
            system_prompt=system_prompt,
            fallback_default_category=default_category,
            max_train_examples=None,
        )
        validation_dataset = AssistantOnlyChatDataset(
            rows=validation_rows,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        validation_stats = validation_dataset.stats
        typer.echo(
            "Validation examples: "
            f"kept={validation_stats.kept_examples}, "
            f"avg_supervised={validation_stats.avg_supervised_tokens:.1f}"
        )

    model_obj = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model_obj = move_model_if_needed(model_obj)
    model_obj.config.use_cache = False
    if gradient_checkpointing and hasattr(model_obj, "gradient_checkpointing_enable"):
        model_obj.gradient_checkpointing_enable()
    if gradient_checkpointing and hasattr(model_obj, "enable_input_require_grads"):
        model_obj.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model_obj = get_peft_model(model_obj, lora_cfg)
    model_obj.print_trainable_parameters()

    training_args = build_training_arguments(
        output_dir=output_dir_path,
        logging_dir=logging_dir_path,
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_steps=eval_steps,
        seed=seed,
        tensorboard=tensorboard,
        has_eval_dataset=validation_dataset is not None,
    )
    collator = AssistantOnlyDataCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = build_trainer(
        model_obj=model_obj,
        training_args=training_args,
        dataset=dataset,
        validation_dataset=validation_dataset,
        collator=collator,
        tokenizer=tokenizer,
    )
    resume_checkpoint_path = resolve_resume_checkpoint(resume_from_checkpoint, output_dir_path)
    if resume_checkpoint_path is not None:
        typer.echo(f"Resuming from checkpoint: {resume_checkpoint_path}")
    trainer.train(
        resume_from_checkpoint=str(resume_checkpoint_path)
        if resume_checkpoint_path is not None
        else None
    )

    model_obj.save_pretrained(output_dir_path)
    tokenizer.save_pretrained(output_dir_path)
    typer.echo(f"Saved LoRA adapter and tokenizer to {output_dir_path}")

    log_history = list(trainer.state.log_history)
    save_training_artifacts(log_history, output_dir_path)
    typer.echo(f"Saved training history and curves to {output_dir_path}")
    if tensorboard:
        typer.echo(f"TensorBoard logs: {logging_dir_path}")

    del trainer
    del model_obj
    clear_accelerator_cache()

    if checkpoint_eval:
        configured_eval = config_data.get("eval_data", {}) if config_data else {}
        if isinstance(configured_eval, dict) and configured_eval.get("combined_path"):
            eval_data = str(configured_eval["combined_path"])
        eval_data_path = resolve_relative_to_script(eval_data)
        report = evaluate_prune_and_export(
            output_dir=output_dir_path,
            model_name=resolved_model,
            eval_data=eval_data_path,
            keep_count=keep_checkpoints,
            max_examples=checkpoint_eval_max_examples,
            max_new_tokens=checkpoint_eval_max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            system_prompt=system_prompt,
        )
        typer.echo(f"Kept checkpoints: {report.get('kept_checkpoints', [])}")
        typer.echo(f"Removed checkpoints: {report.get('removed_checkpoints', [])}")
        if report.get("best_source"):
            typer.echo(f"Best checkpoint by evaluation: {report['best_source']}")
            typer.echo(f"Clean best adapter export: {output_dir_path / 'best_adapter'}")
        if report.get("latest_source"):
            typer.echo(f"Latest checkpoint export: {output_dir_path / 'latest_adapter'}")


if __name__ == "__main__":
    app()
