"""Run inference with a Wolof LoRA adapter on Qwen3 or Gemma."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_utils import SYSTEM_PROMPT, render_chat_template
from src.model_registry import resolve_adapter_dir, resolve_model_name


SCRIPT_DIR = Path(__file__).resolve().parent
app = typer.Typer(
    help="Run inference with a Wolof LoRA adapter on Qwen3 or Gemma.",
    pretty_exceptions_show_locals=False,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def resolve_relative_to_script(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_user_message(instruction: str, category: str) -> str:
    instruction = instruction.strip()
    category = category.strip()
    if category:
        return f"Context category: {category}\nInstruction: {instruction}"
    return f"Instruction: {instruction}"


def build_messages(
    instruction: str,
    category: str,
    system_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt.strip()}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": format_user_message(instruction, category)})
    return messages


def render_generation_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return render_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def clean_model_answer(answer: str) -> str:
    """Remove Qwen thinking tags if an old adapter still emits them."""
    answer = THINK_BLOCK_RE.sub("", answer)
    answer = THINK_TAG_RE.sub("", answer)
    answer = answer.replace("_", " ")
    answer = re.sub(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Ý])", " ", answer)
    answer = re.sub(
        r"(?im)^(context category|instruction|student question|task)\s*:.*$",
        "",
        answer,
    )
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def generate_answer(
    model,
    tokenizer,
    device: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    prompt = render_generation_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_model_answer(answer)


def run_text_mode(
    model,
    tokenizer,
    device: str,
    instruction: str,
    category: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> None:
    messages = build_messages(
        instruction=instruction,
        category=category,
        system_prompt=system_prompt,
    )
    answer = generate_answer(
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

    typer.echo("\n=== Prompt ===")
    typer.echo(f"Category: {category}")
    typer.echo(f"Instruction: {instruction}")
    typer.echo("\n=== Model answer ===")
    typer.echo(answer)


def run_chat_mode(
    model,
    tokenizer,
    device: str,
    category: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> None:
    history: List[Dict[str, str]] = []
    typer.echo("\n=== Chat mode ===")
    typer.echo("Type /exit to stop, /reset to clear the conversation.\n")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nExiting chat.")
            break

        if not user_text:
            continue

        command = user_text.lower()
        if command in {"/exit", "exit", "quit", "/quit"}:
            typer.echo("Exiting chat.")
            break
        if command in {"/reset", "reset"}:
            history.clear()
            typer.echo("Conversation history cleared.\n")
            continue

        messages = build_messages(
            instruction=user_text,
            category=category,
            system_prompt=system_prompt,
            history=history,
        )
        answer = generate_answer(
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

        typer.echo("\nAssistant>")
        typer.echo(answer)
        typer.echo()
        history.append({"role": "user", "content": format_user_message(user_text, category)})
        history.append({"role": "assistant", "content": answer})


@app.command()
def main(
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
        help="LoRA adapter directory. Use auto to derive it from --model-choice.",
    ),
    instruction: str = typer.Option(
        "Naka la ndongo bu am jafe-jafe ci algebra man a jàng bu baax?",
        help="Instruction/question to answer in Wolof.",
    ),
    category: str = typer.Option("education", help="Context category."),
    max_new_tokens: int = typer.Option(160, help="Maximum generated tokens."),
    temperature: float = typer.Option(0.2, help="Sampling temperature. Use 0 for greedy decoding."),
    top_p: float = typer.Option(0.9, help="Nucleus sampling probability."),
    repetition_penalty: float = typer.Option(1.12, help="Penalty to reduce repeated phrases."),
    no_repeat_ngram_size: int = typer.Option(3, help="Block repeated n-grams of this size."),
    system_prompt: str = typer.Option(SYSTEM_PROMPT, help="System prompt."),
    chat: bool = typer.Option(False, "--chat", help="Start an interactive chat session."),
) -> None:
    device = pick_device()
    base_model_name = resolve_model_name(model_choice, model)
    adapter_path = resolve_relative_to_script(
        resolve_adapter_dir(model_choice, adapter, SCRIPT_DIR)
    )
    typer.echo(f"Model choice: {model_choice}")
    typer.echo(f"Base model: {base_model_name}")
    typer.echo(f"Adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    if device != "cuda":
        model = model.to(device)
    model.eval()

    if chat:
        run_chat_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            category=category,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return

    run_text_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        instruction=instruction,
        category=category,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


if __name__ == "__main__":
    app()
