"""Gradio Space for the Wolof LoRA assistant.

Loads the base model + LoRA adapter from the Hugging Face Hub, and uses
`CategoryContextStateMachine` (src/context_state_machine.py) for:

- category- and source-aware few-shot retrieval from the training data,
- a lexical-overlap confidence score shown to the user,
- a keyword-based safety filter that makes the app refuse to generate for
  unsafe categories (medication dosage, self-harm, weapons, etc.) and shows
  a limitation message instead,
- showing the retrieved examples used to build the prompt, for
  transparency.

Set MODEL_REPO_ID to your pushed LoRA adapter repo before deploying.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make src/ importable when running from the Space root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context_state_machine import (  # noqa: E402
    CategoryContextStateMachine,
    DEFAULT_WEB_CATEGORIES,
    load_training_rows,
)


BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "YOUR_USERNAME/YOUR_LORA_MODEL_REPO")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant specialized in Wolof and Senegalese/African "
    "contexts. Answer in clear Wolof. Keep the response useful, factual, and "
    "appropriate for the requested category. Do not write hidden reasoning. "
    "Never output <think> or </think> tags.",
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DATA_CANDIDATES = [
    PROJECT_DIR / "data" / "syntetic_wolof_instruct_data.jsonl",
    PROJECT_DIR / "data" / "splits" / "eval_all.jsonl",
]


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_answer(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_TAG_RE.sub("", text)
    return text.strip()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, MODEL_REPO_ID)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def load_state_machine() -> CategoryContextStateMachine | None:
    for candidate in TRAINING_DATA_CANDIDATES:
        if candidate.exists():
            try:
                rows = load_training_rows(candidate)
                if rows:
                    return CategoryContextStateMachine(rows, categories=DEFAULT_WEB_CATEGORIES)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not load training rows from {candidate}: {exc}")
    print("No training data found for the context state machine; retrieval disabled.")
    return None


MODEL, TOKENIZER = load_model()
DEVICE = pick_device()
STATE_MACHINE = load_state_machine()
CATEGORY_CHOICES = STATE_MACHINE.available_categories() if STATE_MACHINE else ["All"]


def render_prompt(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
    try:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_generation(prompt_text: str, max_new_tokens: int, temperature: float) -> str:
    inputs = TOKENIZER(prompt_text, return_tensors="pt").to(DEVICE)
    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": TOKENIZER.eos_token_id,
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 3,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, **kwargs)
    generated = output_ids[0, inputs["input_ids"].shape[-1]:]
    return clean_answer(TOKENIZER.decode(generated, skip_special_tokens=True))


def format_examples(examples: list[dict]) -> str:
    if not examples:
        return "_No retrieved examples._"
    lines = []
    for i, ex in enumerate(examples, start=1):
        lines.append(
            f"**{i}.** *(source: {ex.get('source', 'unknown')}, "
            f"category: {ex.get('input', '?')})*\n"
            f"- Q: {ex['instruction']}\n"
            f"- A: {ex['output']}"
        )
    return "\n\n".join(lines)


def generate(
    user_message: str,
    category: str,
    max_new_tokens: int,
    temperature: float,
    use_retrieval: bool,
):
    if not user_message.strip():
        return "Please enter a prompt.", "_No retrieved examples._", ""

    if STATE_MACHINE is None or not use_retrieval:
        prompt = render_prompt(user_message)
        answer = run_generation(prompt, max_new_tokens, temperature)
        return answer, "_Retrieval disabled._", ""

    prep = STATE_MACHINE.prepare(category, user_message, k=4)

    if not prep["should_answer"]:
        return prep["limitation_message"], "_No retrieved examples (unsafe category)._", "should_answer = False (unsafe topic)"

    instruction_text = prep["augmented_instruction"]
    prompt = render_prompt(instruction_text)
    answer = run_generation(prompt, max_new_tokens, temperature)

    examples_md = format_examples(prep["examples"])
    confidence = prep["confidence"]
    status = f"retrieval confidence = {confidence:.2f}"
    if prep["limitation_message"]:
        answer = f"{prep['limitation_message']}\n\n---\n\n{answer}"

    return answer, examples_md, status


with gr.Blocks(title="Real LLM Deployment: Wolof Assistant") as demo:
    gr.Markdown(
        "# Real LLM Deployment: Fine-Tuned Wolof Assistant\n"
        f"Base model: `{BASE_MODEL}` + LoRA adapter `{MODEL_REPO_ID}`.\n\n"
        "**Limitation:** this is a small (0.6B) fine-tuned demo model for "
        "education, agriculture, health-FAQ, transport, and culture topics "
        "in Wolof. It is not a medical, legal, or safety authority and will "
        "refuse questions about medication dosages, self-harm, or weapons."
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(label="Your question (French, English, or Wolof)", lines=4)
            category_dd = gr.Dropdown(
                choices=CATEGORY_CHOICES,
                value=CATEGORY_CHOICES[0],
                label="Category",
            )
            use_retrieval_cb = gr.Checkbox(
                value=STATE_MACHINE is not None,
                label="Use context state machine (retrieval + safety filter)",
                interactive=STATE_MACHINE is not None,
            )
            max_tokens_slider = gr.Slider(16, 256, value=128, step=8, label="Max new tokens")
            temperature_slider = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            answer_box = gr.Textbox(label="Model answer", lines=8)
            status_box = gr.Markdown(label="Status")
            examples_box = gr.Markdown(label="Retrieved examples used as context")

    submit_btn.click(
        fn=generate,
        inputs=[prompt_box, category_dd, max_tokens_slider, temperature_slider, use_retrieval_cb],
        outputs=[answer_box, examples_box, status_box],
    )


if __name__ == "__main__":
    demo.launch()
