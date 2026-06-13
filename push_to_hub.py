"""Push the Wolof LoRA adapter to the Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from src.checkpoint_manager import HUB_EXPORT_DIR, export_clean_adapter, select_adapter_source
from src.model_registry import resolve_model_name, resolve_output_dir


SCRIPT_DIR = Path(__file__).resolve().parent
app = typer.Typer(
    help="Push the trained LoRA adapter folder to Hugging Face Hub.",
    pretty_exceptions_show_locals=False,
)


MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- wolof
- senegal
- education
---

# Wolof LoRA Adapter

This repository contains a LoRA adapter fine-tuned for a classroom demo on
Wolof instruction following in Senegalese/African contexts.

## Base Model

- `{base_model}`

## Intended Use

This adapter is designed for teaching:

- instruction fine-tuning,
- LoRA deployment,
- local inference,
- basic evaluation with Exact Match, F1, BLEU, ROUGE-L, and perplexity.

It is not a production Wolof assistant.

## Loading Example

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "{base_model}"
adapter_id = "YOUR_USERNAME/YOUR_REPO"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_id)
```

## Training Setup

- LoRA rank: 16
- LoRA alpha: 32
- Target modules: attention and MLP projection layers
- Dataset schema: `instruction`, `input`, `output`
- Chat template rendered without hidden thinking traces when supported.

## Limitations

The dataset is small and classroom-oriented. The model may repeat short Wolof
phrases or fail outside the covered categories. Evaluate before reuse.
"""


def resolve_relative_to_script(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate


def ensure_model_card(adapter_dir: Path, overwrite: bool, base_model: str) -> None:
    readme = adapter_dir / "README.md"
    if readme.exists() and not overwrite:
        return
    readme.write_text(MODEL_CARD_TEMPLATE.format(base_model=base_model), encoding="utf-8")


def resolve_token(token: Optional[str], token_env: Optional[str]) -> Optional[str]:
    if token:
        return token
    if token_env:
        if token_env.startswith("hf_"):
            raise ValueError(
                "You passed a raw Hugging Face token to --token-env. "
                "Use --token-env with an environment variable name, for example: "
                "export HF_TOKEN_PERSO='hf_...'; python push_to_hub.py --token-env HF_TOKEN_PERSO. "
                "Alternatively pass the raw token with --token, but this is less safe for shell history."
            )
        env_token = os.environ.get(token_env)
        if not env_token:
            raise ValueError(f"Environment variable {token_env} is not set.")
        return env_token
    return None


def repo_namespace(repo_id: str) -> Optional[str]:
    if "/" not in repo_id:
        return None
    return repo_id.split("/", 1)[0]


def extract_identity_names(whoami: Dict[str, Any]) -> List[str]:
    names = []
    user_name = whoami.get("name")
    if user_name:
        names.append(str(user_name))

    for org in whoami.get("orgs", []) or []:
        if isinstance(org, dict) and org.get("name"):
            names.append(str(org["name"]))
    return names


def print_identity(whoami: Dict[str, Any]) -> None:
    names = extract_identity_names(whoami)
    user_name = whoami.get("name", "unknown")
    typer.echo(f"Authenticated Hugging Face user: {user_name}")
    orgs = [name for name in names if name != user_name]
    if orgs:
        typer.echo(f"Visible organizations: {', '.join(orgs)}")


def validate_repo_namespace(repo_id: str, whoami: Dict[str, Any], allow_mismatch: bool) -> None:
    namespace = repo_namespace(repo_id)
    if namespace is None:
        return

    if namespace == "YOUR_USERNAME":
        raise ValueError(
            "Replace YOUR_USERNAME with your real Hugging Face namespace, "
            "for example papasega/qwen3-wolof-lora."
        )

    identity_names = extract_identity_names(whoami)
    if namespace in identity_names:
        return

    message = (
        f"The repo namespace '{namespace}' is not the authenticated user/org. "
        f"Authenticated namespaces visible to this token: {identity_names}. "
        "Use a token from the right account, or pass --allow-namespace-mismatch "
        "if this token really has access to that namespace."
    )
    if allow_mismatch:
        typer.echo(f"Warning: {message}")
        return
    raise ValueError(message)


def ensure_repo_visibility(api, repo_id: str, private: bool) -> None:
    """Set repo visibility, especially when an existing private repo must become public."""
    visibility = "private" if private else "public"

    update_visibility = getattr(api, "update_repo_visibility", None)
    if update_visibility is not None:
        try:
            update_visibility(repo_id=repo_id, private=private, repo_type="model")
        except TypeError:
            update_visibility(repo_id=repo_id, private=private)
        typer.echo(f"Repository visibility set to {visibility}.")
        return

    update_settings = getattr(api, "update_repo_settings", None)
    if update_settings is not None:
        try:
            update_settings(repo_id=repo_id, private=private, repo_type="model")
        except TypeError:
            update_settings(repo_id=repo_id, private=private)
        typer.echo(f"Repository visibility set to {visibility}.")
        return

    if private:
        typer.echo(
            "Your huggingface_hub version cannot update repo visibility. "
            "Continuing because the requested visibility is private."
        )
        return

    raise RuntimeError(
        "Your installed huggingface_hub does not expose a repo visibility update API. "
        "Run `python -m pip install -U huggingface_hub`, then retry. "
        "Alternative: make the repo public manually on Hugging Face, then rerun with "
        "`--skip-set-visibility`."
    )


def upload_data_files(repo_id: str, token: Optional[str], upload_file) -> None:
    data_files = [
        SCRIPT_DIR / "data" / "README.md",
        SCRIPT_DIR / "data" / "wolof_instruction_data.jsonl",
        SCRIPT_DIR / "data" / "273_wol_instruct_data.jsonl",
        SCRIPT_DIR / "data" / "1000_wol_instruct_data.jsonl",
        SCRIPT_DIR / "data" / "wolof_culture_salutations_1000.jsonl",
        SCRIPT_DIR / "data" / "wolof_instruction_sample.jsonl",
        SCRIPT_DIR / "data" / "wolof_eval_examples.jsonl",
    ]

    for data_path in data_files:
        if not data_path.exists():
            typer.echo(f"Skipping missing data file: {data_path.name}")
            continue
        path_in_repo = f"data/{data_path.name}"
        typer.echo(f"Uploading data file: {path_in_repo}")
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(data_path),
            path_in_repo=path_in_repo,
            commit_message=f"Upload {path_in_repo}",
            token=token,
        )


@app.command()
def main(
    repo_id: Optional[str] = typer.Option(
        None,
        help="Hub repository id, for example username/qwen3-wolof-lora.",
    ),
    model_choice: str = typer.Option(
        "qwen",
        "--model-choice",
        "--model-family",
        help="Preset base model: qwen or gemma.",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Optional explicit base model name or path for the model card.",
    ),
    adapter_dir: str = typer.Option(
        "auto",
        help="Local adapter folder to upload. Use auto to derive it from --model-choice.",
    ),
    checkpoint: str = typer.Option(
        "best",
        help=(
            "Adapter selection: best, latest, final, checkpoint-123, "
            "or a custom adapter directory relative to --adapter-dir."
        ),
    ),
    private: bool = typer.Option(
        True,
        "--private/--no-private",
        help="Create or keep the Hub repository private.",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        help="Alias for --no-private. Force the Hub repository to be public.",
    ),
    token: Optional[str] = typer.Option(
        None,
        help="Hugging Face token. If omitted, uses the logged-in token.",
    ),
    token_env: Optional[str] = typer.Option(
        None,
        help="Read the Hugging Face token from this environment variable.",
    ),
    check_auth: bool = typer.Option(
        False,
        "--check-auth",
        help="Only print which Hugging Face account the token uses, then exit.",
    ),
    allow_namespace_mismatch: bool = typer.Option(
        False,
        "--allow-namespace-mismatch",
        help="Skip namespace validation before creating the repo.",
    ),
    skip_create_repo: bool = typer.Option(
        False,
        "--skip-create-repo",
        help="Do not call create_repo; use this when the Hub repo already exists.",
    ),
    skip_set_visibility: bool = typer.Option(
        False,
        "--skip-set-visibility",
        help="Do not call the Hub visibility API; use after making the repo public manually.",
    ),
    commit_message: str = typer.Option(
        "Upload Wolof LoRA adapter",
        help="Hub commit message.",
    ),
    write_model_card: bool = typer.Option(
        True,
        "--write-model-card/--no-write-model-card",
        help="Write the classroom model card before uploading.",
    ),
    include_training_data: bool = typer.Option(
        False,
        "--include-training-data",
        help="Upload the demo training/evaluation data files. Use only for public-safe data.",
    ),
    include_data: bool = typer.Option(
        False,
        "--include-data",
        help="Alias for --include-training-data.",
    ),
) -> None:
    try:
        from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub first: pip install -r requirements.txt"
        ) from exc

    resolved_token = resolve_token(token, token_env)
    api = HfApi(token=resolved_token)
    whoami = api.whoami()
    print_identity(whoami)

    if check_auth:
        return

    if public:
        private = False

    if not repo_id:
        raise ValueError("Missing --repo-id. Example: --repo-id papasega/qwen3-wolof-lora")
    validate_repo_namespace(
        repo_id=repo_id,
        whoami=whoami,
        allow_mismatch=allow_namespace_mismatch,
    )

    resolved_model = resolve_model_name(model_choice, model)
    resolved_adapter_dir = resolve_output_dir(model_choice, adapter_dir)
    adapter_path = resolve_relative_to_script(resolved_adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter folder not found: {adapter_path}")

    source_path, source_label = select_adapter_source(adapter_path, checkpoint)
    if not source_path.exists():
        raise FileNotFoundError(f"Selected adapter source not found: {source_path}")

    upload_path = export_clean_adapter(
        source_dir=source_path,
        export_dir=adapter_path / HUB_EXPORT_DIR,
    )
    if write_model_card:
        ensure_model_card(upload_path, overwrite=True, base_model=resolved_model)

    typer.echo(f"Model choice: {model_choice}")
    typer.echo(f"Base model: {resolved_model}")
    typer.echo(f"Selected adapter source ({source_label}): {source_path}")
    typer.echo(f"Clean upload folder: {upload_path}")
    if skip_create_repo:
        typer.echo("Skipping repo creation because --skip-create-repo was set.")
    else:
        create_repo(repo_id=repo_id, private=private, exist_ok=True, token=resolved_token)

    if skip_set_visibility:
        typer.echo("Skipping visibility update because --skip-set-visibility was set.")
    else:
        ensure_repo_visibility(api=api, repo_id=repo_id, private=private)

    typer.echo("Uploading clean adapter files only.")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(upload_path),
        path_in_repo=".",
        commit_message=commit_message,
        token=resolved_token,
        ignore_patterns=[
            "checkpoint-*",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "training_args.bin",
            "rng_state.pth",
            "training_history.json",
            "training_curves.json",
            "plots/*",
        ],
    )

    if include_training_data or include_data:
        typer.echo("Uploading demo data files because --include-data was set.")
        upload_data_files(
            repo_id=repo_id,
            token=resolved_token,
            upload_file=upload_file,
        )

    typer.echo(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    app()
