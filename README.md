# Real LLM Deployment and Standard Industrial Methodologies

**Course:** Cases studies Applied Generative and Agentic AI
**Assessment type:** Project-based exam
**Theme:** From fine-tuning to real deployment
**Recommended base model:** `Qwen/Qwen3-0.6B or litert-community/gemma-4-E2B-it-litert-lm.`

This project is a continuation of the Session 7 fine-tuning lab. You will work
in the same groups as for the case studies, as if you were working on a small
industry-style AI project. The goal is not only to fine-tune a model, but to
build a clean, reproducible, industry-style LLM deployment workflow.

You must prepare data, train a LoRA adapter, evaluate the model, push the model
to Hugging Face Hub, and deploy a working Hugging Face Space.

## Final Deliverables

Each group must submit:

1. A GitHub repository containing the complete project.
2. A Hugging Face model repository containing the clean LoRA adapter.
3. A Hugging Face Space using Gradio or Streamlit.
4. A completed model card.
5. A short project report.
6. An individual technical note for each group member.
7. A short demo during the final presentation.

## Project Objective

Build a small instruction-tuned LLM for a real use case under limited compute.

Examples:

- Wolof educational assistant;
- health FAQ assistant with strict limitations;
- agriculture advisory assistant;
- public service assistant;
- local language chatbot;
- domain-specific assistant for AIMS coursework.

The provided Wolof data is a starter example. Your group may keep Wolof, choose
another African language, work in English or French, or build another
domain-specific assistant. However, the methodology is mandatory: separated
data sources, chat formatting, assistant-only training, evaluation, deployment,
and documentation.

## Mandatory Methodology

### 1. Data Sources Must Be Separated

Do not directly train on one mixed dataset without documenting the source.

You must use at least three separated data sources:

1. a general or base dataset;
2. a synthetic instruction dataset;
3. a domain-specific dataset linked to your use case.

For a Wolof project, this could look like:

```text
data/wolof_aya.jsonl
data/wolof_soynade.jsonl
data/wolof_synth.jsonl
```

For another language or domain, use explicit names such as:

```text
data/source_1_general.jsonl
data/source_2_synthetic.jsonl
data/source_3_domain.jsonl
```

Each row should use this schema:

```json
{
  "instruction": "user task or question",
  "input": "category or context",
  "output": "expected assistant answer",
  "source": "source_name",
  "source_detail": "dataset_name_or_generation_method"
}
```

### 2. Chat Formatting Must Be Explicit

You must convert every example into:

```text
system
user
assistant
```

For the Wolof starter example, the pipeline creates:

```text
data/chat_aya.json
data/chat_soynade.json
data/chat_synth.json
```

If you choose another language or domain, rename the chat files clearly and
update the YAML config accordingly.

### 3. Splits Must Be Reproducible

Use a deterministic seed and document your split sizes.

Recommended split:

```text
train      90%
validation 5%
eval       5%
```

The split must be done per source family to avoid one source dominating the
evaluation set.

### 4. Assistant-Only Training Is Required

Your training must use assistant-only loss masking:

```text
system tokens    -> label = -100
user tokens      -> label = -100
padding tokens   -> label = -100
assistant tokens -> normal token ids
```

This prevents the model from learning to copy the prompt and forces it to learn
the expected answer.

The required training script is:

```text
train_lora_assistant_only.py
```

### 5. Evaluation Is Required

You must evaluate on a held-out evaluation split and report:

- Exact Match;
- token F1;
- BLEU;
- ROUGE-L;
- qualitative examples;
- limitations and failure cases.

### 6. Deployment Is Required

You must deploy:

- the LoRA adapter on Hugging Face Hub;
- a Gradio or Streamlit app on Hugging Face Spaces.

The Space must use your uploaded model repository, not a local checkpoint.

## Starter Files

```text
src/download_datasets.py          # data download, normalization, chat conversion, splits, YAML config
train_lora_assistant_only.py      # LoRA training with -100 assistant-only labels
evaluation.py                     # automatic metrics
inference_lora_qwen3_wolof.py     # local text/chat inference
push_to_hub.py                    # clean LoRA adapter upload
src/context_state_machine.py      # optional retrieval/state machine to improve
templates/MODEL_CARD_TEMPLATE.md  # model card template
templates/PROJECT_REPORT_TEMPLATE.md
hf_space/app.py                   # Gradio Space template
```

## Required Improvement: Context State Machine

The file `src/context_state_machine.py` is intentionally kept in this project.
You must improve it or replace it with a better component.

Possible improvements:

- add category-aware retrieval;
- add source-aware retrieval;
- prevent retrieval of irrelevant training examples;
- add simple safety filtering;
- add a confidence score;
- show retrieved examples in the deployed app;
- explain when the state machine should not answer.

If you do not use it in the final Space, you must explain why and propose a
better alternative.

## Individual Differentiation Note

Each student must add a short individual note to the project report.

Recommended length: **half a page per student**. Maximum length: **one page per
student**.

The goal is to distinguish individual understanding inside a group project. Keep
answers precise and technical. Do not write long paragraphs.

Each student must answer:

1. What exact part of the project did you implement or improve? Mention files,
   functions, notebooks, commits or experiments when possible.
2. Which technical choice did you make, and what alternative did you reject?
   Example: model size, LoRA rank, data source, evaluation metric,
   quantization, Space interface, guardrail.
3. What problem or failure did you observe, and how did you diagnose it?
   Example: overfitting, bad generation, memory issue, checkpoint failure,
   unsafe output, deployment error.
4. How did you verify that your contribution works? Give one metric, one test,
   one screenshot, one command output, or one qualitative example.
5. If you had one extra day, what would you improve first and why?

During the final presentation, the instructor may ask one individual question
based on this note. The group grade can be adjusted individually when a student
cannot explain the part they claim to have contributed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your environment already exists:

```bash
pip install -r requirements.txt
```

## Step 1: Prepare the Data

The provided synthetic starter file is:

```text
data/syntetic_wolof_instruct_data.jsonl
```

Prepare separated data files, chat files, splits, and the YAML config:

```bash
python src/download_datasets.py prepare \
  --validation-ratio 0.05 \
  --eval-ratio 0.05
```

If you want to download public datasets before preparing:

```bash
python src/download_datasets.py all \
  --validation-ratio 0.05 \
  --eval-ratio 0.05
```

For a quick test with limited data:

```bash
python src/download_datasets.py all \
  --max-aya-examples 500 \
  --max-soynade-examples 500 \
  --validation-ratio 0.05 \
  --eval-ratio 0.05
```

This should create:

```text
configs/wolof_training_config.yml
data/splits/eval_all.jsonl
```

You may adapt the config for your own use case.

## Step 2: Train the Model

Quick test:

```bash
python train_lora_assistant_only.py \
  --config configs/wolof_training_config.yml \
  --model-choice qwen \
  --max-train-examples 100 \
  --epochs 1 \
  --no-checkpoint-eval \
  --no-use-config-hparams
```

Full Qwen training:

```bash
python train_lora_assistant_only.py \
  --config configs/wolof_training_config.yml \
  --model-choice qwen
```

The script saves:

```text
outputs/qwen3_0_6b_wolof_lora/
outputs/qwen3_0_6b_wolof_lora/best_adapter/
outputs/qwen3_0_6b_wolof_lora/latest_adapter/
```

## Step 3: Monitor Training

TensorBoard:

```bash
tensorboard --logdir outputs/qwen3_0_6b_wolof_lora/runs --port 6006
```

Open:

```text
http://127.0.0.1:6006
```

## Step 4: Evaluate

```bash
python evaluation.py \
  --data data/splits/eval_all.jsonl \
  --generate \
  --model-choice qwen \
  --adapter auto
```

You must include the evaluation results in the report and model card.

## Step 5: Test Locally

```bash
python inference_lora_qwen3_wolof.py \
  --model-choice qwen \
  --adapter auto \
  --category education \
  --instruction "Write a test prompt for your use case."
```

Interactive mode:

```bash
python inference_lora_qwen3_wolof.py \
  --model-choice qwen \
  --adapter auto \
  --category education \
  --chat
```

## Step 6: Push to Hugging Face Hub

Login:

```bash
huggingface-cli login
```

Check account:

```bash
python push_to_hub.py --check-auth
```

Push a clean adapter:

```bash
python push_to_hub.py \
  --repo-id YOUR_USERNAME/YOUR_MODEL_NAME \
  --model-choice qwen \
  --adapter-dir auto \
  --checkpoint best \
  --public
```

Your model card must be filled before the final submission. Use:

```text
templates/MODEL_CARD_TEMPLATE.md
```

## Step 7: Deploy on Hugging Face Spaces

Use the starter Space:

```text
hf_space/app.py
hf_space/requirements.txt
```

You must replace:

```text
YOUR_USERNAME/YOUR_LORA_MODEL_REPO
```

with your real model repository.

Your deployed Space must:

- load the model from Hugging Face Hub;
- provide a clean user interface;
- show meaningful outputs for your use case;
- include at least one guardrail or limitation message.

## Expected Repository Structure

Your final repository should look like:

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── wolof_training_config.yml
├── data/
│   ├── README.md
│   └── syntetic_wolof_instruct_data.jsonl
├── src/
│   ├── download_datasets.py
│   ├── data_utils.py
│   ├── checkpoint_manager.py
│   ├── context_state_machine.py
│   ├── model_registry.py
│   └── training_monitoring.py
├── train_lora_assistant_only.py
├── evaluation.py
├── inference_lora_qwen3_wolof.py
├── push_to_hub.py
├── hf_space/
│   ├── app.py
│   └── requirements.txt
└── templates/
    ├── MODEL_CARD_TEMPLATE.md
    └── PROJECT_REPORT_TEMPLATE.md
```

Do not commit:

- checkpoints;
- optimizer files;
- TensorBoard logs;
- local virtual environments;
- large generated outputs.

## Grading Rubric

| Criterion                                                            | Weight |
| -------------------------------------------------------------------- | ------ |
| Clear use case and problem definition                                | 10%    |
| Data methodology: separated sources, cleaning, documentation         | 15%    |
| Correct chat formatting and assistant-only `-100` masking          | 10%    |
| Training quality and reproducibility                                 | 15%    |
| Evaluation quality and interpretation                                | 15%    |
| Hugging Face Hub deployment and model card                           | 10%    |
| Hugging Face Space deployment                                        | 10%    |
| Improvement of `context_state_machine.py` or justified alternative | 5%     |
| Individual technical note and oral defense                           | 10%    |

## Submission Checklist

Before submission, verify:

- `configs/wolof_training_config.yml` exists and matches your data.
- `data/splits/eval_all.jsonl` exists.
- `outputs/.../best_adapter` exists locally.
- Your Hugging Face model repo is public or accessible to the instructor.
- Your Hugging Face Space runs and uses the Hub model.
- Your model card is complete.
- Your report explains data, training, evaluation, deployment, and limitations.
- Each student has written an individual technical note.
