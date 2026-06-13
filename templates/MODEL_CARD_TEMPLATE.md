# Model Card: <YOUR_MODEL_NAME>

## Model Summary

- Base model: `Qwen/Qwen3-0.6B`
- Adaptation method: LoRA / PEFT
- Target task:
- Target users:
- Target language/domain:
- Hugging Face model repo:
- Hugging Face Space:

## Intended Use

Describe what the model is designed to do.

## Out-of-Scope Use

Describe tasks the model should not be used for.

## Data Methodology

List each data source separately.

| Source | Type | Size | License/Access | Cleaning Method | Role |
|--------|------|------|----------------|-----------------|------|
| Source 1 | public / synthetic / local | | | | train/val/eval |
| Source 2 | public / synthetic / local | | | | train/val/eval |

## Data Splits

| Split | Number of examples | Ratio | Notes |
|-------|--------------------|-------|-------|
| Train | | | |
| Validation | | | |
| Evaluation | | | |

## Chat Template and Training Labels

Explain how examples are converted to messages:

```text
system: ...
user: ...
assistant: ...
```

Explain the training labels:

- `system` tokens: `-100`
- `user` tokens: `-100`
- padding tokens: `-100`
- `assistant` output tokens: learned by the model

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | |
| LoRA rank | |
| LoRA alpha | |
| Learning rate | |
| Epochs | |
| Batch size | |
| Gradient accumulation | |
| Max sequence length | |
| Checkpoint policy | |

## Evaluation

Report both automatic metrics and qualitative observations.

| Metric | Value |
|--------|-------|
| Exact Match | |
| Token F1 | |
| BLEU | |
| ROUGE-L | |

Add 3 to 5 representative examples:

| Prompt | Reference | Model Output | Comment |
|--------|-----------|--------------|---------|
| | | | |

## Deployment

- Model Hub URL:
- Space URL:
- Inference framework: Gradio / Streamlit
- Required hardware:
- Average latency:

## Limitations

Describe known failure modes, weak categories, hallucination risks, repetition risks, and language/domain limitations.

## Safety and Responsible Use

Describe guardrails, refusal behavior, prompt injection risks, and how the deployed app handles unsafe or out-of-scope requests.

## Authors

- Group:
- Members:
- Course: Applied Generative and Agentic AI, AIMS Senegal
