# Model Card: Wolof Classroom & Daily-Life Assistant (Qwen3-0.6B + LoRA)

## Model Summary

- Base model: `Qwen/Qwen3-0.6B`
- Adaptation method: LoRA / PEFT, assistant-only loss masking (`-100` on
  system/user/padding tokens)
- Target task: short-form question answering in Wolof across everyday
  categories (education, agriculture, health FAQ, transport, culture),
  plus French/Wolof translation and Wolof orthography normalization.
- Target users: students and community members in Senegal/West Africa who
  want simple Wolof-language answers and Wolof spelling help.
- Target language/domain: Wolof (with French/Wolof code-switching inputs),
  general/education/agriculture/health-FAQ/transport/culture/orthography.
- Hugging Face model repo: `conde621gmail/qwen-wolof-assistant`
- Hugging Face Space: `conde621gmail/wolof-assistant`
  (https://huggingface.co/spaces/conde621gmail/wolof-assistant)

## Intended Use

- Answering short factual/educational questions in Wolof for the categories
  above.
- Translating short sentences between French and Wolof.
- Correcting non-standard Wolof spelling to standard orthography.
- Demonstrating a small, locally fine-tuned, deployable LLM workflow
  (data -> LoRA -> evaluation -> Hub -> Space).

## Out-of-Scope Use

- Medical diagnosis, medication dosage, or any health decision-making.
- Legal advice.
- Long-form generation, multi-turn complex reasoning, or open-domain
  knowledge outside the training categories.
- Safety-critical or high-stakes decisions of any kind. The model is a 0.6B
  parameter model fine-tuned on a few thousand short examples; it will
  hallucinate outside its training distribution.

## Data Methodology

| Source                                             | Type                                                                     | Size      | License/Access                                                                                               | Cleaning Method                                                                                           | Role           |
| -------------------------------------------------- | ------------------------------------------------------------------------ | --------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | -------------- |
| `synth` (`syntetic_wolof_instruct_data.jsonl`) | local synthetic                                                          | 5300 rows | provided as course starter data                                                                              | dedup not needed (generated), validated via `normalized_instruction_row` (non-empty instruction/output) | train/val/eval |
| `aya` (`wolof_aya.jsonl`)                      | local substitute for `CohereLabs/aya_dataset` (Wolof rows)             | 40 rows   | hand-written, same schema/intent as Aya; real Aya download blocked by network policy in the prep environment | manually written, schema-validated                                                                        | train/val/eval |
| `soynade` (`wolof_soynade.jsonl`)              | local substitute for `soynade-research/Wolof-Non-Standard-Orthography` | 30 rows   | hand-written, same schema/intent; real Soynade download blocked by network policy in the prep environment    | manually written, schema-validated                                                                        | train/val/eval |

All three sources were converted to chat format (`system`/`user`/`assistant`)
via `src/download_datasets.py prepare` and split per source with a
deterministic seed (`42`).

> Note: `aya` and `soynade` are intentionally small placeholder sources. For a
> production run, replace them with the real public datasets via
> `python src/download_datasets.py download` once Hugging Face Hub access is
> available, then re-run `prepare`.

## Data Splits

| Split      | Number of examples                      | Ratio | Notes                                        |
| ---------- | --------------------------------------- | ----- | -------------------------------------------- |
| Train      | 4832 (4770 synth + 36 aya + 26 soynade) | ~90%  | per-source, seed 42                          |
| Validation | 269 (265 synth + 2 aya + 2 soynade)     | ~5%   | per-source, seed 42                          |
| Evaluation | 269 (265 synth + 2 aya + 2 soynade)     | ~5%   | combined into `data/splits/eval_all.jsonl` |

## Chat Template and Training Labels

Every example is converted to:

```text
system: <SYSTEM_PROMPT from src/data_utils.py>
user: Context category: <category>\nInstruction: <instruction>   (or "Instruction: <instruction>" if no category)
assistant: <output>
```

Training labels (assistant-only masking, implemented in
`train_lora_assistant_only.py` via `AssistantOnlyChatDataset`):

- `system` tokens: `-100`
- `user` tokens: `-100`
- padding tokens: `-100`
- `assistant` output tokens: real token ids (only these contribute to the
  loss)

## Training Configuration

From `configs/wolof_training_config.yml` (defaults; adjust per your run):

| Parameter             | Value                                                   |
| --------------------- | ------------------------------------------------------- |
| Base model            | Qwen/Qwen3-0.6B                                         |
| LoRA rank             | 16                                                      |
| LoRA alpha            | 32                                                      |
| LoRA dropout          | 0.05                                                    |
| Learning rate         | 1e-4                                                    |
| Epochs                | 3                                                       |
| Batch size            | 1                                                       |
| Gradient accumulation | 8                                                       |
| Max sequence length   | 1024                                                    |
| Warmup steps          | 50                                                      |
| Weight decay          | 0.01                                                    |
| Checkpoint policy     | save every 100 steps, keep best 5, eval every 100 steps |

**Training run results (actual, Google Colab T4 GPU):**

- Total steps: 1812 (3 epochs over 4832 training examples, effective batch
  size 8 via gradient accumulation)
- Final train loss: **0.2768**
- Final eval loss: **0.02885** (eval_loss at epoch 3 on validation split)
- Total training wall-clock time: **~2h 12min** (`train_runtime` = 7966s)
- Hardware: Google Colab GPU runtime (T4-class)
- Best checkpoint: `checkpoint-1812` (final step), exported to
  `outputs/qwen3_0_6b_wolof_lora/best_adapter`

## Evaluation

Run with:

```bash
python evaluation.py --data data/splits/eval_all.jsonl --generate --model-choice qwen --adapter auto
```

on the 269-row combined evaluation set (`data/splits/eval_all.jsonl`).

| Metric      | Value            |
| ----------- | ---------------- |
| Exact Match | **0.3978** |
| Token F1    | **0.7976** |
| BLEU        | **0.7251** |
| ROUGE-L     | **0.7937** |

Evaluated on the full 269-row combined evaluation set
(`data/splits/eval_all.jsonl`), with `--generate --model-choice qwen --adapter auto`.

Add 3 to 5 representative examples after running evaluation/inference:

| Prompt                                                                             | Reference                                                                             | Model Output                                                                          | Comment                                                                                                                                        |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Tekki ci Wolof: 'Cette ville est tres grande.' (language)                          | Dëkk bii dafa réy lool.                                                             | Tous mën na lañu dafa am solo ci sunu baax.                                         | EM=0.00, F1=0.13 — translation quality is weak, consistent with the very small (40-row)`aya` training source.                               |
| Corrige l'orthographe ... "xale yi dinañ dem lekol suba" (orthography)            | Xale yi dinañu dem lekool suba.                                                      | Xale yi ñëw nga yomb ci dem lekool suba.                                            | EM=0.00, F1=0.67 — partially correct normalization, but adds extra words; consistent with the very small (26-row)`soynade` training source. |
| Naka lañuy gërëme Modou ci Wolof? (culture)                                     | Mën nga wax: Jërëjëf Modou, sant naa la bu baax.                                  | Mën nga wax: Jërëjëf Modou, sant naa la bu baax.                                  | EM=1.00, F1=1.00 — exact match on a `culture` category example, the largest training category.                                              |
| Que dire pour encourager un agriculteur comme Modou ... à Rufisque? (agriculture) | Mën nga ne: Yàlla na Yàlla barkeel mbay mi, rawatina mbaxal mi gi nga bay.         | Mën nga ne: Yàlla na Yàlle fekke mi, rawatina mbaxal mi gi nga bay.                | EM=0.00, F1=0.83 — near-correct, minor word substitution ("barkeel" -> "fekke").                                                              |
| Lan la Aminata wara def ngir rëyusiir au daara ci Saint-Louis? (education)        | Aminata dafa wara topp njàngalekat bi, déglu bu baax te jàng lësonam ci daara ji. | Aminata dafa wara topp njàngalekat bi, déglu bu baax te jàng lësonam ci daara ji. | EM=1.00, F1=1.00 — exact match on an `education` example.                                                                                   |

> Per-example results above are taken from the full 269-row run.

## Deployment

- Model Hub URL: https://huggingface.co/conde621gmail/qwen-wolof-assistant
- Space URL: https://huggingface.co/spaces/conde621gmail/wolof-assistant
- Inference framework: Gradio
- Required hardware: CPU is sufficient for Qwen3-0.6B + LoRA inference
  (small batch size, short generations)
- Average latency: TODO (measure on the deployed Space)

The Space (`hf_space/app.py`) integrates the improved
`src/context_state_machine.py`:

- retrieves up to 4 source- and category-aware few-shot examples for the
  prompt,
- shows a lexical-overlap confidence score,
- refuses to generate for questions matching a keyword-based unsafe list
  (medication dosage, self-harm, weapons, etc.) and shows a limitation
  message instead,
- redirects orthography-correction questions to the `soynade` source
  regardless of selected category.

## Limitations

- The `aya` and `soynade` sources are small (40 and 30 rows respectively),
  hand-written local substitutes rather than the full public datasets;
  translation and orthography-correction quality will be limited compared to
  training on the real datasets.
- A 0.6B model with LoRA on ~4800 short examples will struggle with
  out-of-distribution phrasing, long or multi-step questions, and rare
  vocabulary.
- The safety filter in `context_state_machine.py` is keyword-based, not a
  trained classifier; it will miss paraphrased unsafe requests and may
  occasionally over-trigger on benign questions containing a flagged word.
- The lexical-overlap retrieval and confidence score are heuristic; they do
  not capture semantic similarity (e.g. synonyms, paraphrases).

## Safety and Responsible Use

- The deployed Space refuses to answer questions whose text matches a
  keyword list associated with medication dosage, poisoning, self-harm, or
  weapons, and instead shows a fixed limitation message directing the user
  to a qualified professional or emergency services.
- For low-retrieval-confidence questions (lexical overlap with the training
  data below the configured threshold), the app prepends a caveat noting the
  answer may be unreliable.
- Prompt-injection risk: the retrieved few-shot examples come from the
  project's own training data (not external/untrusted user content), so the
  main injection surface is the user's own question; the system prompt
  instructs the model not to emit hidden `<think>` reasoning, and the app
  strips any `<think>...</think>` blocks from the output as a defense in
  depth.

## Authors

- Group: AIMS Senegal — Applied Generative and Agentic AI, Wolof Assistant
- Members: Ansoumane Conde (Model Developer), Rafiatou Okere
  (Data/Evaluation Analyst), Harilova Juliana (Project Lead), Adama Telly Ba
  (Designer / Space UI), Soukeyna Toure (Full-Stack Developer)
- Course: Applied Cas Study, AIMS Senegal
