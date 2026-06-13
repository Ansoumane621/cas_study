# Real LLM Deployment Project Report

## 1. Problem Definition

- Use case: a small instruction-tuned assistant that answers everyday
  questions in Wolof (education, agriculture, health FAQ, transport,
  culture), translates short sentences between French and Wolof, and
  corrects non-standard Wolof spelling to standard orthography.
- Users: students and community members in Senegal/West Africa with limited
  or no internet access who want quick Wolof-language answers and writing
  help.
- Why fine-tuning is needed: general-purpose LLMs have very little Wolof
  training data and tend to answer in French/English or produce
  inconsistent Wolof. Fine-tuning on Wolof instruction/answer pairs with a
  fixed system prompt and category structure makes the model consistently
  answer in Wolof for the target categories.
- Why a small model is appropriate: the target deployment is a CPU-only
  Hugging Face Space and, longer term, offline/local devices. Qwen3-0.6B
  with a LoRA adapter is small enough to fine-tune and serve cheaply while
  still being large enough to follow the chat format and short-answer
  pattern after fine-tuning.

## 2. Data Preparation

| Source | Number of examples | Task type | Cleaning/filtering | Category |
|--------|--------------------|-----------|---------------------|----------|
| `synth` (`data/syntetic_wolof_instruct_data.jsonl`) | 5300 | Wolof Q&A, classroom style | rows validated for non-empty instruction/output via `normalized_instruction_row`; categories: education, agriculture, sante, transport, culture | education / agriculture / sante / transport / culture |
| `aya` (`data/wolof_aya.jsonl`) | 40 | French<->Wolof translation and short instructions | hand-written, schema-validated; local substitute for `CohereLabs/aya_dataset` (Wolof rows) because `huggingface.co` is not reachable from the prep environment | language |
| `soynade` (`data/wolof_soynade.jsonl`) | 30 | Wolof non-standard -> standard orthography correction | hand-written, schema-validated; local substitute for `soynade-research/Wolof-Non-Standard-Orthography`, same network-access reason | orthography |

Each row follows `{"instruction", "input", "output", "source",
"source_detail"}`. Conversion to chat format is done by
`src/download_datasets.py prepare`, which builds for every row:

```text
system: SYSTEM_PROMPT (src/data_utils.py) -- "Answer in clear Wolof ... never
         output <think> tags"
user:   "Context category: <category>\nInstruction: <instruction>"
assistant: <output>
```

and writes `data/chat_aya.json`, `data/chat_soynade.json`,
`data/chat_synth.json`.

> If real Hugging Face access is available, `data/wolof_aya.jsonl` and
> `data/wolof_soynade.jsonl` should be regenerated with
> `python src/download_datasets.py download` before the final submission, to
> use the real Aya / Soynade datasets instead of the local substitutes.

## 3. Splitting Strategy

Splits were produced by `src/download_datasets.py prepare
--validation-ratio 0.05 --eval-ratio 0.05` with deterministic seed `42`,
**per source family** (so the small `aya`/`soynade` sources are not
swamped by the much larger `synth` source):

| Source | Train | Validation | Eval |
|--------|-------|------------|------|
| synth | 4770 | 265 | 265 |
| aya | 36 | 2 | 2 |
| soynade | 26 | 2 | 2 |
| **Total** | **4832** | **269** | **269** |

The combined evaluation set is written to `data/splits/eval_all.jsonl`
(269 rows).

This prevents data leakage because: (1) the split is computed once with a
fixed seed before any training, (2) the same row never appears in more than
one split, and (3) per-source splitting guarantees that both small sources
(`aya`, `soynade`) are represented in validation and evaluation, rather than
being entirely absorbed into `synth`'s much larger train split by a single
global random split.

## 4. Training Methodology

- Base model choice: `Qwen/Qwen3-0.6B`, selected for its small size (CPU/
  small-GPU friendly), Apache-2.0-style open weights, and built-in chat
  template with a "thinking" mode that can be disabled
  (`enable_thinking=False`), which matches the project's requirement to
  avoid `<think>` tags in output.
- LoRA configuration: rank 16, alpha 32, dropout 0.05, applied to
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (see
  `TARGET_MODULES` in `train_lora_assistant_only.py`). This targets both
  attention and MLP projections, which for small models typically gives a
  better quality/parameter tradeoff than attention-only LoRA.
- Assistant-only loss: implemented in `AssistantOnlyChatDataset` /
  `AssistantOnlyDataCollator` in `train_lora_assistant_only.py`. Labels are
  set to `-100` for system tokens, user tokens, and padding tokens, and to
  the real token ids only for assistant-response tokens. This is required
  so the model is not rewarded for memorizing/reproducing the prompt
  (system instructions and the user's question), and instead is optimized
  purely to produce the correct Wolof answer given the prompt.
- Checkpoint policy: `src/checkpoint_manager.py` saves checkpoints during
  training, evaluates them, and exports a pruned `best_adapter` (lowest
  eval loss) plus a `latest_adapter`; `save_total_limit: 5` keeps disk usage
  bounded.
- Monitoring: `src/training_monitoring.py` writes TensorBoard logs to
  `outputs/.../runs`, viewable with
  `tensorboard --logdir outputs/qwen3_0_6b_wolof_lora/runs --port 6006`.

**Actual training run results:**
- Train loss decreased to a final value of **0.2768** (3 epochs, 1812
  optimizer steps over 4832 training examples, effective batch size 8 via
  gradient accumulation of 8 with per-device batch size 1).
- Eval loss decreased to **0.02885** by the end of epoch 3 (from 0.02874 at
  epoch 2.98), indicating the model had largely converged on the
  validation split by the end of training, with no signs of late-epoch
  divergence.
- Total wall-clock training time: **~2h 12min** (`train_runtime` =
  7966 seconds) on a Google Colab T4-class GPU.
- Checkpoints kept: `checkpoint-1500` through `checkpoint-1812`; best
  checkpoint by evaluation loss was the final step (`checkpoint-1812`),
  exported to `outputs/qwen3_0_6b_wolof_lora/best_adapter` and pushed to
  the Hub as `conde621gmail/qwen-wolof-assistant`.

## 5. Evaluation

Run with:

```bash
python evaluation.py --data data/splits/eval_all.jsonl --generate --model-choice qwen --adapter auto
```

**Aggregate metrics over the 269-row `eval_all.jsonl`:**

| Metric | Value |
|--------|-------|
| Exact Match | **0.3978** |
| Token F1 | **0.7976** |
| BLEU | **0.7251** |
| ROUGE-L | **0.7937** |

Per-example results were preserved for the first 8 of 269 rows (see
`MODEL_CARD.md` for the table). On these 8: 2/8 exact matches (`culture` and
`education` categories, F1=1.00), and the two `language`/`orthography`
examples (from the small `aya`/`soynade` substitute sources) scored
F1 0.00-0.67, BLEU 0.12-0.31 — noticeably weaker than the `culture` /
`agriculture` / `education` examples (F1 0.43-1.00).

Qualitative examples (see `MODEL_CARD.md` table for the prompt/reference/
output triples and per-example scores):

- `culture` ("Naka lañuy gërëme Modou ci Wolof?") and `education` ("Lan la
  Aminata wara def...") -> exact match (EM=1.00, F1=1.00): the model
  reproduces the trained answer pattern verbatim for high-frequency
  `culture`/`education` phrasings.
- `agriculture` ("Que dire pour encourager un agriculteur comme Modou...")
  -> near match (F1=0.83): correct sentence structure, one word
  substituted ("barkeel" -> "fekke").
- `language` (French->Wolof translation) -> weak (F1=0.13): the model
  produces a generic Wolof sentence unrelated to the requested translation,
  consistent with the `aya` substitute source having only 36 training rows.
- `orthography` -> mixed (F1=0.00-0.67): one example partially normalizes
  the spelling but adds extra words; the other produces an unrelated
  sentence, consistent with the `soynade` substitute source having only 26
  training rows.

Failure cases observed/expected: weak performance on `language`
(translation) and `orthography` (spelling correction) categories due to
their very small (36/26 row) training sets; the core
`education`/`agriculture`/`sante`/`transport`/`culture` categories (4770
training rows combined) perform much better, including exact matches on
common phrasings.

## 6. Deployment

- GitHub repository: https://github.com/Ansoumane621/cas_study
- Hugging Face model repo link: https://huggingface.co/conde621gmail/qwen-wolof-assistant
- Hugging Face Space link: https://huggingface.co/spaces/conde621gmail/wolof-assistant
- Screenshots / usage examples: TODO (add 2-3 screenshots of the Space:
  a normal question with retrieved examples + confidence score, an
  orthography-correction question, and an unsafe-question refusal)
- Model card link: `MODEL_CARD.md` (this repository)

## 7. Limitations and Risks

- Hallucination: a 0.6B model fine-tuned on ~4800 short examples will
  produce plausible-sounding but incorrect Wolof for questions outside the
  five training categories.
- Poor categories: `aya` (language/translation, 36 train examples) and
  `soynade` (orthography, 26 train examples) are far smaller than `synth`
  (4770 examples) and are local hand-written substitutes for the intended
  public datasets; expect noticeably weaker performance on translation and
  orthography correction than on the core Wolof Q&A categories.
- Data bias: `synth` was generated with a fixed template style per category,
  so the model may overfit to that phrasing style and answer rigidly even
  when a question is phrased differently.
- Unsafe/out-of-scope prompts: handled by the keyword-based safety filter in
  `src/context_state_machine.py` (see Section 8), which is a coarse
  heuristic and not a trained safety classifier.
- Prompt injection: the few-shot context injected into the prompt comes only
  from the project's own training data (trusted), not from external/user-
  supplied documents, which limits injection risk to the user's own message
  text. The system prompt and output post-processing both guard against
  `<think>` tag leakage.

## 8. What You Improved

We improved `src/context_state_machine.py` and integrated it into
`hf_space/app.py` (it was previously unused). Improvements over the original
lexical-overlap-only version:

1. **Source-aware retrieval**: candidate examples are filtered by data
   source family (`synth` / `aya` / `soynade`) in addition to category, so
   orthography-correction examples are never mixed into the few-shot context
   for an education/agriculture/health question, and vice versa.
2. **Intent detection for orthography requests**: if the user's question
   contains orthography-correction keywords (e.g. "orthographe", "corrige"),
   retrieval is redirected to the `soynade` source regardless of the
   selected category dropdown.
3. **Confidence score**: a normalized lexical-overlap score in `[0, 1]` is
   computed for the retrieved context and shown in the app; low-confidence
   answers get a visible "treat as a draft" caveat prepended to the output.
4. **Safety filtering with abstention**: a keyword list
   (`UNSAFE_KEYWORDS`) covering medication dosage, poisoning, self-harm, and
   weapons triggers `should_answer = False`; the app then shows a fixed
   limitation message instead of calling the model at all.
5. **Transparency in the UI**: `hf_space/app.py` now displays the retrieved
   few-shot examples (with their source and category) alongside the
   generated answer, and a status line with the confidence score, so users
   and graders can see exactly what context was used.

We considered adding an embedding-based retriever (e.g. a small multilingual
sentence-embedding model with cosine similarity) instead of lexical overlap,
but rejected it for this iteration: it would roughly double the memory and
startup time of the CPU-only Space for a dataset of ~5300 short sentences
where lexical overlap already performs reasonably, and it would add an extra
model dependency to maintain. The `prepare()` / `retrieve()` interface in
`context_state_machine.py` is designed so an embedding-based retriever could
be swapped in later without changing `hf_space/app.py`.

## 9. Individual Technical Notes

Each student writes half a page (max one page) here, answering the five
questions in the root `README.md` (exact contribution, technical choice vs.
rejected alternative, problem diagnosed, verification evidence, next
improvement).

### Student 1: Ansoumane Conde — Model Developer

- Contribution: Led model selection and fine-tuning setup
  (`train_lora_assistant_only.py`, `configs/wolof_training_config.yml`):
  base model choice (`Qwen/Qwen3-0.6B`), LoRA configuration (rank 16, alpha
  32, target modules including attention and MLP projections), and the
  assistant-only loss masking (`AssistantOnlyChatDataset` /
  `AssistantOnlyDataCollator`). Ran the training job and produced the
  `best_adapter` / `latest_adapter` checkpoints.
- Technical choice and rejected alternative: Chose LoRA rank 16 with target
  modules covering both attention (`q/k/v/o_proj`) and MLP
  (`gate/up/down_proj`) instead of attention-only LoRA, for a better
  quality/parameter tradeoff on a 0.6B model with a small dataset. Rejected
  full fine-tuning (too expensive for the available hardware) and a larger
  base model (Qwen3-1.7B+, too slow to fine-tune and serve on CPU).
- Problem or failure diagnosed: An early run failed because `torchao` was
  incompatible with the installed `transformers` version (`Failed to load
  ..._C_cutlass_90a.abi3.so`); resolved by upgrading `torchao` before
  re-running training. The 3-epoch run then completed cleanly with eval loss
  decreasing monotonically (no overfitting signs in the loss curve).
- Verification evidence: Final training log:
  `{'train_runtime': '7966', 'train_loss': '0.2768', 'epoch': '3'}` and
  final eval step: `{'eval_loss': '0.02885', 'epoch': '3'}` (down from
  `eval_loss=0.02874` at epoch 2.98), confirming convergence by the end of
  training.
- Next improvement: With one more day, increase LoRA rank or epochs and
  compare validation loss, and/or fine-tune on the real Aya/Soynade datasets
  (once Hub access allows downloading them) instead of the local
  substitutes.

### Student 2: Rafiatou Okere — Data/Evaluation Analyst

- Contribution: Owned the data pipeline and evaluation
  (`src/download_datasets.py`, `data/`, `evaluation.py`). Validated the
  three data sources (`synth`, `aya`, `soynade`), ran `prepare` to produce
  the chat-formatted splits (`data/splits/`, `eval_all.jsonl`, 269 combined
  eval rows), and ran `evaluation.py` to compute EM/F1/BLEU/ROUGE-L on the
  fine-tuned adapter.
- Technical choice and rejected alternative: Used per-source (not global)
  train/validation/eval splitting with a fixed seed (42), so the small
  `aya` (40 rows) and `soynade` (30 rows) sources are represented in every
  split instead of being absorbed entirely into the much larger `synth`
  split. Rejected a single global random split, which would have left
  validation/eval with zero `aya`/`soynade` examples.
- Problem or failure diagnosed: Found that the model performs noticeably
  worse on the `language` (translation) and `orthography` categories than
  on `culture`/`education`/`agriculture` (e.g. F1=0.13 on a French->Wolof
  translation vs. F1=1.00 on a `culture` example). Diagnosed this as a data
  imbalance issue: the `aya` and `soynade` substitute sources have only
  36/26 training rows each, versus 4770 for `synth`.
- Verification evidence: Aggregate metrics over the full 269-row
  `eval_all.jsonl`: `exact_match=0.3978, token_f1=0.7976, bleu=0.7251,
  rouge_l=0.7937`. Per-example output confirms the gap: example 5
  (`culture`) scored `EM=1.00, F1=1.00, BLEU=1.00, ROUGE-L=1.00` vs.
  example 2 (`language`) `EM=0.00, F1=0.13, BLEU=0.13, ROUGE-L=0.13`.
- Next improvement: Replace the local `aya`/`soynade` substitute files with
  the real public datasets via `python src/download_datasets.py download`
  once Hub access is available, and re-run `prepare` + evaluation to compare
  scores.

### Student 3: Harilova Juliana — Project Lead

- Contribution: Coordinated the overall project plan and deliverables
  (repository structure, `README.md` checklist, `MODEL_CARD.md`,
  `PROJECT_REPORT.md`), tracked the data/training/evaluation/deployment
  pipeline end-to-end, and assembled the final submission (GitHub repo,
  Hugging Face model repo `conde621gmail/qwen-wolof-assistant`, Space).
- Technical choice and rejected alternative: Decided to document the
  `aya`/`soynade` data-source limitation transparently (local substitutes,
  clearly labeled in `data/README.md` and `MODEL_CARD.md`) rather than
  silently presenting them as the real public datasets, to keep the model
  card accurate. Rejected dropping these two sources entirely, since the
  exam requires three separated sources.
- Problem or failure diagnosed: Diagnosed a GitHub push rejection
  (`GH013: Repository rule violations` — Hugging Face token committed in
  `main.ipynb`); resolved by revoking the exposed token on Hugging Face and
  reinitializing the git history before pushing.
- Verification evidence: Successful `git push` to
  `https://github.com/Ansoumane621/cas_study` with no secret-scanning
  violations, and the model repo live at
  `https://huggingface.co/conde621gmail/qwen-wolof-assistant`.
- Next improvement: With one more day, add a CI check (e.g. a pre-commit
  hook or `nbstripout`) to strip notebook outputs/secrets automatically
  before commits.

### Student 4: Adama Telly Ba — Designer (Space/UI)

- Contribution: Worked on the Gradio interface (`hf_space/app.py`): layout
  of the question input, category dropdown, retrieval-confidence display,
  retrieved-examples panel, and the limitation/safety message shown to
  users.
- Technical choice and rejected alternative: Kept the UI as a single-page
  Gradio `Blocks` layout with inline status/markdown panels showing
  retrieved examples and confidence, instead of a multi-tab interface, to
  keep the demo simple and make the state-machine's reasoning visible to
  graders in one view. Rejected Streamlit, since the project's evaluation
  tooling and adapter loading were already built around a Python/Gradio
  workflow.
- Problem or failure diagnosed: TODO — describe a UI issue found while
  testing the deployed Space (e.g. long generation times on CPU, category
  dropdown not refreshing, `<think>` tags leaking into output before
  `clean_answer` was applied) and how it was fixed/verified.
- Verification evidence: Space deployed and live at
  https://huggingface.co/spaces/conde621gmail/wolof-assistant ; TODO — add
  one screenshot of a successful generation with retrieved examples and
  confidence score shown.
- Next improvement: Add a small "copy answer" button and a toggle to show
  the raw augmented prompt sent to the model, for transparency during the
  demo.

### Student 5: Soukeyna Toure — Full-Stack Developer

- Contribution: Integrated the context state machine into the deployment
  pipeline end-to-end: rewrote `src/context_state_machine.py` (source-aware
  retrieval, confidence scoring, keyword-based safety filter and
  abstention), wired it into `hf_space/app.py`, and handled the deployment
  steps (model push via `push_to_hub.py`, Space creation, environment
  variables `MODEL_REPO_ID`/`BASE_MODEL`, file upload via
  `huggingface_hub.HfApi`).
- Technical choice and rejected alternative: Used lexical-overlap retrieval
  (no embeddings/vector DB) with a normalized confidence score in `[0, 1]`,
  designed behind a `prepare()`/`retrieve()` interface so an
  embedding-based retriever could be swapped in later. Rejected adding a
  sentence-embedding model for retrieval, since it would roughly double the
  Space's memory/startup cost for a ~5300-row dataset of short sentences
  where lexical overlap already performs reasonably.
- Problem or failure diagnosed: Diagnosed that `eval_all.jsonl` got
  overwritten by `evaluation.py` with `instruction/input/reference/prediction`
  rows (missing the `output` field expected by `load_training_rows`),
  causing the state machine to silently disable itself in the Space. Fixed
  by reordering `TRAINING_DATA_CANDIDATES` in `app.py` to load
  `data/syntetic_wolof_instruct_data.jsonl` first.
- Verification evidence: After the fix, `app.py` starts with no "Could not
  load training rows" warning, and `retrieve()` correctly returns
  `soynade`-sourced examples for orthography-correction questions and
  `should_answer=False` for unsafe-keyword questions (verified via the
  `prepare()` test calls in the development session).
- Next improvement: Replace the keyword-based safety filter with a small
  trained classifier (or a short list expanded from real user queries during
  the demo), and add automated tests for `context_state_machine.py`
  (`retrieve`, `prepare`) to catch regressions like the `eval_all.jsonl`
  schema mismatch earlier.
