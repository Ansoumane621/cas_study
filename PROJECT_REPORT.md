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

**TODO (fill in after the real run):**
- final training/validation loss values and curve description,
- total number of optimizer steps and epochs actually completed,
- wall-clock training time and hardware (e.g. "1x T4 GPU on Colab, ~X
  minutes").

## 5. Evaluation

Run with:

```bash
python evaluation.py --data data/splits/eval_all.jsonl --generate --model-choice qwen --adapter auto
```

**TODO (fill in from the actual run on the 269-row `eval_all.jsonl`):**

| Metric | Value |
|--------|-------|
| Exact Match | TODO |
| Token F1 | TODO |
| BLEU | TODO |
| ROUGE-L | TODO |

Qualitative examples (3-5), comparing prompt / reference / model output, and
short comments on what went right or wrong: TODO.

Failure cases to look for and document: repetition loops, answering in
French/English instead of Wolof, copying the in-domain few-shot example
verbatim instead of answering the actual question, and degraded quality on
the `aya`/`soynade` categories given their very small training size (36/26
examples).

## 6. Deployment

- Hugging Face model repo link: TODO (after `push_to_hub.py --repo-id
  YOUR_USERNAME/YOUR_MODEL_NAME --model-choice qwen --adapter-dir auto
  --checkpoint best --public`)
- Hugging Face Space link: TODO (after deploying `hf_space/app.py` with
  `MODEL_REPO_ID` set to the repo above)
- Screenshots / usage examples: TODO (add after deployment)
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

### Student 1: Name

- Contribution:
- Technical choice and rejected alternative:
- Problem or failure diagnosed:
- Verification evidence:
- Next improvement:

### Student 2: Name

- Contribution:
- Technical choice and rejected alternative:
- Problem or failure diagnosed:
- Verification evidence:
- Next improvement:

### Student 3: Name

- Contribution:
- Technical choice and rejected alternative:
- Problem or failure diagnosed:
- Verification evidence:
- Next improvement:
