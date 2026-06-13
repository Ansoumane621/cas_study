# Data Folder

This project uses three separated data sources, as required by the
methodology in the root `README.md`.

## Sources

| File | Source name | Description | Rows |
|------|-------------|--------------|------|
| `syntetic_wolof_instruct_data.jsonl` | `synth` | Local synthetic classroom-style Wolof instruction/answer pairs covering education, agriculture, sante, transport, culture. Provided as the starter file for this exam. | 5300 |
| `wolof_aya.jsonl` | `aya` | General French/Wolof translation and short-instruction pairs. **Local substitute** for `CohereLabs/aya_dataset` (Wolof rows): the live Hugging Face download is not reachable from the environment used to prepare this repo (`huggingface.co` blocked by network egress rules), so this file was hand-written to follow the same schema (`instruction`/`input`/`output`/`source`/`source_detail`) and the same intent (general multilingual instruction/translation data). | 40 |
| `wolof_soynade.jsonl` | `soynade` | Wolof non-standard-orthography -> standard-orthography correction pairs. **Local substitute** for `soynade-research/Wolof-Non-Standard-Orthography`, for the same network-access reason as above. | 30 |

If you have working access to Hugging Face Hub, you can replace these two
substitute files with the real public datasets by running:

```bash
python src/download_datasets.py download \
  --max-aya-examples 500 \
  --max-soynade-examples 500
```

This only downloads if the target files do not already exist, so delete
`data/wolof_aya.jsonl` and `data/wolof_soynade.jsonl` first if you want to
overwrite them with real Aya / Soynade rows.

## Pipeline Outputs

Running:

```bash
python src/download_datasets.py prepare --validation-ratio 0.05 --eval-ratio 0.05
```

produced (with the files above):

```text
data/chat_aya.json          40 chat-formatted rows
data/chat_soynade.json      30 chat-formatted rows
data/chat_synth.json      5300 chat-formatted rows

data/splits/chat_aya_train.json          36
data/splits/chat_aya_validation.json      2
data/splits/chat_aya_eval.json            2

data/splits/chat_soynade_train.json      26
data/splits/chat_soynade_validation.json  2
data/splits/chat_soynade_eval.json        2

data/splits/chat_synth_train.json      4770
data/splits/chat_synth_validation.json  265
data/splits/chat_synth_eval.json        265

data/splits/eval_all.jsonl              269   # combined eval set, all sources

configs/wolof_training_config.yml
```

Splits are deterministic (seed `42`) and computed per source family, so the
small `aya` and `soynade` sources are not swamped by the much larger `synth`
source in the evaluation set.

Each row of every `*.jsonl` source file follows the schema:

```json
{
  "instruction": "user task or question",
  "input": "category or context",
  "output": "expected assistant answer",
  "source": "source_name",
  "source_detail": "dataset_name_or_generation_method"
}
```

Generated chat files and splits (`data/chat_*.json`, `data/splits/`) are
ignored by Git by default. Re-run `prepare` after cloning to regenerate them.
