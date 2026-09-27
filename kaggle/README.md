# Kaggle port

Training runs **on Kaggle**; this Windows checkout is the place we test. The port
lives entirely in this directory — `src/`, `ns.py` and the local commands are
untouched and stay Windows-native.

## Why a port is needed at all

Everything this repo writes on Windows stores OS-native separators:

| file | Windows value | why it breaks on Kaggle |
| --- | --- | --- |
| `data/processed/manifest*.csv` | `data\fleurs_ceb_ph\train\00000.wav` | on Linux `\` is a legal filename char, so the decode fails |
| `data/processed/split_*.json` | same, in `audio_path` | same |
| `checkpoints/<line>/latest.json`, `best.json` | `checkpoints\mms\v004` | `-resume latest` / `-continue best` resolve to one bogus filename |

`kaggle/make_upload.py` therefore rewrites **only those path fields** in copies.
Ids, text, metrics and the manifest fingerprint are copied byte-for-byte, so no
split is silently rebuilt with different rows.

The per-split `manifest.tsv` files need no rewrite: their `filename` column is a
bare `00000.wav` that the registry joins to the split directory at read time.

**Where the run stands:** `checkpoints/mms` holds only `v004` — a 26-token
word-corpus model. No FLEURS epoch has ever been written, so the first Kaggle
run must use `-continue`, not `--resume` (see 3c).

## 1. Build the bundles (on Windows)

Bundle the newest version that actually exists on disk (each is ~1.55 GB; never
upload the whole line). Right now that is **v004** — the word-corpus model:

```powershell
python kaggle/make_upload.py --version v004
```

If a local or previous Kaggle run wrote a FLEURS version (`v005`+), bundle that
one instead and use `--resume` in step 3c.

Output in `kaggle/dist/`, each verified to be free of backslashes:

| bundle | contents | size |
| --- | --- | --- |
| `code.zip` | `ns.py`, `src/`, `requirements.txt`, `commands.txt`, `copy_repo.sh` | ~0.1 MB |
| `data.zip` | `data/fleurs_ceb_ph/`, `data/raw/`, rewritten `data/processed/*` | ~1.6 GB |
| `ckpt.zip` | `checkpoints/mms/v004/` + rewritten `latest.json` / `best.json` | ~1.1-1.6 GB |

Use `--no-audio` for a fast dry run (rewrites + verification only), and
`--line checkpoints/<name>` for a non-`mms` line. `--check` re-verifies an
existing `dist/`.

**Do not bundle mid-save.** A version directory appears before `rng.pt` is
written; wait for the "saved" log line, then copy.

## 2. Upload (3 Kaggle Datasets, Create new dataset -> upload zip)

| local file | dataset name | mounts at |
| --- | --- | --- |
| `code.zip` | `neural-speech-code` | `/kaggle/input/neural-speech-code` |
| `data.zip` | `neural-speech-data` | `/kaggle/input/neural-speech-data` |
| `ckpt.zip` | `neural-speech-ckpt` | `/kaggle/input/neural-speech-ckpt` |

**Notebook settings:** GPU on, **Internet on** (the `facebook/mms-300m` backbone
is downloaded at startup; with Internet off you must also upload your local HF
cache). Accelerator: pick one T4 (16 GB) or P100 — this project uses
`cuda:0` only, so `T4 x2` does not give 32 GB.

## 3. Notebook cells

```python
# --- 3a. assemble a writable repo under /kaggle/working -------------------
# /kaggle/input is read-only, so copy out instead of symlinking into it
!cd /kaggle/working && unzip -q /kaggle/input/neural-speech-code/code.zip -d repo
!cd /kaggle/working && unzip -q /kaggle/input/neural-speech-data/data.zip -d repo
!cd /kaggle/working && unzip -q /kaggle/input/neural-speech-ckpt/ckpt.zip -d repo
```

```python
# --- 3b. dependencies (no torch/torchaudio reinstall - Kaggle ships them) ---
!pip install -q "transformers>=4.45" "soundfile>=0.12" "numpy>=1.24"
!pip install -q "huggingface_hub[hf_transfer]" || pip install -q huggingface_hub
```

Keep the downloaded backbone outside `/kaggle/working` churn, and set
`HF_HUB_OFFLINE=1` for any *later* session that reuses the cache:

```python
%env HF_HOME=/kaggle/working/hf-cache
```

```python
# --- 3c. continue onto FLEURS on the GPU ----------------------------------
# -continue (NOT --resume) - see the warning below.
!cd /kaggle/working/repo && CUDA_VISIBLE_DEVICES=0 python ns.py train \
    -continue -c checkpoints/mms \
    --data fleurs_ceb_ph \
    -mtl 300 -d cuda:0 -e 30 \
    --max-batch-frames 1600000 --num-workers 2
```

`-mtl 300` is mandatory: the default 64 truncates 98% of FLEURS hypotheses and
inflates CER.

### Use `-continue`, not `--resume`, until a v005 exists

`v004` is a **word-corpus** model: 26 tokens, `manifest.csv`. FLEURS needs 81.
Vocabulary growth happens *only* on the `--from-checkpoint` path
(`_load_datasets`), never on `--resume` — resume assumes its data is unchanged
and its model was already restored at the parent's vocabulary.

| command | tokenizer | outcome |
| --- | --- | --- |
| `-continue -c checkpoints/mms` | 26 → 81 (55 added) | correct — restores 421 parent tensors, fresh-init only the 2 resized CTC head tensors |
| `--resume checkpoints/mms/v004` | stays 26 | **silently broken** — 55 characters become UNK |

So the rule is: bundle the newest version you actually have, and
- no FLEURS version written yet → `-continue`
- resuming a run that already trained on FLEURS (`v005`+) → `--resume checkpoints/mms/vNNN`

`-continue` also auto-wires replay and regression from the parent's stored
dataset (`data/processed/manifest.csv`, ratio 0.3), because that file ships in
`data.zip`. Expect this line at startup:

```
-continue: parent data data\processed\manifest.csv reused for replay (ratio 0.30) and regression eval
Incremental vocabulary: 26 -> 81 token(s) (55 added for the new data)
```

If you see `parent dataset ... not found; skipping replay/regression`, the word
manifest is missing from the bundle — re-run the bundler without `--no-audio`.

`-continue` intentionally starts a **fresh** optimizer and scheduler (it is a new
dataset, not a pause). The LR schedule restarts from the configured warmup.

## 4. Frame budget

`--max-batch-frames` counts padded 16 kHz samples, not encoder frames. Measured
on a 6 GB RTX 3050: 400k → 2.1 GB / 12.5 min per epoch, 800k → 2.7 GB / 9.9 min
(fastest), 1.6M → 3.7 GB / 15 min, 6M → OOM. A 16 GB T4 has room for more, but
that is untested — **start at 1.6M** and raise it if a run logs no OOM skips. A
batch that OOMs is logged and skipped, so the run survives but loses rows.

The cap applies to **validation too**, not just training. This matters: FLEURS
validation has no single-word clips, so all 225 rows land in the "sentence" view,
and an uncapped window there reached 4.65M padded samples against an 800k
training cap — enough to OOM *after* a finished epoch. If you raise the budget,
watch the `sentence validation` line as well as the training batches.

RAM is not a concern: the loader is lazy, a run sits at ~260 MB plus the ~1.6 GB
transient of writing a checkpoint, against ~13 GB on Kaggle. The real ceilings
are the 30 GPU-hours/week quota and the ~12 h session cap.

## 5. Get the checkpoints back

Checkpoints are git-ignored and `/kaggle/working` is wiped on session end, so
download explicitly — `FileDownloader` in a cell, or a `kagglehub`-style output
commit:

```python
from kaggle_secrets import EnvironmentNotebookException  # noqa: F401
import shutil, os
shutil.make_archive('/kaggle/working/checkpoints-mms', 'zip', '/kaggle/working/repo/checkpoints')
print(os.path.getsize('/kaggle/working/checkpoints-mms.zip') / 2**20, 'MB')
```

Expect ~6 GB: auto-prune keeps best + final plus whatever `--retain-every` allows.

## 6. Unzipping a downloaded bundle back on Windows

The rewritten manifests use forward slashes, which Windows accepts, but so do
the originals. To go back, extract `ckpt.zip` and let the line rebuild its
indexes: `latest.json` / `best.json` are regenerated on the next save, and
`CheckpointManager.resolve` tolerates either separator.
