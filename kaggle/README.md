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

## 1. Build the bundles (on Windows)

Wait until the background run reports a saved version, then bundle that one
version (each is ~1.55 GB; never upload the whole line):

```powershell
python kaggle/make_upload.py --version v004
```

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
# --- 3c. resume the run on the GPU ----------------------------------------
# --resume keeps optimizer/scheduler/epoch/RNG; -continue would reset them.
# Re-declare replay/eval explicitly: -resume does not auto-wire them.
!cd /kaggle/working/repo && CUDA_VISIBLE_DEVICES=0 python ns.py train \
    --resume checkpoints/mms/v004 \
    -c checkpoints/mms \
    --data fleurs_ceb_ph \
    --replay-manifest data/processed/manifest.csv --replay-ratio 0.3 \
    -mtl 300 -d cuda:0 -e 30 \
    --max-batch-frames 1600000 --num-workers 2
```

`-mtl 300` is mandatory: the default 64 truncates 98% of FLEURS hypotheses and
inflates CER.

## 4. Frame budget

`--max-batch-frames` counts padded 16 kHz samples, not encoder frames. Measured
on a 6 GB RTX 3050: 400k → 2.1 GB / 12.5 min per epoch, 800k → 2.7 GB / 9.9 min
(fastest), 1.6M → 3.7 GB / 15 min, 6M → OOM. A 16 GB T4 has room for more, but
that is untested — **start at 1.6M** and raise it if a run logs no OOM skips. A
batch that OOMs is logged and skipped, so the run survives but loses rows.

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
