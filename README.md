# neural-speech

An experimental custom neural speech system built with PyTorch. The initial
milestone is **speech-to-text (STT)** trained from your own audio dataset, with
the architecture structured so **text-to-speech (TTS)** can be added later
without rewriting the core.

The project does **not** use a pretrained STT/TTS model as the core neural model.
It trains an original, small, understandable model from scratch on your own
recordings, with a strong focus on:

- interpretable data pipeline (index → manifest → dataset)
- versioned, immutable checkpoints
- incremental / resumed training (learning new data while keeping old ability)
- replay training to reduce catastrophic forgetting
- regression evaluation (old vs new data) after incremental training
- deployment optimization (INT8 quantization, ONNX, optional TensorRT)

---

## 1. Project purpose

Learn the mapping

```
audio → text
```

from isolated-word recordings today, and from sentence recordings in the future.
The numeric audio ID is only an identifier — it is **never** used as a class label.

## 2. Dataset structure

```
data/
└── raw/
    ├── index.txt
    └── assets/
        ├── 1.opus
        ├── 2.opus
        └── ...
```

## 3. Expected index.txt format

`index.txt` groups words under `[section]` headers. Sections are organizational
only and are **not** used as classes.

```
[ad]
adlaw = 104
adto = 83

[ak]
ako = 24
akong = 35
```

Meaning: `104.opus → "adlaw"`, `83.opus → "adto"`, ...

The original `index.txt` is immutable source data. It is never modified.

## 4. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or: source .venv/bin/activate (Linux/macOS)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Opus decoding**: torchaudio (via torchcodec) needs FFmpeg for `.opus`.
> Install the "full-shared" FFmpeg build, or rely on the built-in **soundfile**
> fallback (libsndfile supports WAV/FLAC/OGG/Opus) which needs no extra setup.
> The audio loader tries torchaudio first, then soundfile automatically.

Install torch with GPU support (optional, recommended for training):

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 5. Dataset preparation

Parse `index.txt` into a normalized manifest:

```powershell
python -m src.data.prepare
# custom paths
python -m src.data.prepare --index data/raw/index.txt --output data/processed/manifest.csv
```

Produces `data/processed/manifest.csv`:

```csv
id,audio,text,section
104,data/raw/assets/104.opus,adlaw,ad
83,data/raw/assets/83.opus,adto,ad
```

## 6. Dataset validation

```powershell
python -m src.data.validate
python -m src.data.validate --fail-on-error   # non-zero exit code on errors
```

Checks: invalid index lines, duplicate IDs/paths, missing or corrupt audio,
empty transcriptions, invalid text encoding, unexpected filenames, and prints a
summary of valid/invalid counts.

## 7. Initial training

```powershell
python -m src.training.train \
    --dataset data/processed/manifest.csv \
    --epochs 30 --batch-size 8 --device auto
```

Outputs per epoch: `train_loss`, `val_loss`, `CER`, `WER`, and `lr`. Mixed
precision (AMP) is used automatically when CUDA is available.

## 8. Checkpoint structure

```
checkpoints/
├── v001/
│   ├── model.pt          # model weights
│   ├── optimizer.pt      # optimizer state
│   ├── scheduler.pt      # scheduler state
│   ├── vocabulary.json   # trained character tokenizer
│   ├── config.json       # model configuration
│   ├── training_state.json  # version, epoch, step, metrics, parent
│   ├── manifest.json     # dataset/replay metadata
│   └── rng.pt            # random states for reproducibility
├── v002/
├── v003/
└── latest.json           # points to the newest version
```

Checkpoints are **immutable versions** — a new training run always creates a new
`vNNN` directory and never overwrites an old one.

## 9. Resume training

```powershell
python -m src.training.train \
    --resume checkpoints/v001 \
    --epochs 60
```

Restores model weights, optimizer state, scheduler state, vocabulary, config,
epoch, step, and random state where practical, then continues.

## 10. Incremental training

```powershell
python -m src.training.train \
    --from-checkpoint checkpoints/v001 \
    --dataset data/processed/manifest_v002.csv \
    --eval-manifest data/processed/manifest_v001.csv
```

Initializes from v001, trains on the new dataset, and produces a new version
(e.g. v002). The parent is recorded in `training_state.json`, and the old
checkpoint is never touched.

## 11. Replay training (reducing catastrophic forgetting)

```powershell
python -m src.training.train \
    --from-checkpoint checkpoints/v001 \
    --dataset data/processed/manifest_v002.csv \
    --replay-manifest data/processed/manifest_v001.csv \
    --replay-ratio 0.30
```

Mixes 30% samples from the previous dataset into every training epoch's stream,
70% new data, to reduce catastrophic forgetting. Replay reduces but does not
completely prevent forgetting.

## 12. Evaluation / regression

Every incremental run should be evaluated against both the new dataset and the
old/reference dataset:

```powershell
python -m src.training.train \...    # --eval-manifest data/processed/manifest_v001.csv
```

The reference WER/CER are stored in `training_state.json["regression"]` so you can
compare before/after:

```json
"regression": {
  "reference_manifest": "data/processed/manifest_v001.csv",
  "cer": 0.12,
  "wer": 0.14
}
```

## 13. Standalone inference

```powershell
python -m src.inference.transcriber sample.opus --checkpoint checkpoints/latest
```

Or programmatically:

```python
from src.inference.transcriber import Transcriber

transcriber = Transcriber(checkpoint="checkpoints/v001")
text = transcriber.transcribe("sample.opus")
print(text)
```

Inference uses the same audio preprocessing, vocabulary, and model config saved
with the checkpoint — nothing is duplicated.

## 14. FastAPI usage

Start the API (loads the model once at startup):

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Environment variables:

```powershell
$env:STT_CHECKPOINT = "checkpoints/latest"   # version alias or path
$env:STT_DEVICE      = "auto"                # auto | cpu | cuda
$env:API_HOST        = "0.0.0.0"
$env:API_PORT        = "8000"
```

Endpoints:

```
GET /health
  → {"status": "ok"}

POST /stt   (multipart/form-data, field "file")
  → {"text": "recognized text"}
```

Curl example:

```powershell
curl.exe -X POST http://localhost:8000/stt -F "file=@sample.opus"
```

The API fails clearly at startup if the configured checkpoint does not exist.
It is an **inference layer only** — training logic is not exposed.

## 15. Deployment optimization

The checkpoint module ships deployment helpers to shrink/lower-latency the model:

```powershell
python -m src.deploy.optimize --checkpoint checkpoints/latest
```

This produces:

- `checkpoints/<ver>/quantized/model_quantized.pt` — PyTorch **dynamic INT8
  quantization** (works best on CPU; Embedding uses per-channel float qparams)
- `checkpoints/<ver>/onnx/model.onnx` — standalone ONNX graph (dynamic axes)

Optional TensorRT (NVIDIA GPU):

```powershell
python -m src.deploy.optimize --checkpoint checkpoints/latest --tensorrt --tensorrt-precision fp16
```

On machines without TensorRT the step is skipped gracefully. For real latency
budgets, benchmark float32 vs quantized vs ONNX in your serving environment.

## 16. Future TTS architecture

TTS can be added later without touching STT code, following this layout:

```
src/
├── models/
│   ├── stt.py
│   └── tts.py
├── inference/
│   ├── transcriber.py
│   └── synthesizer.py
└── api/
    └── routes/
        ├── stt.py
        └── tts.py
```

## 17. Tests

```powershell
python -m pytest tests -q
```

Tests use tiny synthetic audio samples and never require the full dataset.

## 18. Key design rules

1. The numeric audio ID is an identifier, **not** a class label.
2. The model learns `audio → text`, never `audio → ID`.
3. `index.txt` is immutable source data.
4. Manifests are derived data.
5. Checkpoints are immutable versions.
6. New training starts from an existing checkpoint when requested.
7. Old checkpoints are never silently overwritten.
8. Incremental training can replay old data to reduce forgetting.
9. Every new version should be evaluated against new **and** reference data.
10. FastAPI is an inference layer, not a training layer.
11. The system works with isolated-word data today and supports future sentences.