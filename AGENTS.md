# AGENTS.md

OpenCode working notes for the neural-speech repo. Everything here was verified
by running the commands in this environment (Windows, PowerShell).

## Core commands

```powershell
python ns.py prepare                        # index.txt -> data/processed/manifest.csv
python ns.py validate --fail-on-error       # dataset checks; exits 1 on errors
python ns.py train                          # train; see flags below
python ns.py transcribe <audio> --checkpoint checkpoints/latest
python ns.py mic --seconds 5                # record mic + transcribe (real-time test)
python ns.py mic --loop --seconds 3         # keep going until Ctrl+C
python ns.py mic --tts-backend piper        # also speak the result back
python ns.py mic --tts-backend none         # transcription only
python ns.py optimize                       # INT8 quant + ONNX export
python ns.py devices                        # detect compute devices (CUDA/MPS/CPU)
python ns.py devices --check cuda           # resolve a device string like --device does
python -m src.data.prepare                  # underlying equivalents (same args)
python -m src.data.validate --fail-on-error
python -m src.training.train
python -m src.inference.transcriber <audio> --checkpoint checkpoints/latest
python -m src.deploy.optimize
uvicorn src.api.main:app --port 8000        # FastAPI (model loaded once at startup)
python -m pytest tests -q                   # 85 tests; no real dataset needed
```

`ns.py` is a shell-agnostic wrapper: it forwards all subcommand args verbatim to the
`src.*` modules, so any flag accepted by `python -m src.<module>` works identically.
Run `python ns.py train --help` for the full flag list.

Custom commands are registered in `ns.py`'s `COMMANDS` dict, each mapping a subcommand
name to `(module_main, one-line_help)`. Adding a new command means only: add an entry to
`COMMANDS`, keep `__main__` + `main()` in the target module (so both `ns.py <cmd>` and
`python -m src.<module>` work), and it shows up automatically in `python ns.py help`.
Args are forwarded raw — parser options live in the target module's argparse, not `ns.py`.

Training flags that matter: `--resume <dir>` (restores optimizer/scheduler/epoch/RNG),
`--from-checkpoint <dir>` (incremental: fresh optimizer, records `parent_checkpoint`),
`--replay-manifest <csv> --replay-ratio 0.3`, `--eval-manifest <old-csv>` (regression eval).
Never combine `--resume` and `--from-checkpoint`.

## Checkpoint retention policy (STRICT — to keep repo size down)

Checkpoints are the only large artifact (a 30-epoch MMS run ≈ 46.5 GB). Policy,
enforced by the trainer's auto-prune (default ON):

- Keep **only**: the best version by `validation_loss` (flag 'best'), the **final**
  version (newest, `latest.json` stays valid), and **every Nth version** for trend
  visibility — `--retain-every N` (default **10**; 0 = disable pruning, keep all).
- `--retain-protect v026` (repeatable) adds any version that must never be deleted,
  regardless of the keep-set (e.g. published reference points).
- Pruning runs automatically after every validated checkpoint save and again at the
  end of the run; deleted versions are logged at WARNING level. `CheckpointManager.prune`
  also has a `dry_run=True` for rehearsing.
- `--resume` continues from a *kept* version: resume from `.../latest` (always kept)
  or an explicit milestone/best/protected version. Resuming from a pruned intermediate
  version is not possible by design.
- Retroactive pruning: run a dry-run snippet against a line (e.g.
  `CheckpointManager('checkpoints/mms').prune(retain_every=10, keep_best=True, dry_run=True)`)
  then apply with `dry_run=False`. 46.5 GB (mms line) → ~6.2 GB with the default
  keep-set (v010, v020, v026 best, v030 final). Applied as of 2026-09-24:
  retroactive choose-best-only pruning left `checkpoints/mms` at **v026 (best) +
  v030 (final) = 3.1 GB**; repo total dropped 49.4 → 6.0 GB.
- Weights are **never tracked by git**: `git ls-files checkpoints` shows only
  `checkpoints/.gitkeep`; `.gitignore` has `checkpoints/*` + `!checkpoints/.gitkeep`.
  If off-site backup is needed, use DVC / cloud object storage with the *same*
  keep-set policy — do not accumulate locally or in git history.

Device selection is portable: every CLI that touches a model takes `--device`
(`auto` | `cuda[:N]` | `mps` | `cpu`). `auto` picks CUDA -> MPS -> CPU by
availability; requesting a device this machine lacks raises a ValueError that
lists what it has (instead of a late CUDA crash). `ns.py devices` prints that
list and what `--device auto` would select.

## Gotchas (all hit in practice)

- **Audio decoding**: torchaudio 2.11 + torchcodec needs FFmpeg "full-shared" installed
  on Windows; without it `torchaudio.load` fails. The loader in `src/data/audio.py`
  automatically falls back to soundfile. Tests exploit this: fixtures write real WAV
  bytes under `.opus` filenames (decoding is content-based, so it works without FFmpeg).
  Real `.opus` transcription relies on the fallback too.
- **Mic capture** (`src/inference/mic.py`): records 16 kHz mono float32 via sounddevice
  (wheel bundles PortAudio, no system install) to a temp WAV, then runs it through the
  same `Transcriber` as the CLI/API. The sounddevice import is lazy, so commands like
  `ns.py help` still work if it's not installed.
- **ONNX export** (`src/deploy/onnx_export.py`) must pass `dynamo=False` — the default
  dynamo exporter needs `onnxscript`, which is not a dependency. The exported graph
  wraps the model because `STTModel.forward` takes 4 args; the wrapper derives
  `token_lengths` from token shape.
- **Dynamic quantization** (`src/deploy/quantize.py`): `quantize_dynamic` must get a
  `qconfig_spec = {nn.Embedding: float_qparams_weight_only_qconfig}` or it raises
  "Embedding quantization is only supported with float_qparams...".
- **Checkpoints are immutable `vNNN/` dirs; one version is saved per validated epoch**.
  With the default `--validation-frequency 1`, a 30-epoch run creates 30 versions plus
  final test-metric updates via `update_latest_state`. Never save into an existing
  version dir; the manager errors if you do. `latest.json` points at the newest.
  Auto-pruning (see retention policy above) deletes intermediate versions as a run
  proceeds, so a 30-epoch run keeps ~4 versions by default, not 30.
- `torch.load(..., weights_only=False)` is required for optimizer/scheduler/RNG files
  (torch ≥2.6 defaults to weights_only=True and fails).
- **TTS speaker** (`src/inference/speaker.py`): piper-tts 1.8 no longer has an
  auto voice downloader — voices (`.onnx` + `.onnx.json`) must be fetched from
  HuggingFace `rhasspy/piper-voices` into `voices/`. Its `AudioChunk` bytes are
  raw int16 PCM (no RIFF header); `_merge_piper_chunks` wraps them. `piper-tts`
  installs on 3.14 via a cp39-abi3 wheel; `pyttsx3` falls back to OS voices.
  The `online` backend name is registered but unimplemented (edge-tts/Azure hook).
- **Hugging Face inference cache**: MMS loading explicitly requests safetensors
  and supports `--offline` on `ns.py mic` / standalone transcription. Set
  `HF_HUB_OFFLINE=1` for repeated local runs after the backbone is cached; set
  `HF_TOKEN` only when the Hugging Face repository requires authentication.

## Architecture (non-obvious)

- The model learns `audio -> text tokens`. Numeric audio IDs are identifiers only and
  must never be used as labels. `STTModel.forward(audio, audio_lengths, tokens,
  token_lengths)` is teacher-forced; inference uses `greedy_decode` (appends EOS if
  `max_len` is reached).
- Data flow: `index.txt` (immutable) -> `manifest.csv` (derived) -> PyTorch Dataset ->
  model. Splits are persisted to `data/processed/split_<manifest_stem>.json` and
  reused on resume so evaluation sets stay stable. Character tokenizer special tokens
  are at fixed indices 0-3 (PAD/UNK/BOS/EOS); vocabulary is saved per checkpoint
  version.
- `Transcriber` (used by both the standalone CLI and FastAPI) loads vocabulary +
  config from the checkpoint itself — never duplicate preprocessing in new inference
  code.
- Reusable audio helper returns an `AudioResult` dataclass (not a tuple); unpacking
  it like a tuple raises. Use `.waveform` / `.sample_rate` or the `load_audio_raw`
  tuple alias.
- `resolve_device` lives in `src/device_info.py` and is re-exported from
  `src.training.train` (transcriber/API still import it from there).
- The MMS backbone (`STTConfig.backbone`, `MMSAudioEncoder`) normalizes each
  utterance to zero-mean/unit-var, builds the wav2vec2 attention mask from
  `audio_lengths` (1 = valid), zeroes all backbone dropout at load, and freezes
  everything except the last `backbone_unfreeze_layers` (default 4) transformer
  layers. `d_model` is overridden to the backbone's hidden size (1024) before
  the head is built.
- **Encoder-collapse probe** (`src/inference/probe.py`, `ns.py probe`): pairwise
  cosine is dominated by a huge shared component in wav2vec2-style encoders — a
  *healthy* backbone still shows ~0.97-0.99 pooled cosine across unrelated clips.
  So the verdict keys on input-dependence, not cosine magnitude: max abs element
  diff between pooled encodings, the input-specific variance fraction
  `||p_i - mean||^2 / ||p_i||^2`, and centered (mean-subtracted) cosine. The
  frame-sequence metric (first-K-frames concatenated) is far more discriminative
  than time-pooling.

## Environment

- This machine has an NVIDIA GeForce RTX 3050 6GB (compute 8.6). The project
  venv (`.\venv\Scripts\python.exe`) carries `torch 2.14.0+cu130` /
  `torchaudio 2.11.0+cu130` (installed from the `cu130` wheel index), which
  shadow the CPU build still present inside the base `D:\python` distribution
  — venv site-packages precedes the `base_site.pth` path, so always run via
  the venv python. `--device auto` now lands on CUDA, and fp16 AMP
  (autocast + `torch.amp.GradScaler("cuda", ...)`) turns on automatically for
  CUDA training. CPU throughput was ~5.5s/step; the 3050 does an epoch (20
  steps + eval + 1.26GB checkpoint save) in ~11s.
- The MMS fine-tune line lives at `checkpoints/mms/` (`facebook/mms-300m`,
  hidden 1024, last-4 transformer layers unfrozen = 50.4M trainable of 315.5M,
  ~1.26GB/model.pt). Verified end-to-end: 30 epochs, resumed `v009 -> v030`,
  final Test CER=0.2823 / WER=0.7000, and the collapse probe reports NO
  COLLAPSE (pooled cosine 0.937-0.995 vs 1.0000 bit-identical before).
- `src/deploy/onnx_export.py` TensorRT path is skipped gracefully when
  unavailable. Windows paths use backslashes — tests must use `os.path.join`
  when asserting path suffixes.