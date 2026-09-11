# AGENTS.md

OpenCode working notes for the neural-speech repo. Everything here was verified
by running the commands in this environment (Windows, PowerShell).

## Core commands

```powershell
python ns.py prepare                        # index.txt -> data/processed/manifest.csv
python ns.py validate --fail-on-error       # dataset checks; exits 1 on errors
python ns.py train                          # train; see flags below
python ns.py transcribe <audio> --checkpoint checkpoints/latest
python ns.py optimize                       # INT8 quant + ONNX export
python -m src.data.prepare                  # underlying equivalents (same args)
python -m src.data.validate --fail-on-error
python -m src.training.train
python -m src.inference.transcriber <audio> --checkpoint checkpoints/latest
python -m src.deploy.optimize
uvicorn src.api.main:app --port 8000        # FastAPI (model loaded once at startup)
python -m pytest tests -q                   # 50 tests; no real dataset needed
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

## Gotchas (all hit in practice)

- **Audio decoding**: torchaudio 2.11 + torchcodec needs FFmpeg "full-shared" installed
  on Windows; without it `torchaudio.load` fails. The loader in `src/data/audio.py`
  automatically falls back to soundfile. Tests exploit this: fixtures write real WAV
  bytes under `.opus` filenames (decoding is content-based, so it works without FFmpeg).
  Real `.opus` transcription relies on the fallback too.
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
- `torch.load(..., weights_only=False)` is required for optimizer/scheduler/RNG files
  (torch ≥2.6 defaults to weights_only=True and fails).

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

## Environment

- Only Python 3.14 + CPU torch on this machine; `src/deploy/onnx_export.py` TensorRT
  path is skipped gracefully when unavailable. Windows paths use backslashes — tests
  must use `os.path.join` when asserting path suffixes.