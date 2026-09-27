# AGENTS.md

OpenCode working notes for the neural-speech repo. Everything here was verified
by running the commands in this environment (Windows, PowerShell).

## Core commands

```powershell
python ns.py prepare                        # index.txt -> data/processed/manifest.csv
python ns.py validate --fail-on-error       # dataset checks; exits 1 on errors
python ns.py datasets                       # list --data datasets + split status
python ns.py datasets --data fleurs_ceb_ph  # resolve one; writes the combined manifest
python ns.py datasets --data fleurs_ceb_ph --build   # force a rebuild from the export
python ns.py train                          # train; see flags below
python ns.py train -new -d auto -e 30       # fresh run on current manifest.csv (backbone defaults to fb)
python ns.py train -new --data default ...  # explicit: data/processed/manifest.csv, random 80/10/10
python ns.py train -new --data fleurs_ceb_ph -mtl 300   # FLEURS, official train/val/test split
python ns.py train -continue -d auto -e 30  # continue from best version onto new data (auto replay + regression)
python ns.py train -continue latest ...     # same, but parent = newest version
python ns.py train -resume -d auto -e 30    # same-run resume: auto-resolves checkpoints/<newest-line>/latest
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
python -m src.data.registry                 # == ns.py datasets
python -m src.training.train
python -m src.inference.transcriber <audio> --checkpoint checkpoints/latest
python -m src.deploy.optimize
uvicorn src.api.main:app --port 8000        # FastAPI (model loaded once at startup)
python -m pytest tests -q                   # 148 tests; no real dataset needed
```

`ns.py` is a shell-agnostic wrapper: it forwards all subcommand args verbatim to the
`src.*` modules, so any flag accepted by `python -m src.<module>` works identically.
Run `python ns.py train --help` for the full flag list. `TRAIN_FLAGS` (module-level in
`ns.py`) lists the flags that make a bare `ns.py --data ...` / `ns.py -e 30` mean
`ns.py train ...`.

Custom commands are registered in `ns.py`'s `COMMANDS` dict, each mapping a subcommand
name to `(module_main, one-line_help)`. Adding a new command means only: add an entry to
`COMMANDS`, keep `__main__` + `main()` in the target module (so both `ns.py <cmd>` and
`python -m src.<module>` work), and it shows up automatically in `python ns.py help`.
Args are forwarded raw — parser options live in the target module's argparse, not `ns.py`.

## Datasets (`--data`)

`src/data/registry.py` maps a name (or path) to a manifest plus, when the corpus ships
one, its official partition. `ns.py train --data <name|path>` and `ns.py datasets`:

- `--data default` — `data/processed/manifest.csv` (from `ns.py prepare`), random 80/10/10.
- `--data fleurs_ceb_ph` — `data/fleurs_ceb_ph/{train,validation,test}/manifest.tsv`, official
  split honored (3261/225/541 rows). `--text-field raw_transcript` switches to the
  un-normalized column.
- A dataset **directory** (with `train/validation/test` holding `manifest.tsv`) or a plain
  manifest CSV/TSV path also resolve, so ad-hoc exports need no registry edit.
- `--data` and `--dataset` are mutually exclusive; `_apply_data_defaults` runs *before*
  `_apply_mode_defaults`, so `-continue` still finds the parent's manifest and wires it as
  replay/regression reference.
- A preset dataset is materialized into `data/processed/manifest_<name>.csv` with the split
  name in the `section` column, and the split is saved as the usual
  `split_manifest_<name>.json` (fingerprint = SHA-256 of the combined manifest). Everything
  downstream — tokenizer rebuild, split reuse on resume, checkpoint manifests, `update_best`
  dataset awareness — is unchanged. The combined manifest is rewritten only when a source
  `manifest.tsv` is newer, so its fingerprint stays stable and resume reuses the split.
- Registered datasets are listed in `_builtin_datasets()`; `DatasetSpec` is a plain frozen
  dataclass, so a new corpus = one entry.

Training modes (mutually exclusive, `ns.py train --help` for full flag list):
- `-new` — fresh run on the current dataset (rebuilds tokenizer + splits, overwrites stale
  split files). No `-m` defaults to `fb`.
- `-continue [best|latest]` — incremental continuation onto the current dataset: parent is the
  lowest-`validation_loss` version (default) or `latest`; fresh optimizer, records
  `parent_checkpoint`, and auto-wires `--replay-manifest`/`--eval-manifest` from the parent's
  stored dataset manifest when it differs from the current one (ratio 0.3).
- `-resume [<checkpoint>]` — same-run continuation (restores optimizer/scheduler/epoch/RNG).
  Bare `-resume` auto-resolves `checkpoints/<newest-line>/latest`.
- `--from-checkpoint <dir>` — explicit incremental parent (no auto-wiring).
Incremental runs **inherit the parent's trained weights**, not just its config: the parent's
  `model_sd` is stashed in `_prepare_epoch_from_checkpoint` and applied by
  `_load_parent_weights` *after* the tokenizer has been extended, so the vocabulary-sized
  tensors (`decoder.embed.weight`, `decoder.out.weight/bias`) can be resized instead of
  silently falling back to a fresh random model. `_load_datasets` appends the new data's
  characters to the parent vocabulary (existing ids keep their trained embeddings);
  `--resume` deliberately does **not** grow it (its data is unchanged). Two tests in
  `tests/test_train.py` lock this in (`..._starts_from_parent_weights`,
  `..._extends_parent_vocabulary`) — they caught a regression where the incremental path
  loaded the parent config/tokenizer and then threw every trained weight away.
When `-c`/`--checkpoint-dir` is omitted, `-continue`/`-resume` auto-detect the newest line
under `checkpoints/` (a subdir holding `latest.json`); `-new` defaults to `checkpoints`.
`CheckpointManager.resolve` also accepts a line root (e.g. `checkpoints/mms`) and resolves it
to that line's latest version. Never combine resume modes with each other.

## Checkpoint retention policy (STRICT — to keep repo size down)

Checkpoints are the only large artifact (a 30-epoch MMS run ≈ 46.5 GB). Policy,
enforced by the trainer's auto-prune (default ON):

- Keep **only**: the recorded best (`best.json`, maintained by `update_best`), the
  best version by on-disk `validation_loss` (flag 'best'), and the **final** version
  (newest, `latest.json` stays valid). Positive `--retain-every N` additionally
  keeps every Nth version for trend visibility; default `0` is best-only plus final,
  while a negative value disables pruning.
- **`latest` ≠ `best`**: `latest.json` points at the most recent epoch; `best.json`
  at the best validation performance. Both survive pruning independently.
  `CheckpointManager.resolve`/`load` accept `best` / `checkpoints/mms/best`
  (works for `--resume best`, `--from-checkpoint best`, `--checkpoint .../best`).
- `update_best` is called by the trainer after every validated save and is
  **dataset-aware**: each `best.json` entry stores the manifest path + SHA-256
  fingerprint. Two minima are tracked independently — `val_loss` (canonical,
  drives the LR scheduler) and `cer` (CER and val_loss can peak at different
  epochs, e.g. the first combined run: val_loss best ~epoch 20, CER best
  ~epoch 28). On a fingerprint mismatch both entries are reset immediately, so
  an old dataset's lower val_loss never shadows a newer run's best and lets the
  interval pruner delete it — which is exactly what happened to the first
  combined word+sentence run (its best ~epoch 26-28 was deleted; only
  milestones + the old parent survived).
- `--retain-protect v026` (repeatable) adds any version that must never be deleted,
  regardless of the keep-set (e.g. published reference points). Per-line durable
  protection lives in `<line>/protect.json` (list of version names, e.g. `["v030"]`),
  auto-merged into *every* `prune()` call — no CLI flag needed on future runs, so a
  historical reference survives even retroactive pruning.
- Pruning runs automatically after every validated checkpoint save and again at the
  end of the run; deleted versions are logged at WARNING level. `CheckpointManager.prune`
  also has a `dry_run=True` for rehearsing.
- `--resume` continues from a *kept* version: resume from `.../latest` (always kept)
  or an explicit milestone/best/protected version. Resuming from a pruned intermediate
  version is not possible by design.
- Retroactive pruning: run a dry-run snippet against a line (e.g.
  `CheckpointManager('checkpoints/mms').prune(retain_every=0, keep_best=True, dry_run=True)`)
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
- **FLEURS export is not valid TSV**: `manifest.tsv` is written unquoted, and 37 rows
  contain a `"` plus a trailing tab inside `transcript`. `csv.reader` therefore merges
  columns and *silently drops the text* — `read_split_tsv` in `src/data/registry.py`
  splits on plain tabs and never uses the csv module. Every row does have exactly 5
  tab-separated fields; each split numbers its clips from `00000`, so ids are offset
  per split to stay unique in the combined manifest.
- **FLEURS transcripts are long**: median 144 chars (incl. BOS/EOS), p90 214, max 381
  (76 distinct characters total, so the char vocab stays tiny: 80 tokens). The default
  `-mtl 64` only bounds *eval decoding*, not training, so it truncates 98% of hypotheses
  and inflates CER — pass `-mtl 300` for FLEURS (0.8% truncated; 64→0 rows would cost
  ~70% more decode steps). `Trainer._warn_decode_budget` logs the truncated fraction and
  a suggested cap at startup.
- **Dataset blobs stay out of git**: `.gitignore` covers `data/fleurs_ceb_ph/` plus the
  one-off fetch helpers `huggingfacedata.py` / `movedata.py` (both hardcode
  `C:\Users\Sphinx0945\...` paths and delete the HF cache). Re-create the export with
  those scripts, or drop a `manifest.tsv` per split into `data/fleurs_ceb_ph/`.

## Architecture (non-obvious)

- The model learns `audio -> text tokens`. Numeric audio IDs are identifiers only and
  must never be used as labels. `STTModel.forward(audio, audio_lengths, tokens,
  token_lengths)` is teacher-forced; inference uses `greedy_decode` (appends EOS if
  `max_len` is reached).
- Data flow: `index.txt` (immutable) -> `manifest.csv` (derived) -> PyTorch Dataset ->
  model. Splits are persisted to `data/processed/split_<manifest_stem>.json` and
  reused on resume/continue so evaluation sets stay stable. Each split stores a SHA-256
  fingerprint of the manifest it was built from: if the manifest changed (new data),
  resume/continue detect it (fingerprint mismatch; legacy splits fall back to a row-count
  comparison) and re-split + overwrite with a warning instead of silently training on
  stale rows. Character tokenizer special tokens are at fixed indices 0-3 (PAD/UNK/BOS/EOS);
  vocabulary is saved per checkpoint version.
- `Trainer._partition_rows` is the single place the train/valid/test row lists are
  decided: the dataset's preset split when `config.preset_splits` is set (`--data` on a
  corpus that ships one), otherwise the historical seeded 80/10/10 `split_dataset`.
  Everything downstream only ever sees the three row lists.
- **Loading is lazy per sample, and batches are length-grouped.** `SpeechDataset.__getitem__`- **Loading is lazy per sample, and batches are length-grouped.** `SpeechDataset.__getitem__`
  decodes one file on demand via `load_audio` (peak-normalize + resample inside it) and keeps
  no cache; `collate_speech` materializes only the current batch, so a 916-minute corpus
  never sits in RAM. There is no bucket/chunk/shard preloading anywhere — a "bucket" is only
  an *ordering* device. `LengthGroupedBatchSampler` (dataset.py) shuffles all rows, sorts each
  megabatch by probed length, cuts batches, then shuffles batch order, advancing `self.epoch`
  per pass so each epoch is a different order. `probe_num_samples` reads headers only
  (soundfile `info`, no decode); unreadable files count as median length, not zero.
  `--max-batch-frames` caps `len(batch) * longest_clip` — the padded sample count, i.e. the
  direct VRAM bound — and matters far more than `-b` on long-audio corpora: FLEURS is 10x the
  word clips (median 13.0s vs 1.3s) and plain `-b 8` OOMs a 6GB card. Measured on the 3050:
  400k -> 2.1GB/12.5 min per epoch, 800k -> 2.7GB/9.9 min (best), 1.6M -> 3.7GB/15 min,
  6M -> OOM. Past 800k it gets *slower* (quadratic attention). A batch that still OOMs is
  logged and skipped (`OOM_ERRORS` covers both `torch.OutOfMemoryError` and
  `torch.AcceleratorError`, which are siblings, not parent/child) after `zero_grad` +
  `empty_cache`, so a multi-hour run survives a spike instead of dying. Optimizer/scheduler
  are built once and never reset per epoch/batch; only `-continue`/`--from-checkpoint`
  intentionally start a fresh optimizer.
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

- This machine has an NVIDIA GeForce RTX 3050 6GB (compute 8.6). Python is a
  native user install (3.14.7, `winget install Python.Python.3.14`, at
  `%LOCALAPPDATA%\Programs\Python\Python314\`) and `python` on PATH resolves to
  it ahead of the MS Store alias. It carries `torch 2.14.0+cu130` /
  `torchaudio 2.11.0+cu130` (installed from the `cu130` wheel index;
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130`).
  No venv — `python ns.py ...` / `pytest` run straight on the system Python.
  The old `.venv` was deleted (2026-09-24). `--device auto` lands on CUDA, and fp16
  AMP (autocast + `torch.amp.GradScaler("cuda", ...)`) turns on automatically for
  CUDA training. CPU throughput was ~5.5s/step; the 3050 does an epoch (20
  steps + eval + 1.26GB checkpoint save) in ~11s.
- The MMS fine-tune line lives at `checkpoints/mms/` (`facebook/mms-300m`,
  hidden 1024, last-4 transformer layers unfrozen = 50.4M trainable of 315.5M,
  ~1.26GB/model.pt). Verified end-to-end: 30 epochs, resumed `v009 -> v030`,
  final Test CER=0.2823 / WER=0.7000, and the collapse probe reports NO
  COLLAPSE (pooled cosine 0.937-0.995 vs 1.0000 bit-identical before).
- `src/deploy/onnx_export.py` TensorRT path is skipped gracefully when
  unavailable. Windows paths use backslashes — tests must use `os.path.join`
  when asserting path suffixes. Manifests and `latest.json`/`best.json` therefore
  store OS-native separators: **they are not portable to Linux as-is.** Anything
  targeting Linux rewrites them at upload time; the local pipeline never needs it.