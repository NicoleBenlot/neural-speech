"""Encoder-collapse probe + greedy-decode holdout evaluation.

Diagnostic: the from-scratch encoder collapsed during training — the trained
encoder output was bit-identical (pairwise cosine = 1.0000) across all held-out
probe clips regardless of input audio, proving the model learned nothing about
audio at this data scale.

This command replays that probe on any checkpoint:

1. **Collapse probe (primary signal):** for each held-out clip, embed the audio
   with the model's encoder, time-pool over valid frames, and compute the
   pairwise cosine similarity between clip embeddings. If the encoder is
   healthy, cosine varies meaningfully (well below ~1.0). A mean pairwise
   cosine ~1.0000 means the encoder output no longer depends on input audio.

2. **Greedy-decode holdout table:** decode each held-out clip greedily
   (CTC collapse-repeats/remove-blank) and report per-clip CER/WER plus
   aggregates.

Usage:
    python -m src.inference.probe --checkpoint checkpoints/mms/latest \\
        --split data/processed/split_manifest.json
    python -m src.inference.probe --checkpoint checkpoints/mms/latest --untrained
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from src.data.audio import load_audio
from src.data.parser import parse_index
from src.models.stt import STTModel
from src.training.checkpoint import CheckpointManager
from src.training.metrics import compute_metrics, cer, wer
from src.training.train import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_held_out_rows(split_path: str, dataset: str) -> List[dict]:
    """Return the test (held-out) rows, from the persisted split when present."""
    split = Path(split_path)
    if split.exists():
        data = json.loads(split.read_text(encoding="utf-8"))
        rows = data.get("test") or []
        if rows:
            return rows
        logger.warning("Split file has no 'test' section; falling back to manifest split")

    from src.data.dataset import read_manifest, split_dataset
    from src.data.prepare import write_manifest

    if Path(dataset).exists():
        rows_all = read_manifest(Path(dataset))
        _, _, test_rows = split_dataset(rows_all, seed=42)
        return [r.__dict__ for r in test_rows]

    parsed = parse_index(Path("data/raw/index.txt"))
    _, _, test_rows = split_dataset(parsed.entries, seed=42)
    return [r.__dict__ for r in test_rows]


def _load_probe_audio(rows: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    waveforms = []
    lengths = []
    for r in rows:
        result = load_audio(r["audio_path"], target_sr=16000)
        waveforms.append(result.waveform)
        lengths.append(result.waveform.shape[1])
    max_t = max(lengths)
    audio = torch.zeros(len(rows), 1, max_t)
    for i, w in enumerate(waveforms):
        audio[i, :, : w.shape[1]] = w
    return audio, torch.tensor(lengths)


def pairwise_cosine(matrix: torch.Tensor) -> torch.Tensor:
    normed = matrix / matrix.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return normed @ normed.T


def run_probe(
    model: STTModel,
    rows: List[dict],
    device: torch.device,
    batch: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute pooled, frame-sequence, and per-clip encoder embeddings.

    Returns (pooled, seq), two pairwise-cosine matrices, plus the pooled
    vectors and their valid-frame counts for the bit-identical test.
    """
    model.eval()
    pooled_vecs = []
    seq_vecs = []
    max_frames = None
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        audio, lengths = _load_probe_audio(chunk)
        audio, lengths = audio.to(device), lengths.to(device)
        with torch.no_grad():
            encoder_out, enc_lengths = model.encode(audio, lengths)
        b, t, d = encoder_out.shape
        frame_mask = (
            torch.arange(t, device=device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        )
        denom = enc_lengths.float().clamp(min=1).unsqueeze(-1)
        pooled = (encoder_out * frame_mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled_vecs.append(pooled)
        n_frames = enc_lengths.min().item()
        max_frames = n_frames if max_frames is None else min(max_frames, n_frames)
    pooled = torch.cat(pooled_vecs, dim=0)

    # Re-extract a fixed-length frame prefix per clip for the sequence probe.
    seq_vecs = []
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        audio, lengths = _load_probe_audio(chunk)
        audio, lengths = audio.to(device), lengths.to(device)
        with torch.no_grad():
            encoder_out, enc_lengths = model.encode(audio, lengths)
        frames = encoder_out[:, :max_frames, :]  # (B, K, D)
        seq_vecs.append(frames.reshape(frames.shape[0], -1))
    seq = torch.cat(seq_vecs, dim=0)

    cos_pooled = pairwise_cosine(pooled)
    cos_seq = pairwise_cosine(seq)
    return cos_pooled, cos_seq, pooled, max_frames


def print_probe_report(
    cos_pooled: torch.Tensor,
    cos_seq: torch.Tensor,
    pooled: torch.Tensor,
    rows: List[dict],
    label: str,
):
    n = len(rows)
    print(f"\n=== Encoder-collapse probe ({label}, {n} held-out clips) ===")

    print("\n[time-pooled] pairwise cosine of encoder outputs:")
    print("        " + "".join(f"{r['id']:>7}" for r in rows))
    for i, r in enumerate(rows):
        line = f"{r['id']:>7}" + "".join(f"{cos_pooled[i, j].item():7.4f}" for j in range(n))
        print(line)
    off_p = cos_pooled.clone()
    off_p.fill_diagonal_(0.0)
    vp = off_p[off_p > 0]
    print(f"  off-diagonal: n={vp.numel()} mean={vp.mean().item():.4f} "
          f"std={vp.std().item():.4f} min={vp.min().item():.4f} max={vp.max().item():.4f}")

    print("\n[frame-sequence] pairwise cosine (first K frames concatenated):")
    off_s = cos_seq.clone()
    off_s.fill_diagonal_(0.0)
    vs = off_s[off_s > 0]
    print(f"  off-diagonal: n={vs.numel()} mean={vs.mean().item():.4f} "
          f"std={vs.std().item():.4f} min={vs.min().item():.4f} max={vs.max().item():.4f}")

    # Input-dependence test. Raw cosine is dominated by a large shared
    # component (healthy wav2vec2 shows ~0.97-0.99 pooled cosine even across
    # unrelated clips), so the collapse criterion is NOT the cosine magnitude
    # but whether the output is input-invariant:
    #   1) bit-identical (max abs element diff ~ float rounding), and
    #   2) pooled vectors differ only by noise (input-specific residual ~0).
    mean_vec = pooled.mean(dim=0, keepdim=True)
    p_norms = pooled.norm(dim=-1)
    rel = torch.zeros(n)
    for i in range(n):
        if p_norms[i] > 0:
            rel[i] = ((pooled[i] - mean_vec[0]).pow(2).sum() / pooled[i].pow(2).sum()).item()
    rel_mean = rel.mean().item()

    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append((pooled[i] - pooled[j]).abs().max().item())
    max_diff = max(diffs) if diffs else 0.0

    # Centered (mean-subtracted) residuals: structured values show real input
    # conditioning even when pooled cosine hugs ~1.0.
    c = pooled - mean_vec
    cnorms = c.norm(dim=-1)
    ceds = []
    for i in range(n):
        for j in range(i + 1, n):
            if cnorms[i] > 0 and cnorms[j] > 0:
                ceds.append(
                    (c[i] / cnorms[i]).dot(c[j] / cnorms[j]).item()
                )
    ced = torch.tensor(ceds) if ceds else torch.zeros(0)

    print(f"\noff-diagonal time-pooled cosine: mean={vp.mean().item():.4f} (fraction > 0.999: "
          f"{(vp > 0.999).float().mean().item():.3f})")
    print(f"max abs element diff between pooled encodings across clips: {max_diff:.3e}")
    print(f"input-specific variance fraction (residual/||p||^2): mean={rel_mean:.4f} "
          f"min={rel.min().item():.4f} max={rel.max().item():.4f}")
    if ced.numel():
        print(f"residual pairwise cosine: mean={ced.mean().item():+.3f} std={ced.std().item():.3f} "
              f"min={ced.min().item():+.3f} max={ced.max().item():+.3f}")

    if max_diff < 1e-4 and rel_mean < 1e-6:
        verdict = "COLLAPSE PERSISTS: encoder outputs are bit-identical / input-invariant."
    elif max_diff < 1e-4 or rel_mean < 1e-6:
        verdict = "COLLAPSE PERSISTS: input-specific component is negligible."
    else:
        verdict = (
            f"NO COLLAPSE: encoder output depends on input audio "
            f"(max elem diff {max_diff:.1e}, {rel_mean * 100:.2f}% of pooled variance "
            f"is input-specific)."
        )
    print(f"verdict: {verdict}")


def run_holdout_greedy(model, tokenizer, rows, device) -> None:
    print("\n=== Greedy-decode holdout table ===")
    refs, hyps = [], []
    for i, r in enumerate(rows):
        result = load_audio(r["audio_path"], target_sr=16000)
        audio = result.waveform.unsqueeze(0).to(device)
        lengths = torch.tensor([result.waveform.shape[1]], device=device)
        with torch.no_grad():
            ids = model.decode(audio, lengths, beam_size=1)[0]
        hyp = tokenizer.decode(ids)
        ref = r["text"]
        refs.append(ref)
        hyps.append(hyp)
        print(f"{r['id']:>6} | ref: {ref:<12} | hyp: {hyp:<12} | CER {cer(ref, hyp):.3f} | WER {wer(ref, hyp):.3f}")

    agg = compute_metrics(refs, hyps)
    print(f"\naggregate over {len(rows)} held-out clips: CER={agg.cer:.4f} WER={agg.wer:.4f}")


def _print_model_size(model: STTModel, checkpoint: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Model size (pre-shrink) ===")
    print(f"total params:   {total:,}  ({total * 4 / 1e6:.1f} MB fp32)")
    print(f"trainable:      {trainable:,}  ({trainable * 4 / 1e6:.1f} MB fp32)")
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir() and (ckpt_path / "model.pt").exists():
        size_mb = (ckpt_path / "model.pt").stat().st_size / 1e6
        print(f"model.pt on disk: {size_mb:.1f} MB")


def _build_model(checkpoint: str, device: torch.device, untrained: bool):
    manager = CheckpointManager(_checkpoint_root(checkpoint))
    name = _checkpoint_name(checkpoint)
    data = manager.load(name)
    tokenizer = data["tokenizer"]
    config = data["config"]
    if untrained:
        data["model_sd"] = None  # keep fresh pretrained backbone weights
    model = STTModel(config, vocab_size=tokenizer.vocab_size())
    if data["model_sd"] is not None:
        model.load_state_dict(data["model_sd"])
    model.to(device)
    model.eval()
    return model, tokenizer


def _checkpoint_root(checkpoint: str) -> str:
    name = Path(checkpoint).name
    if checkpoint == "latest" or name == "latest" or (name.startswith("v") and name[1:].isdigit()):
        parent = Path(checkpoint).parent
        return str(parent) if str(parent) != "." else "checkpoints"
    return str(Path(checkpoint).parent)


def _checkpoint_name(checkpoint: str) -> str:
    name = Path(checkpoint).name
    if checkpoint == "latest" or name == "latest" or (name.startswith("v") and name[1:].isdigit()):
        return name if checkpoint == "latest" else name
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Encoder-collapse probe + greedy-decode holdout evaluation"
    )
    parser.add_argument("--checkpoint", default="checkpoints/latest")
    parser.add_argument("--split", default="data/processed/split_manifest.json")
    parser.add_argument("--dataset", default="data/processed/manifest.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all held-out clips")
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Probe the fresh pretrained backbone (ignore trained weights).",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    rows = _load_held_out_rows(args.split, args.dataset)
    if args.max_clips > 0:
        rows = rows[: args.max_clips]

    model, tokenizer = _build_model(args.checkpoint, device, args.untrained)
    label = "untrained backbone" if args.untrained else "trained"
    print(f"Probing {len(rows)} held-out clips on {device} ({label})")

    cos_pooled, cos_seq, pooled, max_frames = run_probe(model, rows, device)
    print(f"(frame-sequence probe uses first {max_frames} frames per clip)")
    print_probe_report(cos_pooled, cos_seq, pooled, rows, label)
    if not args.untrained:
        run_holdout_greedy(model, tokenizer, rows, device)
    _print_model_size(model, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())