"""Reusable standalone inference transcriber."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from src.data.audio import load_audio
from src.models.stt import STTConfig, STTModel
from src.tokens.tokenizer import CharTokenizer
from src.training.checkpoint import (
    CheckpointManager,
)
from src.device_info import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints/mms/latest"


def _split_checkpoint(checkpoint: str) -> tuple[str, str]:
    """Split a checkpoint reference into (manager_root, checkpoint_name).

    Accepts: 'latest', 'v001', 'checkpoints/latest', 'dir/v001', or a path.
    """
    p = Path(checkpoint)
    name = p.name

    if checkpoint == "latest" or name == "latest":
        root = str(p.parent) if checkpoint != "latest" and str(p.parent) != "." else "checkpoints"
        return root, "latest"

    if name.startswith("v") and name[1:].isdigit():
        root = str(p.parent) if str(p.parent) != "." else "checkpoints"
        return root, name

    # A concrete path to a checkpoint directory
    return str(p.parent), str(p)


class Transcriber:
    """Transcribe audio files using a trained model checkpoint.

    Uses the same audio preprocessing, vocabulary, and model configuration
    as training (loaded from the checkpoint directory).
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "auto",
        max_length: int = 128,
        beam_size: int = 5,
        offline: Optional[bool] = None,
    ):
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        self.beam_size = beam_size
        self.offline = (
            os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
            if offline is None
            else offline
        )

        root, name = _split_checkpoint(checkpoint)
        self.manager = CheckpointManager(root)
        data = self.manager.load(name)

        self.tokenizer: CharTokenizer = data["tokenizer"]
        self.config: STTConfig = data["config"]

        self.model = STTModel(
            self.config,
            vocab_size=self.tokenizer.vocab_size(),
            device=self.device,
            local_files_only=self.offline,
        )
        self.model.load_state_dict(data["model_sd"])
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

        logger.info(
            "Transcriber loaded from %s (device=%s, vocab=%d)",
            checkpoint,
            self.device,
            self.tokenizer.vocab_size(),
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file to text."""
        try:
            result = load_audio(audio_path)
            waveform: torch.Tensor = result.waveform
        except Exception as exc:
            logger.error("Failed to load audio %s: %s", audio_path, exc)
            raise RuntimeError(f"Could not load audio: {exc}") from exc

        audio = waveform.unsqueeze(0).to(self.device)
        lengths = torch.tensor([waveform.shape[1]], device=self.device)

        with torch.no_grad():
            ids = self.model.decode(
                audio,
                lengths,
                max_len=self.max_length,
                beam_size=self.beam_size,
                bos_token_id=self.tokenizer.bos_id,
                eos_token_id=self.tokenizer.eos_id,
            )

        text = self.tokenizer.decode(ids[0])
        return text

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        return [self.transcribe(p) for p in audio_paths]


def _checkpoint_root(checkpoint: str) -> str:
    """Return the CheckpointManager root for the given checkpoint path."""
    root, _ = _split_checkpoint(checkpoint)
    return root


def _checkpoint_name(checkpoint: str) -> str:
    """Return the checkpoint name (version alias) for the given reference."""
    _, name = _split_checkpoint(checkpoint)
    return name


def main():
    parser = argparse.ArgumentParser(description="Standalone STT transcription")
    parser.add_argument("audio", help="Path to audio file (e.g. sample.opus)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint dir or alias")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Use only the local Hugging Face cache (also enabled by HF_HUB_OFFLINE=1)",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam width (1 = greedy)")
    args = parser.parse_args()

    t = Transcriber(
        checkpoint=args.checkpoint,
        device=args.device,
        beam_size=args.beam_size,
        offline=args.offline,
    )
    text = t.transcribe(args.audio)
    print(text)


if __name__ == "__main__":
    main()