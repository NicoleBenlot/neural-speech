"""Record microphone audio and transcribe it in the terminal.

Usage:
    python ns.py mic --seconds 5
    python ns.py mic --loop --seconds 3
    python ns.py mic --list-devices --checkpoint checkpoints/v005
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - depends on local environment
    sd = None

from src.inference.transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def record_clip(seconds: float, samplerate: int = SAMPLE_RATE):
    """Record `seconds` of mono float32 audio from the default microphone."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed. Run: python -m pip install sounddevice"
        )
    _check_input_device()
    logger.info(
        "Recording %ss from %s ...", seconds, sd.default.device or "default device"
    )
    audio = sd.rec(
        int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    return audio.reshape(-1)


def _check_input_device():
    """Raise a friendly error when no microphone is available."""
    try:
        devices = sd.query_devices()
    except Exception:
        return
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError(
            "No microphone detected. Connect one or enable mic access: "
            "Settings > Privacy & security > Microphone. "
            "Run 'python ns.py mic --list-devices' to see available devices."
        )


def _transcribe_clip(transcriber: Transcriber, seconds: float) -> str:
    import soundfile as sf

    audio = record_clip(seconds)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name
    try:
        return transcriber.transcribe(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record from the microphone and transcribe with a checkpoint"
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording length in seconds")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep recording and transcribing until Ctrl+C",
    )
    parser.add_argument("--checkpoint", default="checkpoints/latest", help="Checkpoint dir or alias")
    parser.add_argument("--device", default="auto", help="Torch device (auto, cpu, cuda)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        if sd is None:
            parser.error("sounddevice is not installed. Run: python -m pip install sounddevice")
        print(sd.query_devices())
        return 0

    if sd is None:
        parser.error("sounddevice is not installed. Run: python -m pip install sounddevice")

    try:
        _check_input_device()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1

    transcriber = Transcriber(checkpoint=args.checkpoint, device=args.device)

    if not args.loop:
        try:
            print(_transcribe_clip(transcriber, args.seconds))
        except RuntimeError as exc:
            print(f"error: {exc}")
            return 1
        return 0

    try:
        while True:
            try:
                text = _transcribe_clip(transcriber, args.seconds)
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"> {text}")
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception:
        logger.exception("Mic transcription failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())