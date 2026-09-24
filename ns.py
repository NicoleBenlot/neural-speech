"""Universal neural-speech CLI. Works in PowerShell, cmd, and bash.

Usage:
    python ns.py prepare [--index data/raw/index.txt] [--output data/processed/manifest.csv]
    python ns.py validate [--fail-on-error]
    python ns.py train [--epochs 30] [--batch-size 8] [--device auto] ...
    python ns.py transcribe <audio> [--checkpoint checkpoints/latest]
    python ns.py optimize [--checkpoint checkpoints/latest]

Each subcommand forwards remaining arguments verbatim to the underlying
src module, so every flag from `python -m src.<module>` works unchanged.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Tuple

from src.data import prepare as prepare_mod
from src.data import validate as validate_mod
from src.deploy import optimize as optimize_mod
from src.device_info import main as devices_mod
from src.inference import mic as mic_mod
from src.inference import probe as probe_mod
from src.inference import transcriber as transcribe_mod
from src.training import train as train_mod

COMMANDS: Dict[str, tuple] = {
    "prepare": (prepare_mod.main, "index.txt -> data/processed/manifest.csv"),
    "validate": (validate_mod.main, "dataset integrity checks"),
    "train": (train_mod.main, "train the STT model"),
    "transcribe": (transcribe_mod.main, "transcribe an audio file"),
    "mic": (mic_mod.main, "record from mic and transcribe (real-time test)"),
    "optimize": (optimize_mod.main, "quantize + export a checkpoint"),
    "probe": (probe_mod.main, "encoder-collapse probe + greedy holdout eval"),
    "devices": (devices_mod, "detect available compute devices (CUDA/MPS/CPU)"),
}


def _invoke(module_main, argv: list) -> int:
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + argv
    try:
        return module_main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ns",
        description="Universal neural-speech CLI",
        add_help=False,
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="subcommand: " + ", ".join(sorted(COMMANDS)),
    )
    args, rest = parser.parse_known_args()

    wants_help = args.command in (None, "-h", "--help", "help") or any(
        flag in rest for flag in ("-h", "--help")
    )

    if wants_help:
        if args.command and args.command not in (None, "-h", "--help", "help"):
            module_main, _ = COMMANDS[args.command]
            return _invoke(module_main, rest)
        parser.print_help()
        print()
        print("Subcommands:")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:<11} {help_text}")
        return 0

    if args.command not in COMMANDS:
        parser.error(f"unknown command: {args.command}")

    module_main, _ = COMMANDS[args.command]
    return _invoke(module_main, rest)


if __name__ == "__main__":
    raise SystemExit(main())