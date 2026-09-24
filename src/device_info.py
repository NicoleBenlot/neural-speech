"""Compute-device detection and resolution.

Works on any machine (CUDA GPU / Apple MPS / CPU) so ``--device`` stays
portable: ``auto`` picks the best available accelerator, and requesting an
accelerator that is missing raises an actionable error instead of a runtime
CUDA crash.

Runnable as a subcommand: ``python ns.py devices`` (or ``python -m
src.device_info``) prints what this box has and what ``--device auto`` would
select.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import torch


def mps_available() -> bool:
    backend: Optional[object] = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def cuda_devices() -> List[dict]:
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except (RuntimeError, AssertionError):
            free = None
        cap = torch.cuda.get_device_capability(i)
        out.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_bytes": torch.cuda.get_device_properties(i).total_memory,
                "free_bytes": free,
                "capability": (int(cap[0]), int(cap[1])),
            }
        )
    return out


def auto_device() -> str:
    """Device string ``--device auto`` resolves to."""
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def summarize() -> List[str]:
    gb = 1024 ** 3
    lines = [
        f"torch {torch.__version__} "
        f"(cuda build: {torch.version.cuda or 'none'})",
    ]
    cudas = cuda_devices()
    if cudas:
        for d in cudas:
            free = (
                f"{d['free_bytes'] / gb:.1f} GB free" if d["free_bytes"] is not None else "free n/a"
            )
            lines.append(
                f"cuda:{d['index']} {d['name']} "
                f"(compute {d['capability'][0]}.{d['capability'][1]}, "
                f"{d['total_bytes'] / gb:.1f} GB total, {free})"
            )
    else:
        lines.append("cuda: not available")
    lines.append(f"mps: {'available' if mps_available() else 'not available'}")
    lines.append("cpu: available")
    lines.append(f"--device auto would select: {auto_device()}")
    return lines


def _unavailable(requested: str, device_type: str) -> ValueError:
    detail = "\n".join("  " + line for line in summarize())
    return ValueError(
        f"Requested device {requested!r} ({device_type}) is not available here.\n"
        f"Detected:\n{detail}\n"
        f"Use --device auto to pick automatically, or --device one of the "
        f"detected devices above."
    )


def resolve_device(device: str) -> torch.device:
    """Resolve ``device`` (as passed to the CLI) to a torch.device.

    ``auto`` picks the best accelerator. Requesting an accelerator that is not
    installed (e.g. ``--device cuda`` on a CPU-only box) raises a ValueError
    that lists what this machine actually has.
    """
    if not device or device == "auto":
        return torch.device(auto_device())

    req = device.lower()
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise _unavailable(device, "cuda")
        index = 0
        if ":" in req:
            index = int(req.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise _unavailable(device, f"cuda device index {index}")
    elif req.startswith("mps"):
        if not mps_available():
            raise _unavailable(device, "mps")
    elif req == "cpu":
        pass
    else:
        raise ValueError(f"Unsupported --device value: {device!r}")
    return torch.device(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ns devices",
        description="Detect available compute devices (CUDA / MPS / CPU).",
    )
    parser.add_argument(
        "--check",
        default="auto",
        metavar="DEVICE",
        help="Resolve a device string like --device would and print the result "
        "(e.g. --check cuda or --check cpu).",
    )
    args = parser.parse_args()

    print("\n".join(summarize()))
    try:
        resolved = resolve_device(args.check)
    except ValueError as exc:
        print(f"\n--check {args.check}: {exc}")
        return 2
    print(f"\n--check {args.check} resolves to: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())