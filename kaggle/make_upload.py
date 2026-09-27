"""Build Kaggle-ready upload bundles from this Windows working copy.

Kaggle runs Linux, where a backslash is an ordinary filename character rather
than a path separator. Everything this repo writes on Windows therefore records
``data\\fleurs_ceb_ph\\train\\00000.wav`` and ``checkpoints\\mms\\v004``, which
resolve to a single nonexistent filename there.

The local pipeline stays exactly as it is (it is what we test against); this
script instead emits *copies* with forward-slash paths, ready to upload as
Kaggle Datasets:

    python kaggle/make_upload.py --version v004
    python kaggle/make_upload.py --check kaggle/dist

Bundles land in ``--out`` (default ``kaggle/dist``):

    code.zip   the repo without data/, checkpoints/, .git, caches
    data.zip   data/fleurs_ceb_ph/ + data/raw/ + rewritten data/processed/*
    ckpt.zip   checkpoints/<line>/<version> + rewritten latest/best/protect

Rewritten (path fields only - ids, text, metrics and fingerprints are copied
byte-for-byte, so a split is never silently rebuilt with different rows):

    data/processed/manifest*.csv        column "audio"
    data/processed/split_*.json         every row's "audio_path"
    checkpoints/<line>/latest.json      "path"
    checkpoints/<line>/best.json        "path", "dataset", "replay_manifest"

The per-split ``manifest.tsv`` files need no rewrite: their ``filename`` column
is a bare ``00000.wav`` that the registry joins to the split dir at read time,
so a Kaggle-side rebuild produces forward slashes anyway. The rewritten CSVs are
given a fresh mtime, so ``ns.py datasets`` will not try to rebuild them.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

REPO = Path(__file__).resolve().parent.parent

CODE_INCLUDE = ("ns.py", "src", "requirements.txt", "commands.txt", "copy_repo.sh")
CODE_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data",
    "checkpoints",
    "kaggle",
    "voices",
    ".venv",
    "dist",
}
DATA_AUDIO_DIRS = ("data/fleurs_ceb_ph", "data/raw")
AUDIO_SUFFIXES = {".wav", ".opus", ".flac", ".ogg", ".mp3", ".m4a"}
# Sidecars that must ride along with the audio: the per-split FLEURS
# ``manifest.tsv`` (DatasetSpec.missing_sources requires it, even when the
# combined manifest CSV is present) and ``data/raw/index.txt``. Both store bare
# filenames / slash-free paths, so they need no rewrite.
DATA_SIDECAR_SUFFIXES = {".tsv", ".txt"}
CKPT_INDEXES = ("latest.json", "best.json", "protect.json")


def _posix(value: str) -> str:
    return str(value).replace("\\", "/")


# --------------------------------------------------------------------------- #
# rewrites
# --------------------------------------------------------------------------- #
def rewrite_manifest_csv(src: Path, dst: Path) -> int:
    """Copy a manifest CSV, normalizing the ``audio`` column."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with src.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        with dst.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                if "audio" in row:
                    row["audio"] = _posix(row["audio"])
                writer.writerow(row)
                rows += 1
    return rows


def rewrite_split_json(src: Path, dst: Path) -> int:
    """Copy a saved split, normalizing every row's ``audio_path``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = 0
    for key in ("train", "valid", "test"):
        for row in data.get(key, []):
            if "audio_path" in row:
                row["audio_path"] = _posix(row["audio_path"])
                rows += 1
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def rewrite_ckpt_index(src: Path, dst: Path) -> List[str]:
    """Copy latest.json / best.json / protect.json, normalizing path fields."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    touched: List[str] = []
    if isinstance(data, dict):
        for key in ("path", "dataset", "replay_manifest"):
            if isinstance(data.get(key), str):
                data[key] = _posix(data[key])
                touched.append(key)
        for block in ("val_loss", "cer"):
            if isinstance(data.get(block), dict) and isinstance(
                data[block].get("path"), str
            ):
                data[block]["path"] = _posix(data[block]["path"])
                touched.append(f"{block}.path")
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return touched


# --------------------------------------------------------------------------- #
# bundling
# --------------------------------------------------------------------------- #
def _add(zf: zipfile.ZipFile, src: Path, arc: str) -> None:
    """Add a file (or directory tree) to the zip, skipping junk."""
    if src.is_dir():
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            if any(part in CODE_EXCLUDE_DIRS for part in path.parts):
                continue
            if path.suffix == ".pyc":
                continue
            zf.write(path, arc.rstrip("/") + "/" + path.relative_to(src).as_posix())
    else:
        zf.write(src, arc)


def build_code_zip(out: Path) -> Tuple[Path, int]:
    dest = out / "code.zip"
    count = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in CODE_INCLUDE:
            src = REPO / name
            if not src.exists():
                print(f"  ! missing {name}, skipped")
                continue
            _add(zf, src, name)
            count += 1
    return dest, count


def build_data_zip(out: Path, include_audio: bool) -> Tuple[Path, int]:
    dest = out / "data.zip"
    files = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        if include_audio:
            for rel in DATA_AUDIO_DIRS:
                root = REPO / rel
                if not root.exists():
                    print(f"  ! missing {rel}, skipped (re-create the export)")
                    continue
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    suffix = path.suffix.lower()
                    if suffix not in AUDIO_SUFFIXES | DATA_SIDECAR_SUFFIXES:
                        continue
                    zf.write(path, path.relative_to(REPO).as_posix())
                    files += 1
        processed = REPO / "data" / "processed"
        if processed.exists():
            for csv_path in sorted(processed.glob("manifest*.csv")):
                rewritten = out / "_tmp" / csv_path.name
                rows = rewrite_manifest_csv(csv_path, rewritten)
                zf.write(rewritten, csv_path.relative_to(REPO).as_posix())
                files += 1
                print(f"  manifest {csv_path.name}: {rows} rows -> posix paths")
            for json_path in sorted(processed.glob("split_*.json")):
                rewritten = out / "_tmp" / json_path.name
                rows = rewrite_split_json(json_path, rewritten)
                zf.write(rewritten, json_path.relative_to(REPO).as_posix())
                files += 1
                print(f"  split    {json_path.name}: {rows} rows -> posix paths")
            shutil.rmtree(out / "_tmp", ignore_errors=True)
    return dest, files


def build_ckpt_zip(out: Path, line: str, version: str) -> Tuple[Path, int]:
    """Bundle exactly one version (1.55 GB) plus the rewritten line indexes."""
    dest = out / "ckpt.zip"
    src_line = REPO / line
    src_version = src_line / version
    if not src_version.is_dir():
        raise SystemExit(
            f"No such version: {src_version}\n"
            f"Available: {', '.join(p.name for p in sorted(src_line.glob('v*')))}"
        )

    files = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_version.iterdir()):
            if path.is_file():
                zf.write(path, f"{line}/{version}/{path.name}")
                files += 1
        for name in CKPT_INDEXES:
            src = src_line / name
            if not src.exists():
                continue
            rewritten = out / "_tmp" / name
            touched = rewrite_ckpt_index(src, rewritten)
            zf.write(rewritten, f"{line}/{name}")
            files += 1
            print(f"  {line}/{name}: rewrote {', '.join(touched) or 'nothing'}")
        shutil.rmtree(out / "_tmp", ignore_errors=True)
    return dest, files


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def check_bundle(bundle: Path, line: str) -> List[str]:
    """Re-open one archive and prove nothing Windows-shaped survived."""
    problems: List[str] = []
    kind = bundle.stem  # code | data | ckpt

    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        for name in names:
            if "\\" in name:
                problems.append(f"zip entry with backslash: {name}")

        if kind == "code" and not any(n.endswith(".py") for n in names):
            problems.append("no .py files in code.zip")
        if kind == "data" and not any(n.startswith("data/") for n in names):
            problems.append("no data/ entries in data.zip")
        if kind == "ckpt":
            if not any(n.startswith(f"{line}/v") for n in names):
                problems.append(f"no {line}/vNNN/ version dir in ckpt.zip")
            for index in ("latest.json", "best.json"):
                if not any(n.endswith(index) for n in names):
                    problems.append(f"no {index} in ckpt.zip")
            if not any(n.endswith("model.pt") for n in names):
                problems.append("no model.pt in ckpt.zip")

        for name in names:
            if not name.endswith((".csv", ".json")):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            if name.endswith(".csv"):
                for row in csv.DictReader(text.splitlines()):
                    audio = row.get("audio", "")
                    if "\\" in audio:
                        problems.append(f"{name}: backslash in audio {audio!r}")
                        break
            else:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ("train", "valid", "test"):
                        for row in data.get(key, []):
                            if "\\" in row.get("audio_path", ""):
                                problems.append(f"{name}: backslash in audio_path")
                                break
                    for key in ("path", "dataset", "replay_manifest"):
                        if isinstance(data.get(key), str) and "\\" in data[key]:
                            problems.append(f"{name}: backslash in {key} {data[key]!r}")
                    for block in ("val_loss", "cer"):
                        entry = data.get(block)
                        if isinstance(entry, dict) and isinstance(
                            entry.get("path"), str
                        ) and "\\" in entry["path"]:
                            problems.append(f"{name}: backslash in {block}.path")
    return problems


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "kaggle" / "dist"))
    ap.add_argument("--line", default="checkpoints/mms")
    ap.add_argument(
        "--version",
        default=None,
        help="Checkpoint version to continue from (e.g. v004). Required unless "
        "--check is used.",
    )
    ap.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip the 1.6 GB audio copy (rewrites manifests only - fast check).",
    )
    ap.add_argument(
        "--check", action="store_true", help="Only verify an existing bundle."
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.check:
        if not args.version:
            ap.error("--version is required (e.g. --version v004)")
        print("code.zip")
        dest, n = build_code_zip(out)
        print(f"  {dest.name}: {n} entries, {dest.stat().st_size / 2**20:.1f} MB")
        print("data.zip")
        dest, n = build_data_zip(out, include_audio=not args.no_audio)
        print(f"  {dest.name}: {n} files, {dest.stat().st_size / 2**20:.1f} MB")
        print("ckpt.zip")
        dest, n = build_ckpt_zip(out, args.line, args.version)
        print(f"  {dest.name}: {n} files, {dest.stat().st_size / 2**20:.1f} MB")

    print("\nverifying bundles")
    failed = False
    for name in ("code.zip", "data.zip", "ckpt.zip"):
        path = out / name
        if not path.exists():
            print(f"  {name}: absent, skipped")
            continue
        problems = check_bundle(path, args.line)
        if problems:
            failed = True
            print(f"  {name}: FAIL")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"  {name}: OK ({path.stat().st_size / 2**20:.1f} MB)")

    if failed:
        print("\nFAILED: Windows-style paths survived; do not upload.")
        return 1
    print("\nAll bundles are Linux-clean. Upload each as a Kaggle Dataset:")
    print("  code.zip -> /kaggle/input/neural-speech-code")
    print("  data.zip -> /kaggle/input/neural-speech-data")
    print("  ckpt.zip -> /kaggle/input/neural-speech-ckpt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
