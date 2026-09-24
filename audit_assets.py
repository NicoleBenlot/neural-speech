"""Audit raw audio assets against index.txt and the prepared manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.data.audio import _decode_audio
from src.data.parser import parse_index


def read_manifest(path: Path) -> tuple[list[int], list[Path]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["id"]) for row in rows], [Path(row["audio"]) for row in rows]


def format_ids(ids: set[int] | list[int]) -> str:
    return ", ".join(str(value) for value in sorted(set(ids))) or "none"


def audit(index_path: Path, assets_dir: Path, manifest_path: Path, strict: bool) -> int:
    result = parse_index(index_path, assets_dir=assets_dir)
    indexed_ids = [entry.id for entry in result.entries]
    indexed_set = set(indexed_ids)

    asset_files = [path for path in assets_dir.iterdir() if path.is_file()] if assets_dir.exists() else []
    audio_files = [path for path in asset_files if path.suffix.lower() == ".opus" and path.stem.isdigit()]
    asset_ids = {int(path.stem) for path in audio_files}
    unrecognized_files = [path.name for path in asset_files if path not in audio_files]

    manifest_ids, manifest_paths = read_manifest(manifest_path)
    manifest_set = set(manifest_ids)
    unreadable_assets = []
    for path in audio_files:
        try:
            _decode_audio(str(path))
        except Exception as exc:  # noqa: BLE001 - report all bad asset files
            unreadable_assets.append(f"{path.name} ({exc})")

    missing_manifest_paths = [str(path) for path in manifest_paths if not path.exists()]

    missing_assets = indexed_set - asset_ids
    undetected_assets = asset_ids - indexed_set
    missing_from_manifest = indexed_set - manifest_set
    unexpected_in_manifest = manifest_set - indexed_set
    duplicate_index_ids = {value for value, count in Counter(indexed_ids).items() if count > 1}
    duplicate_manifest_ids = {value for value, count in Counter(manifest_ids).items() if count > 1}

    print(f"Asset files: {len(asset_files)} total, {len(audio_files)} numeric .opus")
    print(f"Index entries: {len(indexed_ids)} parsed, {len(indexed_set)} unique")
    print(f"Manifest entries: {len(manifest_ids)} rows, {len(manifest_set)} unique")
    print(f"Index parse errors: {result.error_count}")
    print(f"Missing assets for indexed IDs ({len(missing_assets)}): {format_ids(missing_assets)}")
    print(f"Undetected assets not in index ({len(undetected_assets)}): {format_ids(undetected_assets)}")
    print(f"Indexed IDs missing from manifest ({len(missing_from_manifest)}): {format_ids(missing_from_manifest)}")
    print(f"Unexpected manifest IDs ({len(unexpected_in_manifest)}): {format_ids(unexpected_in_manifest)}")
    print(f"Duplicate index IDs ({len(duplicate_index_ids)}): {format_ids(duplicate_index_ids)}")
    print(f"Duplicate manifest IDs ({len(duplicate_manifest_ids)}): {format_ids(duplicate_manifest_ids)}")
    print(f"Unreadable asset files ({len(unreadable_assets)}): {', '.join(unreadable_assets) or 'none'}")
    print(f"Missing manifest audio paths ({len(missing_manifest_paths)}): {', '.join(missing_manifest_paths) or 'none'}")
    print(f"Unrecognized asset files ({len(unrecognized_files)}): {', '.join(sorted(unrecognized_files)) or 'none'}")

    has_findings = any(
        (
            result.errors,
            missing_assets,
            undetected_assets,
            missing_from_manifest,
            unexpected_in_manifest,
            duplicate_index_ids,
            duplicate_manifest_ids,
            unreadable_assets,
            missing_manifest_paths,
            unrecognized_files,
        )
    )
    return 1 if strict and has_findings else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw assets, index.txt, and prepared manifest.csv")
    parser.add_argument("--index", type=Path, default=Path("data/raw/index.txt"))
    parser.add_argument("--assets", type=Path, default=Path("data/raw/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--strict", action="store_true", help="exit 1 when any discrepancy is found")
    args = parser.parse_args()
    return audit(args.index, args.assets, args.manifest, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())