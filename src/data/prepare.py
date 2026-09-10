"""Preprocess raw index.txt into a normalized manifest.csv."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

from src.data.parser import IndexEntry, parse_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def write_manifest(entries: List[IndexEntry], manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "audio", "text", "section"])
        for entry in entries:
            writer.writerow([entry.id, entry.audio_path, entry.text, entry.section or ""])
    logger.info("Wrote manifest with %d entries to %s", len(entries), manifest_path)


def prepare(
    index_path: str,
    output: str,
    assets_dir: Optional[str] = None,
) -> int:
    index = Path(index_path)
    if assets_dir is None:
        assets = index.parent / "assets"
    else:
        assets = Path(assets_dir)

    result = parse_index(index, assets_dir=assets)
    if result.errors:
        logger.warning("Encountered %d parse errors:", result.error_count)
        for err in result.errors[:20]:
            logger.warning("  %s", err)

    write_manifest(result.entries, Path(output))
    print(f"Manifest written: {output} ({result.entry_count} entries)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Process raw index.txt into manifest.csv")
    parser.add_argument("--index", default="data/raw/index.txt", help="Path to index.txt")
    parser.add_argument("--assets", default=None, help="Directory containing audio assets")
    parser.add_argument("--output", default="data/processed/manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    raise SystemExit(prepare(args.index, args.output, args.assets))


if __name__ == "__main__":
    main()