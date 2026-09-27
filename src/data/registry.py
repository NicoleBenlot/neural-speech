"""Named datasets and preset-split resolution for ``--data``.

The trainer historically took a single ``--dataset`` CSV and cut it at random
(80/10/10). Exports that ship their own official partition - FLEURS writes
``train/``, ``validation/`` and ``test/`` with one ``manifest.tsv`` each - must
keep that partition, otherwise reported numbers are not comparable to the
published ones. A named dataset therefore resolves to two things:

* a combined ``manifest.csv`` (the split name lands in its ``section``
  column), so the trainer, the manifest fingerprint, split reuse, and the
  checkpoint manifest keep working exactly as before, and
* an optional preset split keyed by that section, so the official rows are
  used instead of a random cut.

Usage::

    python ns.py train --data default          # data/processed/manifest.csv
    python ns.py train --data fleurs_ceb_ph    # data/fleurs_ceb_ph, official splits
    python ns.py datasets                      # list + status
    python ns.py datasets --data fleurs_ceb_ph --build
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.dataset import ManifestRow
from src.data.parser import IndexEntry
from src.data.prepare import write_manifest

logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
PIPELINE_SPLITS: Tuple[str, ...] = (TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT)

# FLEURS names its second split "validation"; the pipeline calls it "valid".
FLEURS_SPLIT_DIRS: Dict[str, str] = {
    TRAIN_SPLIT: "train",
    VALID_SPLIT: "validation",
    TEST_SPLIT: "test",
}
TSV_NAME = "manifest.tsv"
TEXT_FIELDS = ("transcript", "raw_transcript")

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_INDEX = Path("data/raw/index.txt")
DEFAULT_MANIFEST = Path("data/processed/manifest.csv")


def read_split_tsv(
    tsv_path: Path,
    split_dir: Path,
    split: str,
    text_field: str = "transcript",
    id_offset: int = 0,
) -> List[ManifestRow]:
    """Parse one export ``manifest.tsv`` into manifest rows.

    Plain tab splitting, deliberately not :mod:`csv`: FLEURS text is written
    unquoted and some transcripts contain ``"`` and trailing tabs, which makes
    the csv module merge columns. Ids are offset per split so they stay unique
    across the combined manifest (each split numbers its clips from 00000).
    """
    if text_field not in TEXT_FIELDS:
        raise ValueError(
            f"text_field must be one of {TEXT_FIELDS}, got {text_field!r}"
        )

    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    header = [h.strip() for h in lines[0].split("\t")]
    for required in ("filename", text_field):
        if required not in header:
            raise ValueError(
                f"{tsv_path}: missing {required!r} column in header {header}"
            )
    name_col = header.index("filename")
    text_col = header.index(text_field)

    rows: List[ManifestRow] = []
    skipped = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) <= max(name_col, text_col):
            skipped += 1
            continue
        text = " ".join(fields[text_col].split())
        if not text:
            skipped += 1
            continue
        rows.append(
            ManifestRow(
                id=id_offset + len(rows),
                audio_path=str(split_dir / fields[name_col].strip()),
                text=text,
                section=split,
            )
        )

    if skipped:
        logger.warning(
            "%s: skipped %d row(s) with a missing filename/text field",
            tsv_path,
            skipped,
        )
    logger.info(
        "%s: parsed %d rows (split=%s, text_field=%s)",
        tsv_path,
        len(rows),
        split,
        text_field,
    )
    return rows


@dataclass(frozen=True)
class DatasetSpec:
    """A resolvable training dataset: one manifest plus optional fixed splits."""

    name: str
    description: str
    manifest: Path
    split_dirs: Dict[str, Path] = field(default_factory=dict)
    index: Optional[Path] = None
    prepare_hint: Optional[str] = None

    @property
    def has_preset_splits(self) -> bool:
        return bool(self.split_dirs)

    @property
    def split_names(self) -> Tuple[str, ...]:
        return tuple(s for s in PIPELINE_SPLITS if s in self.split_dirs)

    def missing_sources(self) -> List[str]:
        """Split dirs (or the manifest, for single-CSV datasets) that are absent."""
        if self.split_dirs:
            return [
                str(self.split_dirs[s])
                for s in self.split_names
                if not (self.split_dirs[s] / TSV_NAME).is_file()
            ]
        return [] if self.manifest.is_file() else [str(self.manifest)]

    def read_splits(self, text_field: str = "transcript") -> Dict[str, List[ManifestRow]]:
        """Official train/valid/test rows, refreshing the combined manifest first."""
        splits = self._parse_splits(text_field)
        self._refresh_manifest(splits)
        return splits

    def build(self, text_field: str = "transcript") -> Dict[str, List[ManifestRow]]:
        """Parse the export splits and always rewrite the combined manifest."""
        splits = self._parse_splits(text_field)
        self._write_manifest(splits)
        return splits

    def _parse_splits(self, text_field: str) -> Dict[str, List[ManifestRow]]:
        if not self.split_dirs:
            raise ValueError(f"Dataset {self.name!r} has no preset splits")

        missing = self.missing_sources()
        if missing:
            raise FileNotFoundError(
                f"Dataset {self.name!r} is incomplete; missing: " + ", ".join(missing)
            )

        splits: Dict[str, List[ManifestRow]] = {}
        offset = 0
        for split in self.split_names:
            rows = read_split_tsv(
                self.split_dirs[split] / TSV_NAME,
                self.split_dirs[split],
                split,
                text_field=text_field,
                id_offset=offset,
            )
            splits[split] = rows
            offset += len(rows)
        return splits

    def _refresh_manifest(self, splits: Dict[str, List[ManifestRow]]) -> bool:
        """Rewrite the combined manifest when it is missing or older than a source.

        Leaving an up-to-date manifest untouched keeps its SHA-256 fingerprint
        stable, which is what makes resume/continue reuse the saved split
        instead of re-splitting.
        """
        sources = [self.split_dirs[s] / TSV_NAME for s in self.split_names]
        stale = (
            not self.manifest.is_file()
            or any(s.stat().st_mtime > self.manifest.stat().st_mtime for s in sources)
        )
        if not stale:
            return False
        self._write_manifest(splits)
        return True

    def _write_manifest(self, splits: Dict[str, List[ManifestRow]]):

        entries: List[IndexEntry] = []
        for split in self.split_names:
            entries.extend(
                IndexEntry(
                    id=row.id,
                    audio_path=row.audio_path,
                    text=row.text,
                    section=row.section or split,
                )
                for row in splits[split]
            )
        write_manifest(entries, self.manifest)
        return True

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "manifest": str(self.manifest),
            "manifest_exists": self.manifest.is_file(),
            "preset_splits": list(self.split_names),
            "split_dirs": {s: str(self.split_dirs[s]) for s in self.split_names},
            "index": str(self.index) if self.index else None,
            "missing": self.missing_sources(),
        }


def _builtin_datasets(
    data_root: Path = DEFAULT_DATA_ROOT, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> Dict[str, DatasetSpec]:
    """Registry of the datasets this repo ships wiring for."""
    fleurs_root = data_root / "fleurs_ceb_ph"
    return {
        "default": DatasetSpec(
            name="default",
            description="index.txt -> data/processed/manifest.csv (random 80/10/10 split)",
            manifest=processed_dir / "manifest.csv",
            index=data_root / "raw" / "index.txt",
            prepare_hint="python ns.py prepare",
        ),
        "fleurs_ceb_ph": DatasetSpec(
            name="fleurs_ceb_ph",
            description="FLEURS Cebuano (ceb_ph), official train/validation/test splits",
            manifest=processed_dir / "manifest_fleurs_ceb_ph.csv",
            split_dirs={
                split: fleurs_root / dirname
                for split, dirname in FLEURS_SPLIT_DIRS.items()
            },
        ),
    }


def detect_preset_splits(root: Path) -> Dict[str, Path]:
    """Map pipeline split -> export dir for a directory that ships its own split."""
    return {
        split: root / dirname
        for split, dirname in FLEURS_SPLIT_DIRS.items()
        if (root / dirname / TSV_NAME).is_file()
    }


def resolve_dataset(
    value: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    processed_dir: Optional[Path] = None,
) -> DatasetSpec:
    """Resolve ``--data`` to a DatasetSpec.

    Accepts a registered name, a directory that ships preset splits, or a plain
    manifest file (CSV/TSV) that the trainer splits randomly.
    """
    processed = Path(processed_dir) if processed_dir else data_root / "processed"
    builtins = _builtin_datasets(Path(data_root), processed)

    if value in builtins:
        return builtins[value]

    path = Path(value)
    if path.is_dir():
        split_dirs = detect_preset_splits(path)
        if split_dirs:
            return DatasetSpec(
                name=path.name,
                description=f"preset splits from {path}",
                manifest=processed / f"manifest_{path.name}.csv",
                split_dirs=split_dirs,
            )
        raise FileNotFoundError(
            f"{path} has no {'/'.join(FLEURS_SPLIT_DIRS.values())} split dirs "
            f"holding {TSV_NAME}"
        )

    if path.is_file():
        return DatasetSpec(
            name=path.stem,
            description=f"single manifest {path}",
            manifest=path,
        )

    raise FileNotFoundError(
        f"Unknown dataset {value!r}. Known names: {', '.join(sorted(builtins))} "
        "(or pass a manifest/dataset directory path)"
    )


def _print_datasets(data_root: Path, processed_dir: Path) -> None:
    specs = _builtin_datasets(data_root, processed_dir)
    for name, spec in specs.items():
        info = spec.describe()
        print(f"{name}")
        print(f"  {info['description']}")
        print(f"  manifest: {info['manifest']}"
              f"{'' if info['manifest_exists'] else '  (missing)'}")
        if info["preset_splits"]:
            print(f"  preset splits: {', '.join(info['preset_splits'])}")
            for split, directory in info["split_dirs"].items():
                marker = "ok" if Path(directory, TSV_NAME).is_file() else "MISSING"
                print(f"    {split:<6} {directory} [{marker}]")
        elif info["index"]:
            print(f"  index:    {info['index']}")
            print("  splits:   random 80/10/10 at train time")
        if info["missing"]:
            hint = spec.prepare_hint or "export the dataset first"
            print(f"  -> run `{hint}`")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ns datasets",
        description="List training datasets or resolve one for `--data`",
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="NAME_OR_PATH",
        help="Registered name (default, fleurs_ceb_ph), dataset directory, "
        "or manifest CSV/TSV path. Omit to list the registry.",
    )
    parser.add_argument(
        "--text-field",
        default="transcript",
        choices=list(TEXT_FIELDS),
        help="Which export column becomes the target text. Default 'transcript' "
        "(the normalized column).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force a rebuild of the combined manifest from the export splits.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of text."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.data is None:
        if args.json:
            specs = _builtin_datasets(DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_DIR)
            print(json.dumps([s.describe() for s in specs.values()], indent=2))
        else:
            _print_datasets(DEFAULT_DATA_ROOT, DEFAULT_PROCESSED_DIR)
        return 0

    try:
        spec = resolve_dataset(args.data)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    missing = spec.missing_sources()
    if missing:
        hint = spec.prepare_hint or "export the dataset first"
        print(f"error: dataset {spec.name!r} is incomplete; missing: {', '.join(missing)}")
        print(f"hint:  {hint}")
        return 1

    if not spec.has_preset_splits:
        info = spec.describe()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"{spec.name}: {spec.description}")
            print(f"  manifest: {spec.manifest}")
            print("  splits:   random 80/10/10 at train time")
            print(f"  train with: python ns.py train --dataset {spec.manifest}")
        return 0

    splits = (
        spec.build(text_field=args.text_field)
        if args.build
        else spec.read_splits(text_field=args.text_field)
    )

    if args.json:
        print(
            json.dumps(
                {
                    **spec.describe(),
                    "text_field": args.text_field,
                    "rows": {k: len(v) for k, v in splits.items()},
                },
                indent=2,
            )
        )
        return 0

    print(f"{spec.name}: {spec.description}")
    print(f"  manifest: {spec.manifest}")
    print(f"  text field: {args.text_field}")
    total = 0
    for split in spec.split_names:
        print(f"  {split:<6} {len(splits[split])} rows")
        total += len(splits[split])
    print(f"  total   {total} rows")
    print(f"  train with: python ns.py train --data {spec.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
