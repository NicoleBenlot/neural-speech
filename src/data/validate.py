"""Validate the raw dataset and preprocessed manifest."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from src.data.parser import parse_index
from src.data.prepare import write_manifest
from src.data.audio import _decode_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".opus", ".ogg", ".wav", ".flac", ".mp3"}


@dataclass
class ValidationResult:
    total_entries: int = 0
    valid_entries: int = 0
    missing_audio: int = 0
    corrupt_audio: int = 0
    invalid_entries: int = 0
    duplicate_ids: int = 0
    duplicate_paths: int = 0
    empty_texts: int = 0
    unreadable_audio: int = 0
    encoding_errors: int = 0
    unexpected_files: int = 0
    errors: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return (
            self.missing_audio > 0
            or self.corrupt_audio > 0
            or self.invalid_entries > 0
            or self.duplicate_ids > 0
        )

    def summary(self) -> str:
        lines = [
            "Dataset validation",
            "------------------",
            f"Total entries:       {self.total_entries}",
            f"Valid entries:       {self.valid_entries}",
            f"Missing audio:       {self.missing_audio}",
            f"Corrupt audio:       {self.corrupt_audio}",
            f"Invalid entries:     {self.invalid_entries}",
            f"Duplicate IDs:       {self.duplicate_ids}",
            f"Empty texts:         {self.empty_texts}",
            f"Encoding errors:     {self.encoding_errors}",
        ]
        return "\n".join(lines)


def _extension_is_expected(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def _is_valid_text(text: str) -> bool:
    try:
        text.encode("utf-8")
        return not text.strip() == ""
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def _check_unexpected_files(assets_dir: Path, expected_ids: Set[int]) -> List[str]:
    if not assets_dir.exists():
        return []
    problems = []
    for p in assets_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in VALID_EXTENSIONS:
            problems.append(f"Unexpected file: {p.name}")
            continue
        if p.suffix.lower() == ".opus":
            stem = p.stem
            if not stem.isdigit():
                problems.append(f"Unexpected filename (non-numeric): {p.name}")
                continue
            if int(stem) not in expected_ids:
                problems.append(f"Orphan audio file: {p.name}")
    return problems


def validate(
    index_path: str,
    assets_dir: Optional[str] = None,
    fail_on_error: bool = False,
) -> ValidationResult:
    index = Path(index_path)
    result = ValidationResult()

    parsed = parse_index(index, assets_dir=Path(assets_dir) if assets_dir else None)
    result.total_entries = parsed.entry_count
    result.invalid_entries = parsed.error_count
    result.errors.extend(parsed.errors)

    seen_ids: Set[int] = set()
    seen_paths: Set[str] = set()
    expected_ids: Set[int] = set()

    for entry in parsed.entries:
        expected_ids.add(entry.id)
        has_issue = False

        if entry.id in seen_ids:
            result.duplicate_ids += 1
            result.errors.append(f"Duplicate ID: {entry.id}")
            has_issue = True
        seen_ids.add(entry.id)

        if entry.audio_path in seen_paths:
            result.duplicate_paths += 1
            has_issue = True
        seen_paths.add(entry.audio_path)

        if not _is_valid_text(entry.text):
            result.empty_texts += 1
            result.errors.append(f"Empty/invalid text for ID {entry.id}")
            has_issue = True

        audio_file = Path(entry.audio_path)
        if not audio_file.exists():
            result.missing_audio += 1
            result.errors.append(f"Missing audio: {audio_file}")
            has_issue = True

        if audio_file.exists() and not _extension_is_expected(audio_file):
            result.invalid_entries += 1
            result.errors.append(f"Unexpected extension: {audio_file}")
            has_issue = True

        if audio_file.exists():
            try:
                _decode_audio(str(audio_file))
            except Exception as exc:  # noqa: BLE001 - audio validation must not crash
                result.corrupt_audio += 1
                result.errors.append(f"Unreadable/corrupt audio {audio_file}: {exc}")
                has_issue = True

        if not has_issue:
            result.valid_entries += 1

    result.unexpected_files = len(_check_unexpected_files(index.parent / "assets", expected_ids))

    # Provide logging messages for the summary
    logger.info(result.summary())

    if fail_on_error and result.failed:
        logger.error("Validation failed with %d error(s)", len(result.errors))
        for err in result.errors[:30]:
            logger.error("  %s", err)
        raise SystemExit(1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate the raw dataset")
    parser.add_argument("--index", default="data/raw/index.txt", help="Path to index.txt")
    parser.add_argument("--assets", default=None, help="Directory containing audio assets")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero on errors")
    parser.add_argument("--write-manifest", default=None, help="Optionally write manifest CSV")
    args = parser.parse_args()

    result = validate(args.index, args.assets, fail_on_error=args.fail_on_error)

    print()
    print(result.summary())
    print()

    if args.write_manifest:
        parsed = parse_index(Path(args.index), assets_dir=Path(args.assets) if args.assets else None)
        write_manifest(parsed.entries, Path(args.write_manifest))

    return 0 if not result.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())