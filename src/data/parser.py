"""Parser for the raw index.txt format mapping transcripts (words or sentences) to audio IDs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^\[(\w+)\]\s*$")
_ENTRY_RE = re.compile(r"^(.+?)\s*=\s*(\d+)\s*$")


@dataclass(frozen=True)
class IndexEntry:
    id: int
    audio_path: str
    text: str
    section: Optional[str] = None


@dataclass
class ParseResult:
    entries: List[IndexEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    line_count: int = 0
    entry_count: int = 0
    error_count: int = 0


def parse_index(index_path: Path, assets_dir: Optional[Path] = None) -> ParseResult:
    """Parse index.txt and return structured entries.

    Args:
        index_path: Path to index.txt.
        assets_dir: Directory containing audio files. Defaults to parent/assets.

    Returns:
        ParseResult with entries, errors, and counts.
    """
    if assets_dir is None:
        assets_dir = index_path.parent / "assets"

    result = ParseResult()
    current_section: Optional[str] = None

    if not index_path.exists():
        result.errors.append(f"Index file not found: {index_path}")
        result.error_count = len(result.errors)
        return result

    raw_lines = index_path.read_text(encoding="utf-8").splitlines()
    result.line_count = len(raw_lines)

    for line_num, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        section_match = _SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        entry_match = _ENTRY_RE.match(line)
        if entry_match:
            text = entry_match.group(1)
            audio_id = int(entry_match.group(2))
            audio_path = str(assets_dir / f"{audio_id}.opus")

            result.entries.append(
                IndexEntry(
                    id=audio_id,
                    audio_path=audio_path,
                    text=text,
                    section=current_section,
                )
            )
            result.entry_count += 1
        else:
            result.errors.append(f"Line {line_num}: invalid format: {raw_line}")

    result.error_count = len(result.errors)
    logger.info(
        "Parsed %d entries (%d errors) from %s",
        result.entry_count,
        result.error_count,
        index_path,
    )
    return result
