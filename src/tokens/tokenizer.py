"""Character-level tokenizer with special tokens."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
BLANK = "<BLANK>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


@dataclass
class TokenizerConfig:
    kind: str = "character"
    pad_token: str = PAD
    unk_token: str = UNK
    bos_token: str = BOS
    eos_token: str = EOS
    blank_token: Optional[str] = None  # set to BLANK for CTC training


class BaseTokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
    def vocab_size(self) -> int: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "BaseTokenizer": ...


@dataclass
class CharTokenizer:
    """Character-level tokenizer with learnable vocabulary."""

    char_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_char: Dict[int, str] = field(default_factory=dict)
    config: TokenizerConfig = field(default_factory=TokenizerConfig)

    def __post_init__(self):
        if not self.char_to_id:
            self._init_vocab()

    def _init_vocab(self):
        idx = 0
        for token in SPECIAL_TOKENS:
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
            idx += 1

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.config.pad_token]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.config.unk_token]

    @property
    def bos_id(self) -> int:
        return self.char_to_id[self.config.bos_token]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.config.eos_token]

    @property
    def blank_id(self) -> Optional[int]:
        """CTC blank token id, or None when no blank token is configured."""
        if self.config.blank_token is None:
            return None
        return self.char_to_id.get(self.config.blank_token)

    def add_blank(self, token: str = BLANK):
        """Append the CTC blank token (at the end of the vocabulary)."""
        self.config.blank_token = token
        self.add_char(token)

    def add_char(self, char: str) -> int:
        if char in self.char_to_id:
            return self.char_to_id[char]
        idx = len(self.char_to_id)
        self.char_to_id[char] = idx
        self.id_to_char[idx] = char
        return idx

    def build_from_texts(self, texts: List[str]):
        for text in texts:
            for char in text:
                self.add_char(char)
        logger.info("Vocabulary built: %d tokens", len(self.char_to_id))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        chars = []
        for idx in ids:
            token = self.id_to_char.get(idx, self.config.unk_token)
            if skip_special and (
                token in SPECIAL_TOKENS or token == self.config.blank_token
            ):
                continue
            chars.append(token)
        return "".join(chars)

    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": {
                "kind": self.config.kind,
                "pad_token": self.config.pad_token,
                "unk_token": self.config.unk_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
                "blank_token": self.config.blank_token,
            },
            "char_to_id": self.char_to_id,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Tokenizer saved to %s (%d tokens)", path, self.vocab_size())

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        config = TokenizerConfig(**data["config"])
        tok = cls(config=config)
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {v: k for k, v in tok.char_to_id.items()}
        return tok


def create_tokenizer() -> CharTokenizer:
    """Create a fresh character tokenizer."""
    return CharTokenizer()
