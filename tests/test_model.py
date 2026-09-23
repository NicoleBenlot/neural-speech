"""Model forward pass and variable-length input tests."""

import torch
import pytest

from src.models.stt import STTConfig, STTModel
from src.tokens.tokenizer import CharTokenizer


def _make_model(vocab_size=32):
    config = STTConfig(
        d_model=32,
        nhead=2,
        encoder_layers=2,
        decoder_layers=2,
        dim_feedforward=64,
        feat_channels=8,
        feat_layers=2,
        bos_token_id=2,
        eos_token_id=3,
    )
    return STTModel(config, vocab_size=vocab_size)


def test_forward_pass():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw", "adto"])
    model = _make_model(vocab_size=tok.vocab_size())
    model.eval()

    audio = torch.randn(2, 1, 8000)
    lengths = torch.tensor([8000, 4000])
    tokens = torch.tensor([[tok.bos_id, tok.char_to_id["a"], tok.eos_id],
                           [tok.bos_id, tok.char_to_id["a"], tok.eos_id]])
    token_lengths = torch.tensor([3, 3])

    with torch.no_grad():
        logits = model(audio, lengths, tokens, token_lengths)

    assert logits.shape == (2, 3, tok.vocab_size())


def test_variable_length_audio():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    model = _make_model(vocab_size=tok.vocab_size())
    model.eval()

    audio = torch.randn(2, 1, 16000)
    lengths = torch.tensor([16000, 4000])

    tokens = torch.tensor([[tok.bos_id], [tok.bos_id]])
    token_lengths = torch.tensor([1, 1])

    with torch.no_grad():
        logits = model(audio, lengths, tokens, token_lengths)
    assert logits.shape[0] == 2


def test_greedy_decode_ends():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    model = _make_model(vocab_size=tok.vocab_size())
    model.eval()

    audio = torch.randn(1, 1, 4000)
    lengths = torch.tensor([4000])
    with torch.no_grad():
        ids = model.greedy_decode(
            audio, lengths, max_len=10, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id
        )
    assert isinstance(ids[0], list)
    assert ids[0][-1] == tok.eos_id


def test_beam_decode_matches_greedy_with_beam_1():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    model = _make_model(vocab_size=tok.vocab_size())
    model.eval()

    audio = torch.randn(1, 1, 4000)
    lengths = torch.tensor([4000])
    with torch.no_grad():
        greedy = model.greedy_decode(
            audio, lengths, max_len=10, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id
        )
        beamed = model.beam_decode(
            audio,
            lengths,
            max_len=10,
            beam_size=1,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
        )
    assert isinstance(beamed[0], list)
    assert greedy == beamed


def test_beam_decode_ends_with_eos():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    model = _make_model(vocab_size=tok.vocab_size())
    model.eval()

    audio = torch.randn(1, 1, 4000)
    lengths = torch.tensor([4000])
    with torch.no_grad():
        beamed = model.beam_decode(
            audio,
            lengths,
            max_len=10,
            beam_size=5,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
        )
    assert isinstance(beamed[0], list)
    assert beamed[0][-1] == tok.eos_id


def test_decode_dispatch_with_beam_1():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    model = _make_model(vocab_size=tok.vocab_size())
    model.eval()

    audio = torch.randn(1, 1, 4000)
    lengths = torch.tensor([4000])
    with torch.no_grad():
        ids = model.decode(
            audio,
            lengths,
            max_len=10,
            beam_size=3,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
        )
    assert isinstance(ids[0], list)
    assert -1 < ids[0][-1] < tok.vocab_size()