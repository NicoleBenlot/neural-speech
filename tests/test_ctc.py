"""CTC head / CTC training-path tests (diagnostic replacement for the decoder)."""

import torch
import pytest

from src.models.ctc_head import CTCHead
from src.models.stt import STTConfig, STTModel
from src.tokens.tokenizer import BLANK, CharTokenizer
from src.training.checkpoint import CheckpointManager
from src.training.train import TrainConfig, Trainer, set_seed


def _tiny_ctc_model_config():
    return STTConfig(
        d_model=16,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        feat_channels=4,
        feat_layers=2,
        head="ctc",
    )


def test_tokenizer_blank_appended_at_end():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw", "adto"])
    tok.add_blank()
    assert tok.vocab_size() == 4 + 6 + 1  # specials + unique chars + blank
    assert tok.blank_id == tok.vocab_size() - 1
    assert tok.blank_id not in (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id)


def test_tokenizer_blank_roundtrip(tmp_path):
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    tok.add_blank()
    assert tok.decode([tok.blank_id, tok.char_to_id["a"], tok.blank_id, tok.blank_id, tok.char_to_id["d"]]) == "ad"

    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.blank_id == tok.blank_id
    assert loaded.vocab_size() == tok.vocab_size()


def test_ctc_head_log_softmax_shape():
    head = CTCHead(d_model=16, vocab_size=29, blank_id=28)
    out = head(torch.randn(3, 10, 16))
    assert out.shape == (3, 10, 29)
    sums = out.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_ctc_forward_shape():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    tok.add_blank()
    model = STTModel(
        _tiny_ctc_model_config(), vocab_size=tok.vocab_size()
    )
    model.eval()
    audio = torch.randn(2, 1, 8000)
    lengths = torch.tensor([8000, 4000])
    with torch.no_grad():
        log_probs = model.forward_ctc(audio, lengths)
    assert log_probs.shape == (2, 2000, tok.vocab_size())


def test_ctc_greedy_decode_collapses_and_drops_blanks():
    tok = CharTokenizer()
    tok.build_from_texts(["aba"])
    tok.add_blank()
    vocab = tok.vocab_size()
    blank = tok.blank_id
    model = STTModel(
        _tiny_ctc_model_config(), vocab_size=vocab
    )
    model.config.blank_token_id = blank
    model.eval()

    audio = torch.zeros(1, 1, 16000)  # encoder output T = 2000 frames
    T = 2000
    pattern = [blank, 4, 4, blank, 5, 4, blank, blank]  # 'a','a',blank,'b','a' -> 'aba'
    probs = torch.zeros(1, T, vocab)
    for t, token in enumerate(pattern):
        probs[0, t, token] = 1.0
    probs[0, len(pattern):, blank] = 1.0

    class StubHead(torch.nn.Module):
        def __init__(self, log_probs):
            super().__init__()
            self.blank_id = blank
            self._lp = log_probs

        def forward(self, encoder_out):
            return self._lp

    model.ctc_head = StubHead(probs.log())

    with torch.no_grad():
        ids = model.ctc_greedy_decode(audio, torch.tensor([16000]))
        dispatched = model.decode(audio, torch.tensor([16000]), beam_size=5)
    assert ids[0] == [4, 5, 4]
    assert tok.decode(ids[0]) == "aba"
    assert dispatched == ids


def test_ctc_dispatch_requires_head():
    model = STTModel(
        _tiny_ctc_model_config(), vocab_size=29
    )
    model.config.blank_token_id = 28
    with pytest.raises(RuntimeError, match="head='ctc'"):
        model.forward(torch.zeros(1, 1, 8000), torch.tensor([8000]),
                      torch.zeros(1, 3, dtype=torch.long), torch.tensor([3]))


def test_ctc_train_creates_checkpoint(tmp_path, tiny_manifest):
    config = {
        "dataset": str(tiny_manifest),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "split_dir": str(tmp_path / "processed"),
        "epochs": 1,
        "validation_frequency": 1,
        "batch_size": 2,
        "device": "cpu",
        "mixed_precision": False,
        "eval_manifest": None,
        "model": _tiny_ctc_model_config(),
    }
    set_seed(0)
    trainer = Trainer(TrainConfig(**config))
    trainer.train()

    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    data = manager.load("v001")
    assert data["config"].head == "ctc"
    assert data["config"].blank_token_id == data["tokenizer"].blank_id
    assert data["tokenizer"].vocab_size() == 4 + 10 + 1  # specials + uniq chars + blank
    assert (tmp_path / "checkpoints" / "v001" / "model.pt").exists()


def test_ctc_resume_continues(tmp_path, tiny_manifest):
    config = {
        "dataset": str(tiny_manifest),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "split_dir": str(tmp_path / "processed"),
        "epochs": 1,
        "validation_frequency": 1,
        "batch_size": 2,
        "device": "cpu",
        "mixed_precision": False,
        "eval_manifest": None,
        "model": _tiny_ctc_model_config(),
    }
    set_seed(1)
    Trainer(TrainConfig(**config)).train()

    resume_config = dict(config)
    resume_config["epochs"] = 2
    resume_config["resume"] = "v001"
    set_seed(1)
    Trainer(TrainConfig(**resume_config)).train()

    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    data = manager.load("v002")
    assert data["state"].epoch == 2
    assert data["config"].head == "ctc"