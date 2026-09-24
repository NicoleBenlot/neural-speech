"""Device detection and --device resolution (portable across CUDA/MPS/CPU)."""

import pytest
import torch

from src import device_info
from src.device_info import auto_device, mps_available, resolve_device, summarize


def test_auto_device_is_some_known_backend():
    assert auto_device() in {"cuda", "mps", "cpu"}
    assert torch.device(resolve_device("auto")) == torch.device(auto_device())


def test_resolve_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")
    assert resolve_device("cuda") == torch.device("cuda")
    assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_resolve_cuda_raises_when_unavailable(monkeypatch):
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError) as excinfo:
        resolve_device("cuda")
    msg = str(excinfo.value)
    assert "not available here" in msg
    assert "cuda" in msg


def test_resolve_cuda_index_out_of_range_raises(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError):
        resolve_device("cuda:7")


def test_resolve_mps_raises_when_unavailable():
    if mps_available():
        pytest.skip("MPS is available; nothing to assert")
    with pytest.raises(ValueError) as excinfo:
        resolve_device("mps")
    assert "not available here" in str(excinfo.value)


def test_resolve_unknown_device_raises():
    with pytest.raises(ValueError) as excinfo:
        resolve_device("rocm:0")
    assert "Unsupported" in str(excinfo.value)


def test_summarize_lists_detected_devices():
    lines = summarize()
    assert any("cpu: available" in line for line in lines)
    assert any("cuda" in line for line in lines)
    assert any("auto would select" in line for line in lines)


def test_cuda_devices_reports_current_platform():
    cudas = device_info.cuda_devices()
    if torch.cuda.is_available():
        assert cudas and cudas[0]["index"] == 0
        assert cudas[0]["name"]
    else:
        assert cudas == []


def test_resolve_device_re_exported_from_training():
    from src.training.train import resolve_device as reexported

    assert reexported is resolve_device