"""Post-training quantization / model optimization for deployment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.inference.transcriber import _checkpoint_name, _checkpoint_root
from src.models.stt import STTModel
from src.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def quantize_dynamic_checkpoint(
    checkpoint: str = "checkpoints/latest",
    output_dir: Optional[str] = None,
    dtype: torch.dtype = torch.qint8,
) -> Path:
    """Create a dynamically quantized (INT8) copy of a trained model.

    Uses PyTorch dynamic quantization, which quantizes weights at runtime
    without requiring calibration data. Best for CPU deployment.

    Args:
        checkpoint: Source checkpoint path/alias.
        output_dir: Where to write the quantized model. Defaults to
            <checkpoint>/quantized/.
        dtype: Quantization dtype (qint8 or quint8).

    Returns:
        Path to the quantized model file (model_quantized.pt).
    """
    root = Path(_checkpoint_root(checkpoint))
    name = _checkpoint_name(checkpoint)
    manager = CheckpointManager(str(root))
    data = manager.load(name)

    model = STTModel(data["config"], vocab_size=data["tokenizer"].vocab_size())
    model.load_state_dict(data["model_sd"])
    model.eval()

    # Dynamic quantization works well for Transformer/Linear-heavy models on CPU.
    # Embedding layers must use float_qparams (per-channel) weight-only config.
    qconfig_spec = {
        nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig
    }
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=dtype,
    )

    if output_dir is None:
        output_dir = str(Path(data["path"]) / "quantized")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(quantized.state_dict(), out / "model_quantized.pt")

    meta = {
        "source_checkpoint": str(data["path"]),
        "method": "dynamic_quantization",
        "dtype": str(dtype),
        "size_before_bytes": _model_file_size(data["path"]),
    }
    (out / "quantization_info.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    logger.info(
        "Dynamic quantized model written to %s (source size %s MB)",
        out / "model_quantized.pt",
        _model_file_size(data["path"]) / 1e6,
    )
    return out / "model_quantized.pt"


def load_quantized_model(
    checkpoint: str = "checkpoints/latest",
    device: torch.device = torch.device("cpu"),
) -> tuple[STTModel, object, object]:
    """Load a previously quantized model with its tokenizer/config.

    Not required for normal inference; the Transcriber loads full precision
    weights. This is a convenience for serving quantized models directly.
    """
    root = Path(_checkpoint_root(checkpoint))
    name = _checkpoint_name(checkpoint)
    manager = CheckpointManager(str(root))
    data = manager.load(name)

    q_path = Path(data["path"]) / "quantized" / "model_quantized.pt"
    if not q_path.exists():
        raise FileNotFoundError(f"No quantized model found for {checkpoint}")

    model = STTModel(data["config"], vocab_size=data["tokenizer"].vocab_size())
    model.load_state_dict(torch.load(q_path, map_location=device))
    model.to(device)
    model.eval()
    return model, data["tokenizer"], data["config"]


def _model_file_size(path) -> int:
    p = Path(path)
    return (p / "model.pt").stat().st_size if (p / "model.pt").exists() else 0