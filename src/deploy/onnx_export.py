"""ONNX export and optional TensorRT optimization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.models.stt import STTModel
from src.training.checkpoint import CheckpointManager
from src.inference.transcriber import _checkpoint_name, _checkpoint_root

logger = logging.getLogger(__name__)


class _ONNXWrapper(torch.nn.Module):
    """Wraps the STT model for ONNX export.

    Derives a full-valid token-length vector so the exported graph only needs
    (audio, lengths, tokens) as inputs, simplifying the inference host loop.
    """

    def __init__(self, model: STTModel):
        super().__init__()
        self.model = model

    def forward(self, audio, lengths, tokens):
        b, l = tokens.shape
        token_lengths = torch.full((b,), l, dtype=torch.long, device=tokens.device)
        return self.model(audio, lengths, tokens, token_lengths)


def export_onnx(
    checkpoint: str = "checkpoints/latest",
    output: Optional[str] = None,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> Path:
    """Export the STT model to a standalone ONNX graph.

    The exported graph accepts:
      - audio:   (B, 1, T) float32 waveform
      - lengths: (B,) int32 audio lengths
      - tokens:  (B, L) int32 input tokens (BOS prefix)

    and returns (B, L, V) logits. The autoregressive decoding loop still runs
    in the host and calls the graph one step at a time.

    Args:
        checkpoint: Source checkpoint path/alias.
        output: Output .onnx path. Defaults to <checkpoint>/onnx/model.onnx.
        opset: ONNX opset version.
        dynamic_axes: Export with dynamic sequence lengths.

    Returns:
        Path to the exported ONNX file.
    """
    root = Path(_checkpoint_root(checkpoint))
    name = _checkpoint_name(checkpoint)
    manager = CheckpointManager(str(root))
    data = manager.load(name)

    model = STTModel(data["config"], vocab_size=data["tokenizer"].vocab_size())
    model.load_state_dict(data["model_sd"])
    model.eval()

    wrapper = _ONNXWrapper(model).eval()

    b = 1
    t = 16000  # 1 second of audio for shape tracing
    l = 8
    dummy_audio = torch.randn(b, 1, t)
    dummy_lengths = torch.tensor([t], dtype=torch.long)
    dummy_tokens = torch.tensor([[data["tokenizer"].bos_id] * l], dtype=torch.long)

    if output is None:
        out_dir = Path(data["path"]) / "onnx"
    else:
        out_dir = Path(output).parent
        if str(out_dir) == ".":
            out_dir = Path(data["path"]) / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"

    dynamic_axes_map = None
    if dynamic_axes:
        dynamic_axes_map = {
            "audio": {0: "batch", 2: "audio_time"},
            "lengths": {0: "batch"},
            "tokens": {0: "batch", 1: "text_len"},
            "logits": {0: "batch", 1: "text_len"},
        }

    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_lengths, dummy_tokens),
        str(out_path),
        input_names=["audio", "lengths", "tokens"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes_map,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("ONNX exported to %s", out_path)

    (out_dir / "onnx_meta.json").write_text(
        f'{{"source_checkpoint": "{data["path"]}", "opset": {opset}, "dynamic_axes": {dynamic_axes}}}\n',
        encoding="utf-8",
    )
    return out_path


def check_tensorrt():
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def export_tensorrt(
    checkpoint: str = "checkpoints/latest",
    output: Optional[str] = None,
    precision: str = "fp16",
) -> Optional[Path]:
    """Convert an exported ONNX graph to a TensorRT engine (if available).

    Requires the `tensorrt` Python package and an NVIDIA GPU. Returns None
    when TensorRT is unavailable, so non-GPU environments can ignore it.

    Args:
        checkpoint: Source checkpoint path/alias.
        output: Output .engine path.
        precision: 'fp16' or 'int8'.

    Returns:
        Path to the TensorRT engine, or None if unavailable.
    """
    if not check_tensorrt():
        logger.warning("TensorRT is not installed; skipping TensorRT export.")
        return None

    onnx_path = export_onnx(checkpoint, output=None)

    import tensorrt as trt

    if output is None:
        output = str(onnx_path).replace(".onnx", f"_{precision}.engine")

    logger.info("Building TensorRT engine (precision=%s)...", precision)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(str(onnx_path))

    config = builder.create_builder_config()
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # NOTE: INT8 requires calibration on representative audio; supply a
        # calibrator before use in production.

    engine_bytes = builder.build_serialized_network(network, config)
    Path(output).write_bytes(engine_bytes)
    logger.info("TensorRT engine written to %s", output)
    return Path(output)