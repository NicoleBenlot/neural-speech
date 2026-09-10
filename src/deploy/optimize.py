"""One-shot deployment optimization entry point.

python -m src.deploy.optimize --checkpoint checkpoints/latest
"""

from __future__ import annotations

import argparse
import logging

from src.deploy.quantize import quantize_dynamic_checkpoint
from src.deploy.onnx_export import export_onnx, export_tensorrt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Optimize a checkpoint for deployment")
    parser.add_argument("--checkpoint", default="checkpoints/latest")
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--tensorrt", action="store_true", help="Also build a TensorRT engine (GPU only)")
    parser.add_argument("--tensorrt-precision", default="fp16", choices=["fp16", "int8"])
    parser.add_argument("--onnx-opset", type=int, default=17)
    args = parser.parse_args()

    if not args.skip_quantization:
        print("\n[1/2] Dynamic quantization...")
        quantize_dynamic_checkpoint(args.checkpoint)

    if not args.skip_onnx:
        print("\n[2/2] ONNX export...")
        export_onnx(args.checkpoint, opset=args.onnx_opset)

    if args.tensorrt:
        print("\n[optional] TensorRT export...")
        result = export_tensorrt(args.checkpoint, precision=args.tensorrt_precision)
        if result is None:
            print("TensorRT skipped (not available on this machine).")
        else:
            print(f"TensorRT engine: {result}")


if __name__ == "__main__":
    main()