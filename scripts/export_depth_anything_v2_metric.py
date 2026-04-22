"""Export Depth-Anything-V2-Metric (Indoor / Small) to ONNX.

Usage (from repo root, with .venv active):

    python scripts/export_depth_anything_v2_metric.py \
        --hf-model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
        --output  resources/models/vision/depth_anything_v2_metric_indoor_small.onnx \
        --size    518

Notes
-----
* Input must be a multiple of 14 (DINOv2 ViT patch size). 518 = 14 * 37.
* Exported with fixed input shape so OpenCV's cv::dnn ONNX importer is happy.
* Output: depth in METRES (post-process built into the model head).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForDepthEstimation


class DepthOnly(torch.nn.Module):
    """Wrap the HF model so the ONNX graph has a single tensor output."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values=pixel_values)
        # transformers depth-estimation models return either `predicted_depth`
        # (B, H, W) on relative models or the same on metric models.
        depth = out.predicted_depth
        if depth.dim() == 3:
            # Add an explicit channel dim → [B, 1, H, W] so cv::dnn unpacks
            # into rank-4 [1,1,H,W] which the loader already handles.
            depth = depth.unsqueeze(1)
        return depth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True,
                    help="Hugging Face model ID")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output ONNX path")
    ap.add_argument("--size", type=int, default=518,
                    help="Square input edge in pixels (multiple of 14)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if args.size % 14 != 0:
        raise SystemExit(
            f"--size must be a multiple of 14 (DINOv2 patch size); got {args.size}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading {args.hf_model} …")
    base = AutoModelForDepthEstimation.from_pretrained(args.hf_model)
    base.eval()
    wrapped = DepthOnly(base).eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    print(f"[export] Tracing → ONNX (opset={args.opset}, input={tuple(dummy.shape)}) …")

    torch.onnx.export(
        wrapped,
        (dummy,),
        args.output.as_posix(),
        input_names=["pixel_values"],
        output_names=["depth"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,            # legacy exporter is more cv::dnn-compatible
    )

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"[export] Wrote {args.output}  ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
