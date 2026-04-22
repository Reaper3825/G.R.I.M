#!/usr/bin/env python3
"""
setup_mobileclip.py — Download Apple MobileCLIP-S0, export the image
encoder to ONNX, and bake the GRIM prompts file into a binary text-
embedding bank.

Outputs (overwrites if present):
  resources/models/vision/mobileclip_s0_image.onnx
  resources/models/vision/mobileclip_text_embeddings.bin

Inputs:
  resources/models/vision/mobileclip_prompts.txt    (edit this to change
                                                     the label set)

Binary format of the .bin file (little-endian):
  uint32 N       — number of prompts (rows)
  uint32 D       — embedding dim (must equal image-encoder output dim)
  float32[N*D]   — row-major, L2-normalised per row

Usage (from repo root):
  python scripts/setup_mobileclip.py
  python scripts/setup_mobileclip.py --variant s1   # if you want bigger

Requirements (install once into your venv):
  pip install torch open_clip_torch onnx
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[1]
VISION_DIR     = REPO_ROOT / "resources" / "models" / "vision"
PROMPTS_PATH   = VISION_DIR / "mobileclip_prompts.txt"
IMAGE_ONNX_OUT = VISION_DIR / "mobileclip_s0_image.onnx"
TEXT_EMB_OUT   = VISION_DIR / "mobileclip_text_embeddings.bin"


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return prompts


def export_image_encoder(model, image_size: int, out_path: Path) -> None:
    import torch

    class ImageEncoderWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # open_clip models expose .encode_image(); we want the raw
            # (un-normalised) projection. C++ side L2-normalises.
            return self.m.encode_image(x)

    wrap = ImageEncoderWrap(model).eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap,
        (dummy,),
        str(out_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # torch.onnx.export will sometimes emit large initializers as a
    # sidecar `<name>.onnx.data` file. OpenCV's cv::dnn::readNetFromONNX
    # historically does NOT follow that reference reliably across builds.
    # Re-load the model with external data and re-save it as a single
    # self-contained file. Also delete the sidecar so we never leave a
    # stale split-file artifact.
    import onnx

    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    model_proto = onnx.load(str(out_path), load_external_data=True)
    # Strip any external_data refs so onnx.save embeds the tensors inline.
    for init in model_proto.graph.initializer:
        if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
            init.data_location = onnx.TensorProto.DEFAULT
        del init.external_data[:]
    onnx.save(model_proto, str(out_path))
    if sidecar.exists():
        sidecar.unlink()
    print(f"  ✓ wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def encode_text_bank(model, tokenizer, prompts: list[str], out_path: Path) -> int:
    import torch

    with torch.no_grad():
        tokens = tokenizer(prompts)                       # [N, ctx_len]
        embeddings = model.encode_text(tokens)            # [N, D]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().float().contiguous()

    n, d = embeddings.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(embeddings.numpy().tobytes(order="C"))
    print(f"  ✓ wrote {out_path}  N={n}  D={d}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="s0",
                    choices=["s0", "s1", "s2", "b"],
                    help="MobileCLIP variant (s0 is smallest/fastest, b is best quality)")
    ap.add_argument("--image-size", type=int, default=256,
                    help="Override input resolution (S0 default = 256)")
    args = ap.parse_args()

    try:
        import open_clip
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: pip install torch open_clip_torch onnx", file=sys.stderr)
        return 1

    # open_clip naming: only S1/S2/B exist as "MobileCLIP-*" with the
    # 'datacompdr' pretrained tag. S0 only ships in the v2 family
    # ("MobileCLIP2-S0" / 'dfndr2b'). Map our short flag to whichever the
    # library actually supports.
    variant_map = {
        "s0": ("MobileCLIP2-S0", "dfndr2b"),
        "s1": ("MobileCLIP-S1",  "datacompdr"),
        "s2": ("MobileCLIP-S2",  "datacompdr"),
        "b":  ("MobileCLIP-B",   "datacompdr"),
    }
    model_id, pretrained_tag = variant_map[args.variant]

    print(f"[1/3] Loading {model_id} ({pretrained_tag}) via open_clip ...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_id, pretrained=pretrained_tag)
    tokenizer = open_clip.get_tokenizer(model_id)
    model.eval()

    print(f"[2/3] Exporting image encoder → {IMAGE_ONNX_OUT.name}")
    export_image_encoder(model, args.image_size, IMAGE_ONNX_OUT)

    prompts = load_prompts(PROMPTS_PATH)
    print(f"[3/3] Encoding {len(prompts)} prompts → {TEXT_EMB_OUT.name}")
    d_text = encode_text_bank(model, tokenizer, prompts, TEXT_EMB_OUT)

    print()
    print("Done.")
    print(f"  Image encoder: {IMAGE_ONNX_OUT}")
    print(f"  Text bank:     {TEXT_EMB_OUT}  (N={len(prompts)} D={d_text})")
    print()
    print("Next: rebuild GRIM (cmake --build build --config Debug) and run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
