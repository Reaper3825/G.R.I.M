#!/usr/bin/env python3
"""Validate physical-perception ONNX model files declared in ai_config.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "ai_config.json"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing model file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"empty model file: {path}")


def resolve_config_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def assert_onnx_io(path: Path, expected_inputs: list[str], expected_outputs: list[str]) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    got_inputs = [item.name for item in model.graph.input]
    got_outputs = [item.name for item in model.graph.output]
    missing_inputs = [name for name in expected_inputs if name not in got_inputs]
    missing_outputs = [name for name in expected_outputs if name not in got_outputs]
    if missing_inputs or missing_outputs:
        raise RuntimeError(
            f"{path.name} has incompatible ONNX IO names; "
            f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, "
            f"actual_inputs={got_inputs}, actual_outputs={got_outputs}"
        )


def main() -> int:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    sub_models = cfg["mmo"]["sub_models"]
    checked: list[Path] = []
    for model in sub_models:
        if model.get("kind") != "vision":
            continue
        model_path = model.get("model_path", "")
        if not model_path:
            raise RuntimeError(f"vision model {model.get('id')} has empty model_path")
        paths = [resolve_config_path(model_path)]
        vision = model.get("vision", {})
        for key in (
            "class_names_path",
            "text_embeddings_path",
            "recogniser_onnx_path",
            "recogniser_charset_path",
            "expression_classifier_onnx_path",
            "expression_classifier_class_names_path",
            "instance_seg_decoder_onnx_path",
        ):
            if value := vision.get(key, ""):
                paths.append(resolve_config_path(value))

        for path in paths:
            require_file(path)
            checked.append(path)

    instance_model = next(
        model for model in sub_models
        if model.get("kind") == "vision"
        and model.get("vision", {}).get("operator") == "instance_segmenter"
    )
    instance_vision = instance_model["vision"]
    vision_dir = REPO_ROOT / "resources" / "models" / "vision"
    assert_onnx_io(
        vision_dir / "sam2_hiera_tiny_encoder.onnx",
        [instance_vision.get("instance_seg_encoder_input_name", "image")],
        [
            instance_vision.get("instance_seg_encoder_output_image_embed_name", "image_embed"),
            instance_vision.get("instance_seg_encoder_output_high_res_feats_0_name", "high_res_feats_0"),
            instance_vision.get("instance_seg_encoder_output_high_res_feats_1_name", "high_res_feats_1"),
        ],
    )
    assert_onnx_io(
        vision_dir / "sam2_hiera_tiny_decoder.onnx",
        [
            instance_vision.get("instance_seg_decoder_input_image_embed_name", "image_embed"),
            instance_vision.get("instance_seg_decoder_input_high_res_feats_0_name", "high_res_feats_0"),
            instance_vision.get("instance_seg_decoder_input_high_res_feats_1_name", "high_res_feats_1"),
            instance_vision.get("instance_seg_decoder_input_point_coords_name", "point_coords"),
            instance_vision.get("instance_seg_decoder_input_point_labels_name", "point_labels"),
            instance_vision.get("instance_seg_decoder_input_mask_input_name", "mask_input"),
            instance_vision.get("instance_seg_decoder_input_has_mask_input_name", "has_mask_input"),
        ],
        [
            instance_vision.get("instance_seg_decoder_output_masks_name", "masks"),
            instance_vision.get("instance_seg_decoder_output_iou_predictions_name", "iou_predictions"),
        ],
    )

    unique = sorted({str(path.relative_to(REPO_ROOT)) for path in checked})
    print(f"OK: validated {len(unique)} physical vision files")
    for path in unique:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())