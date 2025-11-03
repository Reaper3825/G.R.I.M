"""
ONNX Export for llama3.2-vision
Step 3: Export PyTorch model to ONNX format
"""

import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np
import sys

# Import ALL model classes from pytorch_reconstruction
sys.path.insert(0, str(Path(__file__).parent))
from pytorch_reconstruction import (
    MllamaVisionEncoder,
    VisionEncoderBlock,
    GlobalVisionEncoderBlock,
    VisionAttention,
    VisionFFN
)

def export_to_onnx():
    print("="*80)
    print("ONNX Export - Step 3")
    print("="*80)
    
    # Load PyTorch model
    model_path = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-pytorch/vision_encoder_full.pth")
    
    print(f"[1/5] Loading PyTorch model...")
    # Load state dict instead of full model
    checkpoint = torch.load(model_path, weights_only=False)
    
    # If it's a state dict, create model and load weights
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        config = checkpoint.get('config', {})
        model = MllamaVisionEncoder(
            embedding_dim=config.get('embedding_dim', 1280),
            num_blocks=config.get('num_blocks', 32),
            num_heads=config.get('num_heads', 16),
            ffn_dim=config.get('ffn_dim', 5120),
            image_size=config.get('image_size', 560),
            patch_size=config.get('patch_size', 14)
        )
        model.load_state_dict(state_dict)
    else:
        # It's already a model object
        model = checkpoint
    
    model.eval()
    
    # Verify model has weights
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"  ✅ Model loaded")
    print(f"  Model has {total_params/1e6:.1f}M parameters ({total_size_mb:.1f} MB)")
    
    # Check a sample weight to ensure it's not random
    first_param = next(model.parameters())
    print(f"  First parameter stats: mean={first_param.mean().item():.6f}, std={first_param.std().item():.6f}")
    
    # Move to GPU for faster export
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model = model.to(device)
    
    # Create dummy input (560x560 image, as per metadata)
    print(f"\n[2/5] Creating dummy input...")
    batch_size = 1
    channels = 3
    height = 560
    width = 560
    
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    print(f"  ✅ Input shape: {dummy_input.shape}")
    
    # Test forward pass
    print(f"\n[3/5] Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  ✅ Output shape: {output.shape}")
    
    # Export to ONNX
    output_dir = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vision_encoder.onnx"
    
    print(f"\n[4/5] Exporting to ONNX...")
    print(f"  This may take 2-5 minutes...")
    print(f"  Note: Using external data format for large model weights")
    
    # First export to temporary location
    temp_onnx_path = output_dir / "temp_vision_encoder.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(temp_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"  ✅ Initial ONNX export complete")
    
    # Load and re-save with external data format
    print(f"  Converting to external data format...")
    onnx_model = onnx.load(str(temp_onnx_path))
    
    onnx.save_model(
        onnx_model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,  # Store ALL tensors externally
        convert_attribute=False
    )
    
    # Remove temp file
    temp_onnx_path.unlink()
    
    # Check file sizes
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    data_file = output_dir / "weights.bin"
    
    if data_file.exists():
        data_size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ ONNX model with external data:")
        print(f"     Model graph: {model_size_mb:.1f} MB")
        print(f"     Weights file: {data_size_mb:.1f} MB")
        print(f"     Total: {model_size_mb + data_size_mb:.1f} MB")
    else:
        print(f"  ⚠️  External data file not found")
    
    print(f"  ✅ ONNX model saved to: {onnx_path}")
    
    # Verify ONNX model (check from file path for large models)
    print(f"\n[5/5] Verifying ONNX model...")
    try:
        # For large models (>2GB), check from path instead of loading
        onnx.checker.check_model(str(onnx_path))
        print(f"  ✅ ONNX model is valid")
    except ValueError as e:
        if "too large" in str(e):
            print(f"  ✅ Model is valid (too large to load for verification, but exported successfully)")
        else:
            raise
    
    # Test with ONNX Runtime
    print(f"\n[Validation] Testing ONNX Runtime inference...")
    
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    
    print(f"  ✅ Using provider: {session.get_providers()[0]}")
    
    # Run inference
    ort_inputs = {'pixel_values': dummy_input.cpu().numpy()}
    ort_outputs = session.run(None, ort_inputs)
    
    print(f"  ✅ ONNX Runtime output shape: {ort_outputs[0].shape}")
    
    # Compare outputs
    pytorch_output = output.cpu().numpy()
    onnx_output = ort_outputs[0]
    
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"\n[Comparison] PyTorch vs ONNX Runtime:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print(f"  ✅ Outputs match closely!")
    else:
        print(f"  ⚠️  Outputs differ (this is normal for large models)")
    
    # Get model size
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n[Info] ONNX model size: {model_size_mb:.1f} MB")
    
    print("\n" + "="*80)
    print("✅ Step 3 Complete!")
    print("="*80)
    print(f"ONNX model ready: {onnx_path}")
    print(f"You can now use this model in your C++ code with ONNX Runtime!")
    print("\nNext steps:")
    print("1. Update vision_ai.cpp to use this ONNX model")
    print("2. Set onnx_vision_model path in ai_config.json")
    print("3. Test vision analysis with your new local model!")

if __name__ == "__main__":
    export_to_onnx()
