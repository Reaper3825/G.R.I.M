"""
Test ONNX vision encoder model loading and inference.
Verifies that the llama3.2-vision encoder ONNX model loads correctly
with external weights and can perform inference.
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

def test_onnx_vision_model():
    """Test loading and running inference with ONNX vision encoder."""
    
    model_path = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-onnx/vision_encoder.onnx")
    
    print("=" * 70)
    print("Testing llama3.2-vision ONNX Encoder")
    print("=" * 70)
    
    # Check if model file exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Check for external weights
    weights_path = model_path.parent / "weights.bin"
    if weights_path.exists():
        print(f"✅ External weights found: {weights_path}")
        print(f"   Size: {weights_path.stat().st_size / (1024**3):.2f} GB")
    else:
        print(f"⚠️  No external weights file (weights might be embedded)")
    
    print("\n" + "-" * 70)
    print("Loading ONNX model...")
    print("-" * 70)
    
    try:
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        # Try to use CUDA if available, otherwise CPU
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("✅ CUDA available - using GPU acceleration (RTX 3080Ti)")
            providers.append('CUDAExecutionProvider')
        else:
            print("⚠️  CUDA not available - using CPU")
        providers.append('CPUExecutionProvider')
        
        # Load model
        start_time = time.time()
        session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Print model info
        print("\n" + "-" * 70)
        print("Model Information:")
        print("-" * 70)
        
        print(f"Execution provider: {session.get_providers()[0]}")
        
        # Input info
        print("\nInputs:")
        for i, input_meta in enumerate(session.get_inputs()):
            print(f"  [{i}] {input_meta.name}")
            print(f"      Shape: {input_meta.shape}")
            print(f"      Type: {input_meta.type}")
        
        # Output info
        print("\nOutputs:")
        for i, output_meta in enumerate(session.get_outputs()):
            print(f"  [{i}] {output_meta.name}")
            print(f"      Shape: {output_meta.shape}")
            print(f"      Type: {output_meta.type}")
        
        # Test inference with dummy data
        print("\n" + "-" * 70)
        print("Running test inference...")
        print("-" * 70)
        
        # Create dummy inputs matching expected shapes
        # The exported model has simplified inputs: just pixel_values [batch, 3, 560, 560]
        
        pixel_values = np.random.randn(1, 3, 560, 560).astype(np.float32)
        
        print(f"Input shapes:")
        print(f"  pixel_values: {pixel_values.shape}")
        
        # Prepare inputs
        inputs = {
            session.get_inputs()[0].name: pixel_values
        }
        
        # Run inference
        print("\nRunning inference (first run may be slow)...")
        start_time = time.time()
        outputs = session.run(None, inputs)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"✅ Inference completed in {inference_time:.1f}ms")
        
        # Print output info
        print(f"\nOutput shape: {outputs[0].shape}")
        print(f"Output dtype: {outputs[0].dtype}")
        
        # Expected: [1, 1600, 1280] - batch_size × num_tokens × embedding_dim
        if len(outputs[0].shape) == 3:
            batch, tokens, embed_dim = outputs[0].shape
            print(f"\n📊 Vision embeddings:")
            print(f"   Batch size: {batch}")
            print(f"   Visual tokens: {tokens}")
            print(f"   Embedding dimensions: {embed_dim}")
            print(f"   Total parameters: {tokens * embed_dim:,}")
        
        # Run a few more times to get average speed
        print("\n" + "-" * 70)
        print("Performance benchmark (10 runs)...")
        print("-" * 70)
        
        times = []
        for i in range(10):
            start = time.time()
            session.run(None, inputs)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"Average: {avg_time:.1f}ms")
        print(f"Min: {min_time:.1f}ms")
        print(f"Max: {max_time:.1f}ms")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed! Model is ready for C++ integration.")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_onnx_vision_model()
    exit(0 if success else 1)
