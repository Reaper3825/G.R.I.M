#!/usr/bin/env python3
"""
Download quantized ONNX vision model for G.R.I.M perception system
Uses BLIP or ViT-GPT2 for image captioning with INT8 quantization
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def download_vision_model():
    """Download quantized ONNX vision model"""
    
    models_dir = Path(__file__).parent.parent / "data" / "models" / "vision"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Downloading quantized ONNX vision model...")
    print(f"📁 Target directory: {models_dir}")
    
    # Option 1: BLIP Image Captioning (Salesforce, optimized for ONNX)
    # ~500MB, INT8 quantized, good quality captions
    try:
        print("\n📥 Attempting to download BLIP-base model...")
        
        # Download ONNX model files
        # Using nlpconnect/vit-gpt2-image-captioning as it has ONNX support
        model_files = [
            "model.onnx",
            "config.json", 
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json"
        ]
        
        repo_id = "nlpconnect/vit-gpt2-image-captioning"
        
        for filename in model_files:
            try:
                print(f"  Downloading {filename}...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(models_dir),
                    local_dir=str(models_dir / "vit-gpt2"),
                    local_dir_use_symlinks=False
                )
                print(f"  ✅ {filename}")
            except Exception as e:
                print(f"  ⚠️ Skipping {filename}: {e}")
        
        print(f"\n✅ Model downloaded to: {models_dir / 'vit-gpt2'}")
        print("\n📝 Model info:")
        print("  - Architecture: ViT-GPT2 (Vision Transformer + GPT2)")
        print("  - Size: ~500MB")
        print("  - Format: ONNX (optimized for inference)")
        print("  - Use case: Image captioning, scene description")
        
        # Check if model file exists
        model_path = models_dir / "vit-gpt2" / "model.onnx"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ Model ready: {model_path}")
            print(f"📊 Size: {size_mb:.1f} MB")
            return str(model_path)
        else:
            print("\n⚠️ Model file not found, trying alternative download method...")
            
            # Alternative: Try downloading entire repo
            print("📥 Downloading complete model repository...")
            local_dir = models_dir / "vit-gpt2"
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                allow_patterns=["*.onnx", "*.json", "*.txt"]
            )
            
            # Search for ONNX file
            for onnx_file in local_dir.rglob("*.onnx"):
                print(f"✅ Found ONNX model: {onnx_file}")
                return str(onnx_file)
            
            print("❌ No ONNX model found in download")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_path = download_vision_model()
    
    if model_path:
        print("\n" + "="*60)
        print("🎉 SUCCESS! Vision model ready for ONNX Runtime")
        print("="*60)
        print(f"\n📄 Update vision_ai.cpp with model path:")
        print(f'   const std::string modelPath = R"({model_path})";')
        print("\n🚀 You can now use VisionAIBackend::ONNX_Vision")
    else:
        print("\n❌ Download failed. Manual download options:")
        print("1. Visit: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning")
        print("2. Download 'model.onnx' and config files")
        print(f"3. Place in: {models_dir / 'vit-gpt2'}")
        sys.exit(1)
