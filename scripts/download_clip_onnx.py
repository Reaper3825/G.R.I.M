#!/usr/bin/env python3
"""
Download pre-converted ONNX vision model for G.R.I.M
Uses CLIP (OpenAI) which has excellent ONNX support
"""

import os
import sys
from pathlib import Path
import urllib.request
import json

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"📥 Downloading {url}...")
    
    def reporthook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook)
    print()  # New line after progress

def download_clip_onnx():
    """Download pre-converted CLIP ONNX model"""
    
    models_dir = Path(__file__).parent.parent / "data" / "models" / "vision" / "clip-onnx"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Downloading CLIP ONNX vision model...")
    print(f"📁 Target directory: {models_dir}")
    
    # Use CLIP ViT-B/32 from Hugging Face (pre-converted to ONNX)
    base_url = "https://huggingface.co/rocm/clip-vit-base-patch32/resolve/main/onnx"
    
    files_to_download = {
        "visual.onnx": f"{base_url}/visual.onnx",
        "textual.onnx": f"{base_url}/textual.onnx",
    }
    
    # Also create a simple config
    config = {
        "model_type": "clip",
        "image_size": 224,
        "vision_model": "visual.onnx",
        "text_model": "textual.onnx",
        "description": "CLIP ViT-B/32 ONNX model for vision-language tasks"
    }
    
    try:
        # Download model files
        for filename, url in files_to_download.items():
            dest_path = models_dir / filename
            if dest_path.exists():
                print(f"  ✅ {filename} (already exists)")
            else:
                print(f"  Downloading {filename}...")
                try:
                    download_file(url, str(dest_path))
                    size_mb = dest_path.stat().st_size / (1024 * 1024)
                    print(f"  ✅ {filename} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"  ⚠️ Failed to download {filename}: {e}")
                    print(f"  You can manually download from: {url}")
        
        # Save config
        config_path = models_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Config saved: {config_path}")
        
        # Check if we have at least the visual model
        visual_path = models_dir / "visual.onnx"
        if visual_path.exists():
            print("\n" + "="*60)
            print("🎉 SUCCESS! CLIP ONNX vision model ready")
            print("="*60)
            print(f"\n📄 Visual model: {visual_path}")
            print(f"\n📝 Update ai_config.json with:")
            print(f'   "onnx_vision_model": "{visual_path.as_posix()}"')
            print("\n🚀 Use VisionAIBackend::ONNX_Vision for fast local inference")
            print("\n💡 CLIP provides:")
            print("   - Image classification")
            print("   - Visual similarity search")
            print("   - Zero-shot image recognition")
            print("   - ~150ms inference time (vs 10-30s with Ollama)")
            return str(visual_path)
        else:
            print("\n⚠️ Visual model not found. Trying alternative...")
            # Alternative: Use a simpler direct download
            print("\n📥 Trying alternative ONNX model source...")
            
            # MobileNet ONNX (smaller, faster, good for screen analysis)
            alt_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
            alt_path = models_dir / "mobilenet-vision.onnx"
            
            try:
                download_file(alt_url, str(alt_path))
                size_mb = alt_path.stat().st_size / (1024 * 1024)
                print(f"\n✅ Downloaded MobileNetV2: {alt_path} ({size_mb:.1f} MB)")
                
                config["model_type"] = "mobilenet"
                config["vision_model"] = "mobilenet-vision.onnx"
                config["description"] = "MobileNetV2 ONNX for fast image classification"
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print("\n" + "="*60)
                print("🎉 SUCCESS! MobileNet ONNX vision model ready")
                print("="*60)
                print(f"\n📄 Model: {alt_path}")
                print(f"\n📝 Update ai_config.json with:")
                print(f'   "onnx_vision_model": "{alt_path.as_posix()}"')
                
                return str(alt_path)
                
            except Exception as e:
                print(f"\n❌ Alternative download also failed: {e}")
                return None
    
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = download_clip_onnx()
    
    if not result:
        print("\n" + "="*60)
        print("❌ Download failed - Manual Setup Instructions")
        print("="*60)
        print("\nOption 1: Download CLIP ONNX manually")
        print("1. Visit: https://huggingface.co/rocm/clip-vit-base-patch32/tree/main/onnx")
        print("2. Download 'visual.onnx'")
        print("3. Place in: data/models/vision/clip-onnx/")
        print("\nOption 2: Use MobileNet (simpler)")
        print("1. Visit: https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet")
        print("2. Download 'mobilenetv2-12.onnx'")
        print("3. Place in: data/models/vision/clip-onnx/")
        sys.exit(1)
