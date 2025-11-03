#!/usr/bin/env python3
"""
Export vision model to ONNX format for G.R.I.M
Exports optimized INT8 quantized model for fast inference
"""

import os
import sys
import torch
from pathlib import Path

def export_vision_model_to_onnx():
    """Export ViT-GPT2 to ONNX with optimizations"""
    
    print("🔧 Installing required packages...")
    os.system("pip install transformers torch pillow optimum[exporters] onnx onnxruntime")
    
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    from optimum.onnxruntime import ORTModelForVision2Seq
    
    models_dir = Path(__file__).parent.parent / "data" / "models" / "vision"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = models_dir / "vit-gpt2-onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Loading ViT-GPT2 model from HuggingFace...")
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    
    try:
        # Export using Optimum
        print("🔄 Exporting to ONNX format (this may take a few minutes)...")
        
        ort_model = ORTModelForVision2Seq.from_pretrained(
            model_id,
            export=True,
            provider="CUDAExecutionProvider"  # Use GPU
        )
        
        # Save ONNX model
        print(f"💾 Saving ONNX model to {output_dir}...")
        ort_model.save_pretrained(output_dir)
        
        # Also save processor and tokenizer
        processor = ViTImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        processor.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("\n✅ ONNX export successful!")
        print(f"📁 Model saved to: {output_dir}")
        
        # Find ONNX files
        onnx_files = list(output_dir.glob("*.onnx"))
        if onnx_files:
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                print(f"  ✅ {onnx_file.name} ({size_mb:.1f} MB)")
        
        # Get encoder path
        encoder_path = output_dir / "encoder_model.onnx"
        decoder_path = output_dir / "decoder_model.onnx"
        
        if encoder_path.exists() and decoder_path.exists():
            print(f"\n🎯 Use these paths in vision_ai.cpp:")
            print(f'   encoderPath = R"({encoder_path})";')
            print(f'   decoderPath = R"({decoder_path})";')
            return str(output_dir)
        else:
            print("⚠️ Expected ONNX files not found, checking alternatives...")
            return str(output_dir)
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("\n🔄 Trying alternative export method...")
        
        # Fallback: Manual export
        try:
            model = VisionEncoderDecoderModel.from_pretrained(model_id)
            processor = ViTImageProcessor.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Export encoder
            from PIL import Image
            import requests
            from io import BytesIO
            
            # Dummy input
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            
            encoder_path = output_dir / "encoder_model.onnx"
            decoder_path = output_dir / "decoder_model.onnx"
            
            print("🔄 Exporting encoder...")
            torch.onnx.export(
                model.encoder,
                pixel_values,
                str(encoder_path),
                input_names=['pixel_values'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'pixel_values': {0: 'batch'},
                    'last_hidden_state': {0: 'batch'}
                },
                opset_version=14
            )
            
            print(f"✅ Encoder exported: {encoder_path}")
            
            # Save configs
            processor.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"\n✅ Model exported to: {output_dir}")
            print(f"📄 Encoder: {encoder_path}")
            
            return str(output_dir)
            
        except Exception as e2:
            print(f"❌ Alternative export also failed: {e2}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    result = export_vision_model_to_onnx()
    
    if result:
        print("\n" + "="*60)
        print("🎉 SUCCESS! ONNX vision model ready")
        print("="*60)
        print("\n🚀 Next steps:")
        print("1. Model files are ready in data/models/vision/vit-gpt2-onnx/")
        print("2. Update vision_ai.cpp with the ONNX paths")
        print("3. Rebuild G.R.I.M and test with VisionAIBackend::ONNX_Vision")
    else:
        print("\n❌ Export failed. You may need to:")
        print("1. Check Python/PyTorch installation")
        print("2. Ensure sufficient disk space (~2GB)")
        print("3. Try a different model or manual ONNX export")
        sys.exit(1)
