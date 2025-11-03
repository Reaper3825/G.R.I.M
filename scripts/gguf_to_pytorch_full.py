"""
GGUF to PyTorch Converter for FULL llama3.2-vision (Vision + Language Model)
Extracts all 908 tensors (vision encoder + language model)
"""

import sys
import io
# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import struct
import numpy as np
from pathlib import Path
from typing import Dict
import json

# Import the existing GGUF reader
from gguf_to_pytorch import GGUFReader

def main():
    print("="*80)
    print("GGUF to PyTorch - FULL Vision-Language Model Extraction")
    print("="*80)
    
    gguf_path = Path(r"C:\Users\Merlinthegrim\.ollama\models\blobs\sha256-9999d473417a8e179d993498195be5f42cab963acc75f4a6b15d981e8b68abed")
    
    if not gguf_path.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return
    
    print(f"Input: {gguf_path}")
    print(f"Size: {gguf_path.stat().st_size / (1024**3):.2f} GB\n")
    
    reader = GGUFReader(gguf_path)
    
    try:
        # Parse GGUF structure
        tensor_count, metadata_count = reader.parse_header()
        reader.parse_metadata(metadata_count)
        tensor_info_list = reader.parse_tensor_info(tensor_count)
        
        # Analyze tensor prefixes
        prefixes = {}
        for info in tensor_info_list:
            name = info['name']
            prefix = name.split('.')[0]
            if prefix not in prefixes:
                prefixes[prefix] = 0
            prefixes[prefix] += 1
        
        print(f"\n[Analysis] Tensor prefixes:")
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} tensors")
        
        # Calculate data offset (after header, metadata, and tensor info)
        data_offset = reader.file.tell()
        
        # Align to 32 bytes
        alignment = 32
        data_offset = ((data_offset + alignment - 1) // alignment) * alignment
        
        print(f"\n[GGUF] Tensor data starts at offset: {data_offset}")
        
        # Extract ALL tensors
        print(f"\n[GGUF] Extracting ALL tensors (this will take a few minutes)...")
        all_extracted = {}
        
        for i, info in enumerate(tensor_info_list):
            name = info['name']
            # Show progress every 10 tensors for better visibility
            if (i + 1) % 10 == 0 or i == 0:
                percent = ((i + 1) / len(tensor_info_list)) * 100
                print(f"  Progress: [{i+1}/{len(tensor_info_list)}] ({percent:.1f}%) {name}", flush=True)
            
            tensor = reader.load_tensor(info, data_offset)
            if tensor is not None:
                all_extracted[name] = tensor
        
        print(f"\n✅ Extracted {len(all_extracted)} tensors")
        
        # Categorize tensors
        vision_tensors = {k: v for k, v in all_extracted.items() if k.startswith('v.')}
        language_tensors = {k: v for k, v in all_extracted.items() if not k.startswith('v.')}
        
        print(f"\n[Categorization]:")
        print(f"  Vision encoder: {len(vision_tensors)} tensors")
        print(f"  Language model: {len(language_tensors)} tensors")
        
        # Save to separate directories
        output_base = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-full")
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(output_base / "metadata.json", 'w') as f:
            json_metadata = {k: str(v) if not isinstance(v, (int, float, str, bool, list)) else v 
                           for k, v in reader.metadata.items()}
            json.dump(json_metadata, f, indent=2)
        print(f"\n✅ Metadata saved to: {output_base / 'metadata.json'}")
        
        # Save categorized tensors first (safer for large files)
        print(f"\n[Saving] Saving categorized tensors...")
        print(f"  Saving vision tensors...")
        np.savez_compressed(output_base / "vision_tensors.npz", **vision_tensors)
        vision_size_gb = (output_base / "vision_tensors.npz").stat().st_size / (1024**3)
        print(f"  ✅ Vision tensors: {vision_size_gb:.2f} GB")
        
        print(f"  Saving language tensors...")
        np.savez_compressed(output_base / "language_tensors.npz", **language_tensors)
        language_size_gb = (output_base / "language_tensors.npz").stat().st_size / (1024**3)
        print(f"  ✅ Language tensors: {language_size_gb:.2f} GB")
        
        # Save all tensors in one file (optional, might fail for large files)
        print(f"\n[Saving] Writing combined file (this may take a minute)...")
        try:
            np.savez_compressed(output_base / "all_tensors.npz", **all_extracted)
            file_size_gb = (output_base / "all_tensors.npz").stat().st_size / (1024**3)
            print(f"✅ All tensors saved: {output_base / 'all_tensors.npz'} ({file_size_gb:.2f} GB)")
        except Exception as e:
            print(f"⚠️  Could not save combined file (too large): {e}")
            print(f"   Use vision_tensors.npz and language_tensors.npz instead")
        
        # Print statistics
        total_params = sum(t.size for t in all_extracted.values())
        vision_params = sum(t.size for t in vision_tensors.values())
        language_params = sum(t.size for t in language_tensors.values())
        
        print("\n" + "="*80)
        print("✅ Full Model Extraction Complete!")
        print("="*80)
        print(f"Total parameters: {total_params/1e9:.2f}B")
        print(f"  Vision encoder: {vision_params/1e6:.1f}M params")
        print(f"  Language model: {language_params/1e9:.2f}B params")
        print(f"\nNext: Build PyTorch model from extracted tensors")
        
    finally:
        reader.close()

if __name__ == "__main__":
    main()
