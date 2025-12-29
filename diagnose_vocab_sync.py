#!/usr/bin/env python3
"""
Diagnose vocab size synchronization between vocab.bin and .grmt files
"""

import struct
import sys
from pathlib import Path

def read_vocab_bin(path):
    """Read vocab.bin file and extract both vocab sizes"""
    try:
        with open(path, 'rb') as f:
            # Read GMTK header
            magic = f.read(4)
            if magic != b'KTMG':  # Little-endian 'GMTK'
                print(f"  ❌ Invalid magic: {magic}")
                return None, None
            
            version = struct.unpack('<H', f.read(2))[0]
            print(f"  Version: {version}")
            
            if version >= 2:
                # Skip checksum (4 bytes)
                checksum = struct.unpack('<I', f.read(4))[0]
                print(f"  Checksum: 0x{checksum:08x}")
                
                # Read config vocab_size (4 bytes) - the TARGET size
                config_vocab_size = struct.unpack('<I', f.read(4))[0]
                print(f"  Config vocab_size (target): {config_vocab_size}")
                
                # Read max_length (4 bytes)
                max_length = struct.unpack('<I', f.read(4))[0]
                print(f"  Max length: {max_length}")
                
                # Skip 3 bools
                f.read(3)
                
                # Read actual vocab_size (4 bytes) - the ACTUAL number of tokens
                actual_vocab_size = struct.unpack('<I', f.read(4))[0]
                print(f"  Actual vocab_size (real): {actual_vocab_size}")
                
                return config_vocab_size, actual_vocab_size
            else:
                print(f"  ⚠️  Old version {version}, cannot parse accurately")
                return None, None
                
    except Exception as e:
        print(f"  ❌ Error reading vocab.bin: {e}")
        return None, None

def read_grmt_file(path):
    """Read GRMT file header and extract vocab size"""
    try:
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x474D5254:  # 'GRMT'
                print(f"  ❌ Invalid magic: 0x{magic:08x}")
                return None
            
            version = struct.unpack('<I', f.read(4))[0]
            num_sequences = struct.unpack('<I', f.read(4))[0]
            vocab_size = struct.unpack('<I', f.read(4))[0]
            
            print(f"  Version: {version}")
            print(f"  Sequences: {num_sequences}")
            print(f"  Vocab size: {vocab_size}")
            
            return vocab_size
            
    except Exception as e:
        print(f"  ❌ Error reading GRMT: {e}")
        return None

def main():
    grim_root = Path(__file__).parent
    data_dir = grim_root / "resources" / "models" / "GRIM-text" / "training" / "data"
    models_dir = grim_root / "resources" / "models" / "GRIM-text" / "training" / "models"
    
    print("=" * 60)
    print("VOCAB SIZE SYNCHRONIZATION DIAGNOSTIC")
    print("=" * 60)
    
    # Check vocab.bin in data directory
    vocab_data_path = data_dir / "vocab.bin"
    print(f"\n📁 Checking {vocab_data_path}")
    if vocab_data_path.exists():
        config_size_data, actual_size_data = read_vocab_bin(vocab_data_path)
    else:
        print("  ⚠️  File not found")
        config_size_data, actual_size_data = None, None
    
    # Check vocab.bin in models directory
    vocab_models_path = models_dir / "vocab.bin"
    print(f"\n📁 Checking {vocab_models_path}")
    if vocab_models_path.exists():
        config_size_models, actual_size_models = read_vocab_bin(vocab_models_path)
    else:
        print("  ⚠️  File not found")
        config_size_models, actual_size_models = None, None
    
    # Check GRMT files
    grmt_files = {
        "training_data.grmt": data_dir / "training_data.grmt",
        "validation_data.grmt": data_dir / "validation_data.grmt",
        "test_data.grmt": data_dir / "test_data.grmt",
    }
    
    grmt_vocab_sizes = {}
    for name, path in grmt_files.items():
        print(f"\n📁 Checking {path}")
        if path.exists():
            grmt_vocab_sizes[name] = read_grmt_file(path)
        else:
            print("  ⚠️  File not found")
            grmt_vocab_sizes[name] = None
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Determine which vocab.bin to use as reference
    if actual_size_data is not None:
        reference_actual = actual_size_data
        reference_config = config_size_data
        reference_location = "data/vocab.bin"
    elif actual_size_models is not None:
        reference_actual = actual_size_models
        reference_config = config_size_models
        reference_location = "models/vocab.bin"
    else:
        print("❌ No valid vocab.bin found!")
        return 1
    
    print(f"\nUsing {reference_location} as reference:")
    print(f"  Config vocab_size (target): {reference_config}")
    print(f"  Actual vocab_size (real):   {reference_actual}")
    
    # Check for mismatches
    issues = []
    
    if reference_config != reference_actual:
        issues.append(f"⚠️  vocab.bin has mismatch: config={reference_config}, actual={reference_actual}")
    
    for name, grmt_size in grmt_vocab_sizes.items():
        if grmt_size is not None:
            if grmt_size != reference_actual:
                issues.append(f"❌ {name} vocab_size ({grmt_size}) != vocab.bin actual ({reference_actual})")
            else:
                print(f"✅ {name} is in sync with vocab.bin")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 RECOMMENDATION:")
        print("  The GRMT files should store the ACTUAL vocab size from vocab_.size(),")
        print("  not the config target vocab_size. Run PrepareTrainingDataFromCache")
        print("  with force_rebuild=true to regenerate synchronized files.")
        return 1
    else:
        print("\n✅ All files are synchronized!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
