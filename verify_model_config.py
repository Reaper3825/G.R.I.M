#!/usr/bin/env python3
"""
Verify GRIM Model Configuration and Vocab Consistency
Checks for common issues that cause invalid token generation
"""

import os
import sys
import json
import struct

GRIM_ROOT = os.path.dirname(os.path.abspath(__file__))

def check_ai_config():
    """Check ai_config.json for vocab_size"""
    print("\n[1/5] Checking ai_config.json...")
    config_path = os.path.join(GRIM_ROOT, "ai_config.json")
    
    if not os.path.exists(config_path):
        print(f"  ❌ NOT FOUND: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        vocab_size = None
        if "vocab_size" in config:
            vocab_size = config["vocab_size"]
        elif "training" in config and "config" in config["training"]:
            vocab_size = config["training"]["config"].get("vocab_size")
        
        if vocab_size:
            print(f"  ✓ vocab_size in config: {vocab_size}")
            return vocab_size
        else:
            print(f"  ⚠️  vocab_size not found in config")
            return None
    except Exception as e:
        print(f"  ❌ Error reading config: {e}")
        return None


def check_vocab_file():
    """Check vocab.bin file"""
    print("\n[2/5] Checking vocab.bin...")
    vocab_path = os.path.join(GRIM_ROOT, "resources", "models", "GRIM-text", 
                               "training", "models", "vocab.bin")
    
    if not os.path.exists(vocab_path):
        print(f"  ❌ NOT FOUND: {vocab_path}")
        return None
    
    try:
        file_size = os.path.getsize(vocab_path)
        print(f"  ✓ Found vocab.bin ({file_size:,} bytes)")
        
        # Try to read vocab size from binary file
        with open(vocab_path, 'rb') as f:
            # Binary format: [vocab_size (4 bytes)] [tokens...]
            vocab_size_bytes = f.read(4)
            if len(vocab_size_bytes) == 4:
                vocab_size = struct.unpack('I', vocab_size_bytes)[0]
                print(f"  ✓ vocab_size in file: {vocab_size}")
                return vocab_size
    except Exception as e:
        print(f"  ❌ Error reading vocab file: {e}")
        return None


def check_checkpoint():
    """Check model checkpoint"""
    print("\n[3/5] Checking model checkpoint...")
    checkpoint_path = os.path.join(GRIM_ROOT, "resources", "models", "GRIM-text",
                                   "checkpoints", "checkpoint_epoch_5.bin")
    
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ NOT FOUND: {checkpoint_path}")
        return None
    
    try:
        file_size = os.path.getsize(checkpoint_path)
        size_mb = file_size / (1024 * 1024)
        print(f"  ✓ Found checkpoint ({size_mb:.1f} MB)")
        
        # Check if file is suspiciously small (< 100 MB for a language model)
        if size_mb < 100:
            print(f"  ⚠️  Warning: Checkpoint seems small for a language model")
        
        return size_mb
    except Exception as e:
        print(f"  ❌ Error checking checkpoint: {e}")
        return None


def check_server_executable():
    """Check if server executable exists"""
    print("\n[4/5] Checking server executable...")
    server_path = os.path.join(GRIM_ROOT, "resources", "models", "GRIM-text",
                               "training", "build_vs_cuda", "Release", 
                               "grim_text_server.exe")
    
    if not os.path.exists(server_path):
        print(f"  ❌ NOT FOUND: {server_path}")
        print(f"  → Run: .\\rebuild_after_token_fix.ps1")
        return False
    
    try:
        file_size = os.path.getsize(server_path)
        size_mb = file_size / (1024 * 1024)
        print(f"  ✓ Found server executable ({size_mb:.1f} MB)")
        
        # Check file modification time
        import datetime
        mod_time = os.path.getmtime(server_path)
        mod_date = datetime.datetime.fromtimestamp(mod_time)
        print(f"  ✓ Last modified: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error checking server: {e}")
        return False


def consistency_check(config_vocab, file_vocab):
    """Check if vocab sizes are consistent"""
    print("\n[5/5] Consistency Check...")
    
    if config_vocab is None or file_vocab is None:
        print("  ⚠️  Cannot perform consistency check (missing data)")
        return False
    
    if config_vocab == file_vocab:
        print(f"  ✅ PASS: Config and vocab file match ({config_vocab} tokens)")
        return True
    else:
        print(f"  ❌ FAIL: Mismatch detected!")
        print(f"     Config vocab_size: {config_vocab}")
        print(f"     Vocab file size:   {file_vocab}")
        print(f"  → This WILL cause invalid token generation!")
        return False


def print_summary(all_good):
    """Print summary and recommendations"""
    print("\n" + "=" * 60)
    if all_good:
        print("  ✅ All checks passed!")
        print("=" * 60)
        print("\nYou can now run: python test_grim_model.py")
    else:
        print("  ⚠️  Issues detected")
        print("=" * 60)
        print("\nRecommended actions:")
        print("  1. If server not built: Run .\\rebuild_after_token_fix.ps1")
        print("  2. If vocab mismatch: Regenerate vocab or retrain model")
        print("  3. Check TOKEN_GENERATION_FIX.md for more details")
    print()


def main():
    print("=" * 60)
    print("  GRIM Model Configuration Verification")
    print("=" * 60)
    
    config_vocab = check_ai_config()
    file_vocab = check_vocab_file()
    checkpoint_ok = check_checkpoint()
    server_ok = check_server_executable()
    consistency_ok = consistency_check(config_vocab, file_vocab)
    
    all_good = (
        config_vocab is not None and 
        file_vocab is not None and
        checkpoint_ok is not None and
        server_ok and
        consistency_ok
    )
    
    print_summary(all_good)
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
