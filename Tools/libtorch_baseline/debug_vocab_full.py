import struct

vocab_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin"

with open(vocab_path, 'rb') as f:
    magic = f.read(4)
    print(f"Magic: {magic}")
    
    version = struct.unpack('<H', f.read(2))[0]
    checksum = struct.unpack('<I', f.read(4))[0]
    config_vocab_size = struct.unpack('<I', f.read(4))[0]  # unigram count
    max_length = struct.unpack('<I', f.read(4))[0]
    flags = f.read(3)
    total_vocab_size = struct.unpack('<I', f.read(4))[0]
    
    print(f"Version {version}, unigram count {config_vocab_size}, total vocab {total_vocab_size}")
    
    # Check for duplicates and gaps
    seen_ids = {}
    expected_sequential = 273  # Should start after atoms
    
    for i in range(min(100, config_vocab_size)):
        length = struct.unpack('<I', f.read(4))[0]
        text = f.read(length).decode('utf-8', errors='replace')
        score = struct.unpack('<f', f.read(4))[0]
        stored_id = struct.unpack('<I', f.read(4))[0]  # Read as unsigned
        
        print(f"  {i}: '{text}' stored_id={stored_id} (expected ~{expected_sequential + i})")
        
        if stored_id in seen_ids:
            print(f"    ⚠️ DUPLICATE ID {stored_id}! First seen at: '{seen_ids[stored_id]}'")
        seen_ids[stored_id] = text

    # Summary
    print("\n--- ID Summary ---")
    ids = sorted(seen_ids.keys())
    print(f"Min ID: {ids[0]}, Max ID: {ids[-1]}")
    print(f"Expected sequential from {ids[0]} to {ids[0] + len(ids) - 1}")
    
    # Find gaps
    for i in range(len(ids) - 1):
        if ids[i+1] != ids[i] + 1:
            print(f"  Gap: {ids[i]} -> {ids[i+1]} (missing {ids[i+1] - ids[i] - 1} IDs)")
