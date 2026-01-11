import struct

vocab_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin"

with open(vocab_path, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<H', f.read(2))[0]
    checksum = struct.unpack('<I', f.read(4))[0]
    config_vocab_size = struct.unpack('<I', f.read(4))[0]
    max_length = struct.unpack('<I', f.read(4))[0]
    flags = f.read(3)
    total_vocab_size = struct.unpack('<I', f.read(4))[0]
    
    print(f"Magic: {magic}")
    print(f"Version {version}")
    print(f"Unigram count: {config_vocab_size}")
    print(f"Total vocab (including bytes+atoms): {total_vocab_size}")
    print()
    
    # Read raw bytes for first 10 pieces
    print("First 10 pieces (raw bytes):")
    for i in range(10):
        pos_before = f.tell()
        length = struct.unpack('<I', f.read(4))[0]
        text = f.read(length)
        score_bytes = f.read(4)
        score = struct.unpack('<f', score_bytes)[0]
        id_bytes = f.read(4)
        stored_id = struct.unpack('<I', id_bytes)[0]  # Unsigned
        stored_id_signed = struct.unpack('<i', id_bytes)[0]  # Signed
        
        print(f"  {i}: len={length} text={text!r} score_bytes={score_bytes.hex()} id_bytes={id_bytes.hex()}")
        print(f"      text='{text.decode('utf-8', errors='replace')}' score={score:.4f} id_unsigned={stored_id} id_signed={stored_id_signed}")
