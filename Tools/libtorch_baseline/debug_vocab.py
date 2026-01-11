#!/usr/bin/env python3
import struct
vocab_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin'

with open(vocab_path, 'rb') as f:
    f.read(4)  # magic
    version = struct.unpack('<H', f.read(2))[0]
    f.read(4)  # checksum
    config_vocab = struct.unpack('<I', f.read(4))[0]
    f.read(4)  # max_len
    f.read(3)  # flags
    total_vocab = struct.unpack('<I', f.read(4))[0]
    
    print(f"Version {version}, unigram count {config_vocab}, total vocab {total_vocab}")
    print("First 20 pieces:")
    
    for i in range(min(20, config_vocab)):
        piece_len = struct.unpack('<I', f.read(4))[0]
        text = f.read(piece_len).decode('utf-8', errors='replace')
        score = struct.unpack('<f', f.read(4))[0]
        if version >= 3:
            stored_id = struct.unpack('<I', f.read(4))[0]
        else:
            stored_id = 512 + i
        
        computed_id = 512 + i
        print(f'  {i}: text="{repr(text)}" stored_id={stored_id} computed_id={computed_id}')
