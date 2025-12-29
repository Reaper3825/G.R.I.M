import struct

vocab_path = 'resources/models/GRIM-text/training/data/vocab.bin'


with open(vocab_path, 'rb') as f:
    # Read header
    header = f.read(4)
    version = struct.unpack('H', f.read(2))[0]
    checksum = struct.unpack('I', f.read(4))[0]
    config_vocab = struct.unpack('I', f.read(4))[0]
    max_token_len = struct.unpack('I', f.read(4))[0]
    flags = f.read(3)
    actual_vocab = struct.unpack('I', f.read(4))[0]
    
    print(f"Header: {header}")
    print(f"Version: {version}")
    print(f"Config vocab: {config_vocab}")
    print(f"Actual vocab: {actual_vocab}")
    print(f"Max token length: {max_token_len}")
    print()
    
    tokens = []
    token_ids = []
    
    # Format: [4-byte length][token_bytes][4-byte token_id]
    for i in range(actual_vocab):
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            break
            
        length = struct.unpack('I', length_bytes)[0]
        
        if 0 < length < 1000:
            token_bytes = f.read(length)
            if len(token_bytes) < length:
                break
                
            try:
                token = token_bytes.decode('utf-8')
                tokens.append(token)
            except:
                tokens.append(f"<bad:{length}>")
            
            # Read the token ID that follows each token
            id_bytes = f.read(4)
            if len(id_bytes) == 4:
                token_id = struct.unpack('I', id_bytes)[0]
                token_ids.append(token_id)
            else:
                token_ids.append(0)
        else:
            # Invalid length - vocab might be corrupted beyond this point
            tokens.append("<invalid>")
            token_ids.append(0)

# ==============================
# NUMERIC TOKEN-ID RANGE (NO LIST)
# ==============================

start = 0
end = min(50, len(tokens))   # Show first 50

print(f"\nTotal tokens parsed: {len(tokens)}\n")
print("First 50 tokens:")
for token_id in range(start, end):
    extra_info = f" [id={token_ids[token_id]}]" if token_id < len(token_ids) else ""
    print(f"  {token_id:5d}: {repr(tokens[token_id])}{extra_info}")

# Show last 20
print("\nLast 20 tokens:")
start_last = max(0, len(tokens) - 20)
for token_id in range(start_last, len(tokens)):
    extra_info = f" [id={token_ids[token_id]}]" if token_id < len(token_ids) else ""
    print(f"  {token_id:5d}: {repr(tokens[token_id])}{extra_info}")


