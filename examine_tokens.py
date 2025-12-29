import struct

vocab_path = 'resources/models/GRIM-text/training/models/vocab.bin'

with open(vocab_path, 'rb') as f:
    # Read header
    header = f.read(4)  # KTMG
    if header != b'KTMG':
        print(f"Invalid header: {header}")
        exit(1)
    
    version = struct.unpack('H', f.read(2))[0]  # 2 bytes, not 4
    checksum = struct.unpack('I', f.read(4))[0]
    config_vocab = struct.unpack('I', f.read(4))[0]
    max_token_len = struct.unpack('I', f.read(4))[0]
    flags = f.read(3)
    
    # Read actual vocab size
    actual_vocab = struct.unpack('I', f.read(4))[0]
    
    print(f"Version: {version}, Config vocab: {config_vocab}, Actual vocab: {actual_vocab}\n")
    
    tokens = []
    for i in range(actual_vocab):  # Read ALL tokens
        try:
            # Each token: length (4 bytes) + text (length bytes) - NO score field!
            length_bytes = f.read(4)
            if len(length_bytes) < 4:
                break
            length = struct.unpack('I', length_bytes)[0]
            
            if length == 0 or length > 1000:
                print(f"Token {i}: invalid length {length}")
                break
                
            token_bytes = f.read(length)
            if len(token_bytes) < length:
                break
                
            try:
                token = token_bytes.decode('utf-8')
            except:
                token = repr(token_bytes)
            tokens.append(token)
        except Exception as e:
            print(f"Error at token {i}: {e}")
            break

print(f"Successfully read {len(tokens)} tokens\n")
print("Tokens 0-50:")
for i in range(min(50, len(tokens))):
    token = tokens[i]
    print(f"{i:3d}: {repr(token):20s}")

print("\nTokens 70-90 (the 'garbage' range):")
for i in range(70, min(90, len(tokens))):
    if i < len(tokens):
        token = tokens[i]
        print(f"{i:3d}: {repr(token):20s}")

print("\nToken types:")
single_char = sum(1 for t in tokens if len(t) == 1 and not t.startswith('<'))
multi_char = sum(1 for t in tokens if len(t) > 1 and not t.startswith('<'))
special = sum(1 for t in tokens if t.startswith('<'))
print(f"  Single characters: {single_char}")
print(f"  Multi-character subwords: {multi_char}")
print(f"  Special tokens: {special}")
