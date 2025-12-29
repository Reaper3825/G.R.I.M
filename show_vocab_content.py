import struct

vocab_path = 'resources/models/GRIM-text/training/models/vocab.bin'

with open(vocab_path, 'rb') as f:
    # Read header
    header = f.read(4)
    version = struct.unpack('H', f.read(2))[0]
    checksum = struct.unpack('I', f.read(4))[0]
    config_vocab = struct.unpack('I', f.read(4))[0]
    max_token_len = struct.unpack('I', f.read(4))[0]
    flags = f.read(3)
    actual_vocab = struct.unpack('I', f.read(4))[0]
    
    tokens = []
    for i in range(actual_vocab):
        length = struct.unpack('I', f.read(4))[0]
        if length > 0 and length < 1000:
            token_bytes = f.read(length)
            try:
                token = token_bytes.decode('utf-8')
                tokens.append(token)
            except:
                tokens.append(f"<bad:{length}>")

# Categorize tokens
single_char = [t for t in tokens if len(t) == 1 and not t.startswith('<')]
multi_char = [t for t in tokens if len(t) > 1 and not t.startswith('<')]
special = [t for t in tokens if t.startswith('<')]

print(f"Total tokens: {len(tokens)}")
print(f"  Single characters: {len(single_char)}")
print(f"  Multi-character: {len(multi_char)}")
print(f"  Special: {len(special)}")

print("\n=== Sample Multi-character Tokens ===")
print("First 50:")
for i, t in enumerate(multi_char[:50]):
    print(f"  {repr(t):20s}", end="")
    if (i+1) % 4 == 0:
        print()
if len(multi_char[:50]) % 4 != 0:
    print()

print("\nLast 50:")
for i, t in enumerate(multi_char[-50:]):
    print(f"  {repr(t):20s}", end="")
    if (i+1) % 4 == 0:
        print()
if len(multi_char[-50:]) % 4 != 0:
    print()

# Check for common words
common_words = ['the', 'hello', 'world', 'is', 'are', 'and', 'or', 'to', 'in', 'of', 'for', 'repeat', 'word']
print("\n=== Common Word Check ===")
for word in common_words:
    if word in tokens:
        idx = tokens.index(word)
        print(f"  '{word}' found at index {idx}")
    else:
        print(f"  '{word}' NOT FOUND")

# Show token length distribution
print("\n=== Token Length Distribution ===")
from collections import Counter
lengths = Counter(len(t) for t in multi_char)
for length in sorted(lengths.keys())[:15]:
    print(f"  Length {length}: {lengths[length]} tokens")
