import struct
from pathlib import Path

path = Path(r"D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt")

with path.open('rb') as f:
    magic = int.from_bytes(f.read(4), 'little')
    version = int.from_bytes(f.read(4), 'little')
    num_sequences = int.from_bytes(f.read(4), 'little')
    file_vocab = int.from_bytes(f.read(4), 'little')

    max_token = -1
    min_token = None
    total_tokens = 0
    longest = 0

    for idx in range(num_sequences):
        len_bytes = f.read(4)
        if not len_bytes:
            break
        seq_len = int.from_bytes(len_bytes, 'little')
        token_bytes = f.read(4 * seq_len)
        if len(token_bytes) < 4 * seq_len:
            break
        tokens = struct.unpack('<{}I'.format(seq_len), token_bytes)
        if tokens:
            local_max = max(tokens)
            local_min = min(tokens)
            if local_max > max_token:
                max_token = local_max
            if min_token is None or local_min < min_token:
                min_token = local_min
        total_tokens += seq_len
        if seq_len > longest:
            longest = seq_len

print({
    'magic': hex(magic),
    'version': version,
    'num_sequences': num_sequences,
    'file_vocab': file_vocab,
    'max_token': max_token,
    'min_token': min_token,
    'longest_seq': longest,
    'total_tokens': total_tokens,
})
