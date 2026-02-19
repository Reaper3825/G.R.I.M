"""Verify GRMT training data: target[j] should == token_ids[j+1]"""
import struct, os, sys

grmt_path = 'resources/models/GRIM-text/training/data'
files = [f for f in os.listdir(grmt_path) if f.endswith('.grmt')]
if not files:
    print('No .grmt files found')
    sys.exit(1)

path = os.path.join(grmt_path, files[0])
print(f'Reading: {path}')
with open(path, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    num_seq = struct.unpack('<I', f.read(4))[0]
    vocab_size = struct.unpack('<I', f.read(4))[0]
    print(f'Magic: {hex(magic)}, Version: {version}, Sequences: {num_seq}, Vocab: {vocab_size}')
    
    total_shift_correct = 0
    total_shift_wrong = 0
    total_masked = 0
    
    # Check first 10 sequences + stats for all
    check_detailed = min(5, num_seq)
    
    for i in range(num_seq):
        seq_len = struct.unpack('<I', f.read(4))[0]
        token_ids = list(struct.unpack(f'<{seq_len}i', f.read(seq_len * 4)))
        targets = list(struct.unpack(f'<{seq_len}i', f.read(seq_len * 4)))
        # Skip side channels
        f.read(seq_len * 4)   # numeric_values (float)
        f.read(seq_len * 1)   # numeric_mask (uint8)
        f.read(seq_len * 16 * 2)  # text_features (16 x uint16)
        f.read(seq_len * 1)   # text_mask (uint8)
        
        # Verify shift: targets[j] should == token_ids[j+1]
        shift_correct = 0
        shift_wrong = 0
        masked = 0
        errors = []
        for j in range(seq_len):
            if targets[j] == -1:
                masked += 1
            elif j + 1 < seq_len and targets[j] == token_ids[j + 1]:
                shift_correct += 1
            else:
                shift_wrong += 1
                if len(errors) < 5:
                    next_tok = token_ids[j + 1] if j + 1 < seq_len else 'N/A'
                    errors.append(f'pos={j}: tok={token_ids[j]} target={targets[j]} next={next_tok}')
        
        total_shift_correct += shift_correct
        total_shift_wrong += shift_wrong
        total_masked += masked
        
        if i < check_detailed:
            print(f'\n  Seq {i}: len={seq_len}')
            print(f'    tokens[:8]={token_ids[:8]}')
            print(f'    targets[:8]={targets[:8]}')
            print(f'    tokens[-5:]={token_ids[-5:]}')
            print(f'    targets[-5:]={targets[-5:]}')
            print(f'    correct={shift_correct} wrong={shift_wrong} masked={masked}')
            for e in errors:
                print(f'    ERROR: {e}')
        
        if shift_wrong > 0 and i >= check_detailed:
            print(f'  Seq {i}: len={seq_len} SHIFT_ERRORS={shift_wrong}')
            for e in errors[:3]:
                print(f'    {e}')

print(f'\n=== SUMMARY ===')
print(f'Total sequences: {num_seq}')
print(f'Total shift_correct: {total_shift_correct}')
print(f'Total shift_wrong: {total_shift_wrong}')
print(f'Total masked: {total_masked}')
if total_shift_wrong > 0:
    print(f'*** {total_shift_wrong} SHIFT ERRORS DETECTED ***')
else:
    print('All targets correctly shifted (target[j] == token_ids[j+1])')
