#!/usr/bin/env python3
"""Analyze GRMT token distribution to understand why training plateaus."""
import struct
from collections import Counter
from pathlib import Path

def analyze_grmt(grmt_path: str, max_sequences: int = 0):
    """Analyze token distribution in GRMT file.
    
    Args:
        grmt_path: Path to .grmt file.
        max_sequences: Max sequences to read. 0 = read ALL sequences.
    """
    with open(grmt_path, 'rb') as f:
        magic, version, num_seq, vocab_size = struct.unpack('<IIII', f.read(16))
        print(f'Header: magic={hex(magic)}, version={version}, num_seq={num_seq}, vocab_size={vocab_size}')
        
        all_tokens = []
        all_targets = []
        seq_lengths = []
        
        sequences_to_read = num_seq if max_sequences <= 0 else min(num_seq, max_sequences)
        is_sampled = sequences_to_read < num_seq
        
        for i in range(sequences_to_read):
            seq_len = struct.unpack('<I', f.read(4))[0]
            seq_lengths.append(seq_len)
            tokens = struct.unpack(f'<{seq_len}I', f.read(seq_len * 4))
            all_tokens.extend(tokens)
            
            # Read targets (int32 × seq_len) - v5+, -1 means masked
            targets = struct.unpack(f'<{seq_len}i', f.read(seq_len * 4))
            all_targets.extend(targets)
            
            # Skip remaining per-sequence fields (GRMT v6 layout from DataLoader.cu):
            #   numeric_values (float32 × seq_len)
            #   numeric_mask   (uint8  × seq_len)
            #   text_features  (uint16 × 16 × seq_len)
            #   text_mask      (uint8  × seq_len)
            #   byte_lengths   (uint16 × seq_len)  - v6
            skip_bytes = (seq_len * 4          # numeric_values (float32)
                        + seq_len              # numeric_mask (uint8)
                        + seq_len * 16 * 2     # text_features (uint16 × 16)
                        + seq_len              # text_mask (uint8)
                        + seq_len * 2)         # byte_lengths (uint16, v6)
            f.seek(skip_bytes, 1)
        
        token_counts = Counter(all_tokens)
        sample_label = f' (sampled {sequences_to_read}/{num_seq})' if is_sampled else ''
        
        print(f'\n=== Sequence Statistics ({sequences_to_read} of {num_seq} sequences{", SAMPLED" if is_sampled else ""}) ===')
        print(f'Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}')
        print(f'Total tokens{sample_label}: {len(all_tokens):,}')
        print(f'Unique tokens seen: {len(token_counts):,} ({len(token_counts)/vocab_size*100:.1f}% of vocab)')
        
        print(f'\n=== Top 30 Most Common Tokens ===')
        for tok, count in token_counts.most_common(30):
            pct = count / len(all_tokens) * 100
            print(f'  Token {tok:5d}: {count:6d} ({pct:5.2f}%)')
        
        print(f'\n=== Token Range ===')
        print(f'Min token ID: {min(all_tokens)}')
        print(f'Max token ID: {max(all_tokens)}')
        
        # Token type distribution
        byte_count = sum(c for t, c in token_counts.items() if t < 256)
        atom_count = sum(c for t, c in token_counts.items() if 256 <= t < 512)
        unigram_count = sum(c for t, c in token_counts.items() if t >= 512)
        
        byte_unique = sum(1 for t in token_counts if t < 256)
        atom_unique = sum(1 for t in token_counts if 256 <= t < 512)
        unigram_unique = sum(1 for t in token_counts if t >= 512)
        
        total = len(all_tokens)
        print(f'\n=== Token Type Distribution ===')
        print(f'Byte (0-255):     {byte_count:7,} occurrences ({byte_count/total*100:5.1f}%)  [{byte_unique:,} unique]')
        print(f'Atom (256-511):   {atom_count:7,} occurrences ({atom_count/total*100:5.1f}%)  [{atom_unique:,} unique]')
        print(f'Unigram (512+):   {unigram_count:7,} occurrences ({unigram_count/total*100:5.1f}%)  [{unigram_unique:,} unique]')
        
        # Frequency distribution with useful buckets
        unique_count = len(token_counts)
        buckets = [
            ('= 1',       lambda c: c == 1),
            ('<= 5',      lambda c: c <= 5),
            ('6-50',      lambda c: 6 <= c <= 50),
            ('51-500',    lambda c: 51 <= c <= 500),
            ('501-5000',  lambda c: 501 <= c <= 5000),
            ('> 5000',    lambda c: c > 5000),
        ]
        
        print(f'\n=== Token Frequency Distribution ({unique_count:,} unique tokens) ===')
        for label, pred in buckets:
            n = sum(1 for c in token_counts.values() if pred(c))
            if n > 0:
                print(f'  Appearing {label:>10s}:  {n:6,} tokens ({n/unique_count*100:5.1f}% of unique)')
        
        # Coverage: what % of total tokens come from top N tokens?
        print(f'\n=== Coverage Analysis ===')
        sorted_counts = sorted(token_counts.values(), reverse=True)
        for threshold in [50, 100, 500, 1000, 5000]:
            if threshold > unique_count:
                cumsum = total
                print(f'Top {threshold:5,} tokens cover: 100.0% of all token occurrences (only {unique_count:,} unique tokens exist in data)')
            else:
                cumsum = sum(sorted_counts[:threshold])
                print(f'Top {threshold:5,} tokens cover: {cumsum/total*100:5.1f}% of all token occurrences')
        
        # Vocab utilization analysis
        print(f'\n=== Vocab Utilization ===')
        used_tokens = len(token_counts)
        unused_vocab = vocab_size - used_tokens
        
        # Break down unused by token type
        used_byte = byte_unique
        used_atom = atom_unique
        used_unigram = unigram_unique
        total_byte_slots = 256
        total_atom_slots = 256
        total_unigram_slots = max(0, vocab_size - 512)
        
        print(f'Byte tokens used:    {used_byte:6,} / {total_byte_slots:6,} ({used_byte/total_byte_slots*100:5.1f}%)')
        print(f'Atom tokens used:    {used_atom:6,} / {total_atom_slots:6,} ({used_atom/total_atom_slots*100:5.1f}%)')
        if total_unigram_slots > 0:
            print(f'Unigram tokens used: {used_unigram:6,} / {total_unigram_slots:6,} ({used_unigram/total_unigram_slots*100:5.1f}%)')
        print(f'Total used:          {used_tokens:6,} / {vocab_size:6,} ({used_tokens/vocab_size*100:5.1f}%)')
        
        if is_sampled:
            print(f'\nNOTE: Only {sequences_to_read}/{num_seq} sequences read ({sequences_to_read/num_seq*100:.1f}%).')
            print(f'      Unused token count is a LOWER BOUND - more tokens likely appear in unread sequences.')
            print(f'      Re-run with max_sequences=0 to scan the full dataset.')
        
        # ==================== TARGET DISTRIBUTION ====================
        masked_count = sum(1 for t in all_targets if t == -1)
        valid_targets = [t for t in all_targets if t >= 0]
        valid_target_count = len(valid_targets)
        target_counts = Counter(valid_targets)
        unique_targets = len(target_counts)
        
        print(f'\n{"=" * 60}')
        print(f'=== TARGET DISTRIBUTION ANALYSIS ===')
        print(f'{"=" * 60}')
        print(f'Total target positions: {len(all_targets):,}')
        print(f'Masked (target=-1):     {masked_count:,} ({masked_count/len(all_targets)*100:.2f}%)')
        print(f'Valid targets:          {valid_target_count:,} ({valid_target_count/len(all_targets)*100:.2f}%)')
        print(f'Unique target tokens:   {unique_targets:,}')
        
        # Where are the masked positions?
        # Check per-sequence: typically position 0 and/or last position
        masked_at_pos0 = 0
        masked_at_last = 0
        masked_at_other = 0
        offset = 0
        for slen in seq_lengths:
            seq_targets = all_targets[offset:offset + slen]
            if slen > 0 and seq_targets[0] == -1:
                masked_at_pos0 += 1
            if slen > 0 and seq_targets[-1] == -1:
                masked_at_last += 1
            mid_masked = sum(1 for t in seq_targets[1:-1] if t == -1) if slen > 2 else 0
            masked_at_other += mid_masked
            offset += slen
        
        print(f'\n=== Target Masking Pattern ===')
        print(f'Sequences with target[0]=-1:    {masked_at_pos0:,} / {len(seq_lengths):,} ({masked_at_pos0/len(seq_lengths)*100:.1f}%)')
        print(f'Sequences with target[-1]=-1:   {masked_at_last:,} / {len(seq_lengths):,} ({masked_at_last/len(seq_lengths)*100:.1f}%)')
        print(f'Masked positions in middle:     {masked_at_other:,}')
        
        print(f'\n=== Top 30 Most Common Targets ===')
        for tok, count in target_counts.most_common(30):
            pct = count / valid_target_count * 100
            # Show how this compares to input distribution
            input_count = token_counts.get(tok, 0)
            input_pct = input_count / total * 100 if total > 0 else 0
            ratio = (count / valid_target_count) / (input_count / total) if input_count > 0 else float('inf')
            print(f'  Target {tok:5d}: {count:7,} ({pct:5.2f}%)  [as input: {input_count:7,} ({input_pct:5.2f}%), ratio={ratio:.2f}]')
        
        # Target type distribution
        tgt_byte = sum(c for t, c in target_counts.items() if t < 256)
        tgt_atom = sum(c for t, c in target_counts.items() if 256 <= t < 512)
        tgt_unigram = sum(c for t, c in target_counts.items() if t >= 512)
        
        print(f'\n=== Target Type Distribution ===')
        print(f'Byte (0-255):     {tgt_byte:7,} ({tgt_byte/valid_target_count*100:5.1f}%)')
        print(f'Atom (256-511):   {tgt_atom:7,} ({tgt_atom/valid_target_count*100:5.1f}%)')
        print(f'Unigram (512+):   {tgt_unigram:7,} ({tgt_unigram/valid_target_count*100:5.1f}%)')
        
        # Target frequency distribution
        tgt_buckets = [
            ('= 1',       lambda c: c == 1),
            ('<= 5',      lambda c: c <= 5),
            ('6-50',      lambda c: 6 <= c <= 50),
            ('51-500',    lambda c: 51 <= c <= 500),
            ('501-5000',  lambda c: 501 <= c <= 5000),
            ('> 5000',    lambda c: c > 5000),
        ]
        
        print(f'\n=== Target Frequency Distribution ({unique_targets:,} unique targets) ===')
        for label, pred in tgt_buckets:
            n = sum(1 for c in target_counts.values() if pred(c))
            if n > 0:
                print(f'  Appearing {label:>10s}:  {n:6,} tokens ({n/unique_targets*100:5.1f}% of unique)')
        
        # Target coverage
        print(f'\n=== Target Coverage Analysis ===')
        sorted_tgt_counts = sorted(target_counts.values(), reverse=True)
        for threshold in [50, 100, 500, 1000, 5000]:
            if threshold > unique_targets:
                print(f'Top {threshold:5,} targets cover: 100.0% of valid targets (only {unique_targets:,} unique targets)')
            else:
                cumsum = sum(sorted_tgt_counts[:threshold])
                print(f'Top {threshold:5,} targets cover: {cumsum/valid_target_count*100:5.1f}% of valid targets')
        
        # Input-vs-target comparison: are tokens that appear often as input
        # also appear proportionally as targets? Big divergence = data issue
        print(f'\n=== Input vs Target Distribution Divergence ===')
        print(f'{"Token":>7s}  {"Input%":>7s}  {"Target%":>8s}  {"Ratio":>6s}  {"Direction":>10s}')
        print(f'{"-"*7:>7s}  {"-"*7:>7s}  {"-"*8:>8s}  {"-"*6:>6s}  {"-"*10:>10s}')
        divergent_count = 0
        for tok, icount in token_counts.most_common(50):
            ipct = icount / total
            tcount = target_counts.get(tok, 0)
            tpct = tcount / valid_target_count if valid_target_count > 0 else 0
            if ipct > 0:
                ratio = tpct / ipct
            else:
                ratio = float('inf')
            # Flag significant divergences (>2x or <0.5x)
            if ratio > 2.0 or ratio < 0.5:
                direction = 'OVER-TGT' if ratio > 2.0 else 'UNDER-TGT'
                print(f'  {tok:5d}  {ipct*100:6.2f}%  {tpct*100:7.2f}%  {ratio:5.2f}x  {direction}')
                divergent_count += 1
        if divergent_count == 0:
            print('  No significant divergences in top 50 tokens.')
        
        # Check for suspicious patterns
        print(f'\n=== Suspicious Patterns ===')
        found_suspicious = False
        
        # Check if there's a dominant token
        top_token, top_count = token_counts.most_common(1)[0]
        if top_count / total > 0.1:
            print(f'WARNING: Token {top_token} appears {top_count:,} times ({top_count/total*100:.1f}% of all tokens) - single token dominates')
            found_suspicious = True
        
        # Check top-5 concentration
        top5_count = sum(c for _, c in token_counts.most_common(5))
        if top5_count / total > 0.3:
            top5_tokens = [str(t) for t, _ in token_counts.most_common(5)]
            print(f'WARNING: Top 5 tokens [{", ".join(top5_tokens)}] cover {top5_count/total*100:.1f}% of all occurrences')
            found_suspicious = True
        
        # Check target vs input consistency
        top_target, top_target_count = target_counts.most_common(1)[0]
        if top_target >= 0 and top_target_count / valid_target_count > 0.15:
            print(f'WARNING: Target {top_target} appears {top_target_count:,} times ({top_target_count/valid_target_count*100:.1f}% of valid targets) - model incentivized to always predict this')
            found_suspicious = True
        
        # Check if target distribution matches input distribution
        # (they should be similar for next-token prediction)
        top_input_set = set(t for t, _ in token_counts.most_common(20))
        top_target_set = set(t for t, _ in target_counts.most_common(20))
        overlap = len(top_input_set & top_target_set)
        if overlap < 10:
            print(f'WARNING: Only {overlap}/20 overlap between top input and target tokens - distribution mismatch')
            found_suspicious = True
        
        # Low diversity check
        if unique_count < 200:
            print(f'WARNING: Only {unique_count} unique tokens in {total:,} total - very low diversity')
            found_suspicious = True
        
        # Unigram utilization check (only meaningful on full dataset)
        if not is_sampled and total_unigram_slots > 0:
            unigram_util = used_unigram / total_unigram_slots * 100
            if unigram_util < 1.0:
                print(f'WARNING: Only {unigram_util:.2f}% of unigram vocab is used - vocab may be oversized or training data too narrow')
                found_suspicious = True
        
        if not found_suspicious:
            print('No suspicious patterns detected.')

if __name__ == '__main__':
    grmt_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data.grmt'
    analyze_grmt(grmt_path)
