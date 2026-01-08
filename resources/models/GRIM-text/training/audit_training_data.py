#!/usr/bin/env python3
"""
Audit training data quality by sampling random sequences.
Works without C++ tokenizer bindings - loads vocab.txt directly.

Usage:
    python audit_training_data.py --count 20
    python audit_training_data.py --count 50 --seed 42
"""

import sys
import struct
import random
import argparse
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ATOM_TOKEN_START = 256
ATOM_VOCAB_SIZE = 256
ATOM_TOKEN_END = ATOM_TOKEN_START + ATOM_VOCAB_SIZE
UNIGRAM_TOKEN_START = ATOM_TOKEN_END

# AtomType ordering matches Shared/UnigramByte/Unigram.hpp
ATOM_TYPE_LABELS = {
    0: "<ATOM_NONE>",
    1: "<ATOM_END>",
    2: "<INT>",
    3: "<FLOAT>",
    4: "<HEX>",
    5: "<BIN>",
    6: "<ID>",
    7: "<STR>",
    8: "<REGEX>",
    9: "<URL>",
    10: "<EMAIL>",
    11: "<PATH>",
    12: "<DATE>",
    13: "<TIME>",
    14: "<IP>",
    15: "<EQUATION>",
    16: "<EXPR>",
}

NUMERIC_ATOM_TYPES = {2, 3, 4, 5}


def load_vocab(vocab_path: str, total_vocab_size: int | None = None) -> tuple[dict, int]:
    """
    Load vocab.txt and create token ID -> text mapping.
    Format: token<TAB>log_probability (SentencePiece/Unigram format)
    
    Token ID Layout (GrimTokenizer):
      [0-255]   = Byte fallback tokens
      [256..]   = Atom placeholder tokens (structural)
      [..]      = Unigram vocabulary (from vocab.txt)
    """
    global ATOM_VOCAB_SIZE, ATOM_TOKEN_END, UNIGRAM_TOKEN_START
    vocab = {}
    try:
        # Byte fallback tokens (0-255)
        for i in range(256):
            if 32 <= i <= 126:
                vocab[i] = chr(i)
            else:
                vocab[i] = f"<BYTE{i:02X}>"
        
        # Unigram vocab from vocab.txt
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        unigram_count = len(lines)
        atom_vocab_size = ATOM_VOCAB_SIZE
        if total_vocab_size is not None:
            atom_vocab_size = max(0, total_vocab_size - 256 - unigram_count)
        ATOM_VOCAB_SIZE = atom_vocab_size
        ATOM_TOKEN_END = ATOM_TOKEN_START + ATOM_VOCAB_SIZE
        UNIGRAM_TOKEN_START = ATOM_TOKEN_END

        # Atom placeholders
        for i in range(atom_vocab_size):
            label = ATOM_TYPE_LABELS.get(i, f"<ATOM{i}>")
            vocab[ATOM_TOKEN_START + i] = label
        
        UNIGRAM_OFFSET = UNIGRAM_TOKEN_START
        for idx, line in enumerate(lines):
            # vocab.txt format: "token\t-logprob"
            # Split on tab and take only the token part
            if '\t' in line:
                token_text = line.split('\t')[0]
            else:
                token_text = line
            vocab[UNIGRAM_OFFSET + idx] = token_text
        
        print(f"✓ Loaded {len(vocab)} tokens total (256 bytes + {atom_vocab_size} atoms + {len(vocab) - UNIGRAM_OFFSET} unigram)")
        return vocab, atom_vocab_size
    except FileNotFoundError:
        print(f"✗ vocab.txt not found at {vocab_path}")
        print("  Creating byte fallback vocab (tokens 0-255)")
        # Fallback: byte-level vocab
        return {i: chr(i) if 32 <= i <= 126 else f"<{i:02X}>" for i in range(256)}, ATOM_VOCAB_SIZE


def atom_label(atom_type: int) -> str:
    return ATOM_TYPE_LABELS.get(atom_type, f"<ATOM{atom_type}>")


def format_numeric_value(value: float) -> str:
    if not math.isfinite(value):
        return "<NAN>"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.6g}"


def decode_tokens(tokens: list[int],
                  vocab: dict,
                  numeric_values=None,
                  numeric_mask=None) -> str:
    """
    Decode token IDs to text using vocab, honoring numeric side-channel values.
    """
    pieces = []
    numeric_count = 0
    if numeric_values is not None and numeric_mask is not None:
        numeric_count = min(len(numeric_values), len(numeric_mask))

    for i, tid in enumerate(tokens):
        if ATOM_TOKEN_START <= tid < ATOM_TOKEN_END:
            atom_type = tid - ATOM_TOKEN_START
            if atom_type in NUMERIC_ATOM_TYPES and i < numeric_count and numeric_mask[i]:
                pieces.append(format_numeric_value(numeric_values[i]))
                continue
            pieces.append(atom_label(atom_type))
            continue

        if tid in vocab:
            pieces.append(vocab[tid])
        elif 0 <= tid < ATOM_TOKEN_START:
            pieces.append(chr(tid) if 32 <= tid <= 126 else f"<{tid:02X}>")
        else:
            pieces.append(f"<UNK{tid}>")

    text = ''.join(pieces)
    text = text.replace('▁', ' ')
    text = ' '.join(text.split())
    return text


def count_atoms(tokens: list[int]) -> int:
    return sum(1 for tid in tokens if ATOM_TOKEN_START <= tid < ATOM_TOKEN_END)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_values[f])
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def compute_atom_stats(sequences: list) -> dict:
    stats = {
        'sequence_count': 0,
        'total_tokens': 0,
        'total_atoms': 0,
        'overall_atom_ratio': 0.0,
        'avg_atoms_per_seq': 0.0,
        'avg_ratio_per_seq': 0.0,
        'min_atoms': 0,
        'max_atoms': 0,
        'p50_atoms': 0.0,
        'p90_atoms': 0.0,
        'p95_atoms': 0.0,
        'p99_atoms': 0.0,
        'min_ratio': 0.0,
        'max_ratio': 0.0,
        'p50_ratio': 0.0,
        'p90_ratio': 0.0,
        'p95_ratio': 0.0,
        'p99_ratio': 0.0,
    }

    if not sequences:
        return stats

    atom_counts = []
    ratios = []
    total_atoms = 0
    total_tokens = 0

    for seq in sequences:
        token_ids = seq.get('token_ids') if isinstance(seq, dict) else seq
        if token_ids is None:
            continue
        seq_len = len(token_ids)
        atom_count = count_atoms(token_ids)
        atom_counts.append(atom_count)
        ratios.append(atom_count / seq_len if seq_len else 0.0)
        total_atoms += atom_count
        total_tokens += seq_len

    atom_counts.sort()
    ratios.sort()

    stats['sequence_count'] = len(sequences)
    stats['total_tokens'] = total_tokens
    stats['total_atoms'] = total_atoms
    stats['overall_atom_ratio'] = (total_atoms / total_tokens) if total_tokens else 0.0
    stats['avg_atoms_per_seq'] = total_atoms / len(sequences)
    stats['avg_ratio_per_seq'] = sum(ratios) / len(ratios)
    stats['min_atoms'] = atom_counts[0]
    stats['max_atoms'] = atom_counts[-1]
    stats['p50_atoms'] = percentile(atom_counts, 50)
    stats['p90_atoms'] = percentile(atom_counts, 90)
    stats['p95_atoms'] = percentile(atom_counts, 95)
    stats['p99_atoms'] = percentile(atom_counts, 99)
    stats['min_ratio'] = ratios[0]
    stats['max_ratio'] = ratios[-1]
    stats['p50_ratio'] = percentile(ratios, 50)
    stats['p90_ratio'] = percentile(ratios, 90)
    stats['p95_ratio'] = percentile(ratios, 95)
    stats['p99_ratio'] = percentile(ratios, 99)
    return stats


def compute_text_feature_stats(sequences: list) -> dict:
    """
    Compute statistics on text features across all sequences.
    
    Text feature layout per token (16 FP16 values):
      [0-3]   Category one-hot (NUMERIC, TEMPORAL, STRUCTURAL, STRING)
      [4-7]   Subtype encoding
      [8-11]  Length/magnitude features
      [12-15] Semantic features
    """
    stats = {
        'sequence_count': 0,
        'total_tokens': 0,
        'tokens_with_features': 0,
        'feature_coverage': 0.0,
        'category_counts': {
            'NUMERIC': 0,
            'TEMPORAL': 0,
            'STRUCTURAL': 0,
            'STRING': 0,
            'UNKNOWN': 0
        },
        'avg_features_per_seq': 0.0,
        'min_features_per_seq': 0,
        'max_features_per_seq': 0,
        'p50_features_per_seq': 0.0,
        'p90_features_per_seq': 0.0,
    }
    
    if not sequences:
        return stats
    
    feature_counts = []
    total_tokens = 0
    tokens_with_features = 0
    
    for seq in sequences:
        text_mask = seq.get('text_mask', [])
        text_features = seq.get('text_features', [])
        token_ids = seq.get('token_ids', [])
        
        seq_len = len(token_ids)
        total_tokens += seq_len
        
        seq_feature_count = 0
        for i, has_feature in enumerate(text_mask):
            if has_feature:
                tokens_with_features += 1
                seq_feature_count += 1
                
                # Decode category from one-hot [0-3]
                if i < len(text_features):
                    feat = text_features[i]
                    # FP16 one-hot: check which category is "hot" (nonzero)
                    # FP16 1.0 = 0x3C00
                    if len(feat) >= 4:
                        if feat[0] == 0x3C00:
                            stats['category_counts']['NUMERIC'] += 1
                        elif feat[1] == 0x3C00:
                            stats['category_counts']['TEMPORAL'] += 1
                        elif feat[2] == 0x3C00:
                            stats['category_counts']['STRUCTURAL'] += 1
                        elif feat[3] == 0x3C00:
                            stats['category_counts']['STRING'] += 1
                        else:
                            stats['category_counts']['UNKNOWN'] += 1
        
        feature_counts.append(seq_feature_count)
    
    feature_counts.sort()
    
    stats['sequence_count'] = len(sequences)
    stats['total_tokens'] = total_tokens
    stats['tokens_with_features'] = tokens_with_features
    stats['feature_coverage'] = (tokens_with_features / total_tokens) if total_tokens else 0.0
    stats['avg_features_per_seq'] = tokens_with_features / len(sequences) if sequences else 0.0
    stats['min_features_per_seq'] = feature_counts[0] if feature_counts else 0
    stats['max_features_per_seq'] = feature_counts[-1] if feature_counts else 0
    stats['p50_features_per_seq'] = percentile(feature_counts, 50)
    stats['p90_features_per_seq'] = percentile(feature_counts, 90)
    
    return stats


# GRMT v4 constants
GRMT_MAGIC = 0x474D5254
GRMT_VERSION = 4
TEXT_FEATURE_DIM = 16  # 16 FP16 values per token


def read_grmt_header(data_path: str) -> tuple[int, int, int, int]:
    with open(data_path, 'rb') as f:
        header = f.read(16)
        if len(header) < 16:
            raise ValueError("GRMT header too small")
        magic, version, num_sequences, vocab_size = struct.unpack('<IIII', header)
    return magic, version, num_sequences, vocab_size


def load_training_sequences(data_path: str) -> list:
    """
    Load sequences from .grmt binary file (GRMT v4).
    
    GRMT v4 format per sequence:
      - seq_len (uint32)
      - token_ids (seq_len × uint32)
      - numeric_values (seq_len × float32)
      - numeric_mask (seq_len × uint8)
      - text_features (seq_len × TEXT_FEATURE_DIM × uint16)  [NEW in v4]
      - text_mask (seq_len × uint8)  [NEW in v4]
    """
    sequences = []
    
    print(f"Reading GRMT data from {data_path}...")
    with open(data_path, 'rb') as f:
        header = f.read(16)
        if len(header) < 16:
            print("✗ File too small for GRMT header")
            return sequences

        magic, version, num_sequences, vocab_size = struct.unpack('<IIII', header)
        if magic != GRMT_MAGIC:
            print(f"✗ Invalid GRMT magic: 0x{magic:08X}")
            return sequences

        if version != GRMT_VERSION:
            print(f"✗ Unsupported GRMT version {version} (expected {GRMT_VERSION})")
            return sequences

        print(f"GRMT version: {version}")
        print(f"Sequences: {num_sequences}")
        print(f"Vocab size: {vocab_size}")

        for _ in range(num_sequences):
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                break
            seq_len = struct.unpack('<I', len_bytes)[0]
            if seq_len == 0:
                sequences.append({
                    'token_ids': [],
                    'numeric_values': [],
                    'numeric_mask': [],
                    'text_features': [],
                    'text_mask': []
                })
                continue

            # Token IDs
            token_bytes = f.read(seq_len * 4)
            if len(token_bytes) < seq_len * 4:
                break
            tokens = list(struct.unpack('<' + 'I' * seq_len, token_bytes))

            # Numeric values (float32)
            numeric_bytes = f.read(seq_len * 4)
            if len(numeric_bytes) < seq_len * 4:
                break
            numeric_values = list(struct.unpack('<' + 'f' * seq_len, numeric_bytes))

            # Numeric mask
            mask_bytes = f.read(seq_len)
            if len(mask_bytes) < seq_len:
                break
            numeric_mask = list(mask_bytes)

            # Text features (FP16 × TEXT_FEATURE_DIM per token)
            text_feature_bytes = f.read(seq_len * TEXT_FEATURE_DIM * 2)
            if len(text_feature_bytes) < seq_len * TEXT_FEATURE_DIM * 2:
                break
            # Parse as uint16 (FP16 binary representation)
            text_features_flat = list(struct.unpack('<' + 'H' * (seq_len * TEXT_FEATURE_DIM), text_feature_bytes))
            # Reshape to [seq_len][TEXT_FEATURE_DIM]
            text_features = [text_features_flat[i * TEXT_FEATURE_DIM:(i + 1) * TEXT_FEATURE_DIM] for i in range(seq_len)]

            # Text mask
            text_mask_bytes = f.read(seq_len)
            if len(text_mask_bytes) < seq_len:
                break
            text_mask = list(text_mask_bytes)

            sequences.append({
                'token_ids': tokens,
                'numeric_values': numeric_values,
                'numeric_mask': numeric_mask,
                'text_features': text_features,
                'text_mask': text_mask
            })

    print(f"✓ Loaded {len(sequences)} sequences")
    return sequences


def classify_sequence(text: str) -> tuple[str, dict]:
    """
    Classify text using same patterns as train_gpu.cu classifySequence().
    Returns (class_name, score_breakdown)
    """
    if not text or len(text) < 10:
        return ("mixed_junk", {})
    
    text_lower = text.lower()
    scores = {
        'boilerplate': 0,
        'documentation': 0,
        'prose': 0,
        'code': 0,
        'junk': 0
    }
    
    # === BOILERPLATE PATTERNS ===
    boilerplate = [
        "find us on", "follow us", "search this site", "submit search",
        "copyright", "all rights reserved", "privacy policy", "terms of",
        "contact us", "sign up", "log in", "sign in", "subscribe",
        "newsletter", "share on", "tweet", "facebook", "instagram",
        "skip to content", "skip to main", "navigation", "breadcrumb",
        "menu", "footer", "header", "sidebar", "retrieved from",
        "category :", "categories:", "tags:", "related posts",
        "previous post", "next post", "read more", "click here"
    ]
    for pat in boilerplate:
        if pat in text_lower:
            scores['boilerplate'] += 2
    
    # === DOCUMENTATION PATTERNS ===
    doc = [
        "parameters", "returns", "example", "usage:", "syntax:",
        "arguments", "description:", "note:", "warning:", "deprecated",
        "see also", "api", "reference", "documentation", "method",
        "function", "class", "module", "import", "export"
    ]
    for pat in doc:
        if pat in text_lower:
            scores['documentation'] += 2
    
    # === PROSE INDICATORS ===
    # Punctuation density
    sentence_ends = text.count('.') + text.count('!') + text.count('?')
    commas = text.count(',')
    punct_ratio = (sentence_ends + commas) / max(len(text), 1)
    
    if 0.02 < punct_ratio < 0.08:
        scores['prose'] += 5
    if sentence_ends > 3:
        scores['prose'] += 3
    
    # === CODE PATTERNS ===
    code_keywords = [
        "def ", "class ", "function ", "return ", "if (", "for (",
        "while (", "switch (", "case ", "import ", "from ",
        "#include", "using namespace", "public:", "private:",
        "const ", "let ", "var ", "async ", "await "
    ]
    for pat in code_keywords:
        if pat in text:
            scores['code'] += 2
    
    # Code punctuation
    braces = text.count('{') + text.count('}')
    semicolons = text.count(';')
    if braces > 4:
        scores['code'] += 3
    if semicolons > 5:
        scores['code'] += 3
    
    # === JUNK PATTERNS ===
    junk = [
        "}} ", "{{ ", " ... ", "...", " <url>", "<url> ",
        " ip }}", "work to do", "peer-to-peer university"
    ]
    for pat in junk:
        if pat in text_lower:
            scores['junk'] += 3
    
    # Repetitive words
    words = [w for w in text_lower.split() if len(w) >= 3]
    word_counts = defaultdict(int)
    for w in words:
        word_counts[w] += 1
    
    repetitive = sum(1 for count in word_counts.values() if count > 5)
    if repetitive > 3:
        scores['junk'] += 5
    
    # Line length variance
    lines = text.split('\n')
    if len(lines) > 1:
        avg_line_len = sum(len(line) for line in lines) / len(lines)
        if avg_line_len < 20.0 or avg_line_len > 500.0:
            scores['junk'] += 2
    
    # === DETERMINE WINNER ===
    max_score = max(scores.values())
    if max_score == 0 or scores['junk'] >= max_score:
        return ("mixed_junk", scores)
    
    winner = max(scores.keys(), key=lambda k: scores[k])
    
    if winner == 'junk':
        return ("mixed_junk", scores)
    elif winner == 'boilerplate':
        return ("boilerplate/nav", scores)
    elif winner == 'documentation':
        return ("documentation", scores)
    elif winner == 'prose':
        return ("prose", scores)
    elif winner == 'code':
        return ("code", scores)
    else:
        return ("mixed_junk", scores)


def main():
    parser = argparse.ArgumentParser(
        description='Sample and classify training data sequences'
    )
    parser.add_argument(
        '--data',
        default='D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt',
        help='Path to .grmt training data file'
    )
    parser.add_argument(
        '--vocab',
        default='D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.txt',
        help='Path to vocab.txt file'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=20,
        help='Number of sequences to sample (default: 20)'
    )
    parser.add_argument(
        '--output',
        default='training_audit_report.txt',
        help='Output file path (default: training_audit_report.txt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Load vocab (use GRMT header to derive atom range)
    try:
        _, _, _, grmt_vocab_size = read_grmt_header(args.data)
    except Exception as exc:
        print(f"✗ Failed to read GRMT header: {exc}")
        sys.exit(1)
    vocab, _ = load_vocab(args.vocab, grmt_vocab_size)
    
    # Load training data
    sequences = load_training_sequences(args.data)
    
    if len(sequences) == 0:
        print("✗ No sequences found in training data!")
        sys.exit(1)

    atom_stats = compute_atom_stats(sequences)
    text_feature_stats = compute_text_feature_stats(sequences)
    
    # Sample random sequences
    sample_count = min(args.count, len(sequences))
    sampled_indices = random.sample(range(len(sequences)), sample_count)
    sampled_indices.sort()  # Keep in order for easier tracking
    
    print(f"\nSampling {sample_count} sequences from {len(sequences)} total...")
    
    # Analyze samples
    class_counts = defaultdict(int)
    samples_data = []
    
    for idx in sampled_indices:
        seq = sequences[idx]
        tokens = seq['token_ids']
        numeric_values = seq['numeric_values']
        numeric_mask = seq['numeric_mask']
        text = decode_tokens(tokens, vocab, numeric_values, numeric_mask)
        seq_class, scores = classify_sequence(text)
        
        class_counts[seq_class] += 1
        samples_data.append({
            'index': idx,
            'tokens': tokens,
            'text': text,
            'class': seq_class,
            'scores': scores
        })
    
    # Write report
    print(f"\nWriting report to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("GRIM TRAINING DATA QUALITY AUDIT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: {args.data}\n")
        f.write(f"Vocab file: {args.vocab}\n")
        f.write(f"GRMT format version: {GRMT_VERSION}\n")
        f.write("Numeric side-channel: decoded\n")
        f.write(f"Text feature dimension: {TEXT_FEATURE_DIM} (FP16)\n")
        f.write(f"Total sequences: {len(sequences)}\n")
        f.write(f"Sampled: {sample_count}\n")
        if args.seed is not None:
            f.write(f"Random seed: {args.seed}\n")
        f.write("=" * 80 + "\n\n")
        
        # Classification summary
        f.write("CLASSIFICATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        for cls in sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True):
            count = class_counts[cls]
            pct = (count / sample_count) * 100.0
            
            # Get expected weight from train_gpu.cu
            weight = 1.0
            if cls == "boilerplate/nav":
                weight = 0.4
            elif cls == "mixed_junk":
                weight = 0.3
            
            f.write(f"  {cls:20s}: {count:3d} ({pct:5.1f}%)  [weight: {weight:.1f}×]\n")
        
        f.write("\n")

        # Atom statistics
        f.write("ATOM STATISTICS (ALL SEQUENCES)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Atom token range: [{ATOM_TOKEN_START}, {ATOM_TOKEN_END})\n")
        f.write(f"Total sequences: {atom_stats['sequence_count']}\n")
        f.write(f"Total tokens: {atom_stats['total_tokens']}\n")
        f.write(f"Total atoms: {atom_stats['total_atoms']}\n")
        f.write(f"Overall atom ratio: {atom_stats['overall_atom_ratio']:.6f}\n")
        f.write(f"Avg atoms per seq: {atom_stats['avg_atoms_per_seq']:.2f}\n")
        f.write(f"Avg atom ratio per seq: {atom_stats['avg_ratio_per_seq']:.6f}\n")
        f.write("Atoms per sequence (count): "
                f"min={atom_stats['min_atoms']} "
                f"p50={atom_stats['p50_atoms']:.2f} "
                f"p90={atom_stats['p90_atoms']:.2f} "
                f"p95={atom_stats['p95_atoms']:.2f} "
                f"p99={atom_stats['p99_atoms']:.2f} "
                f"max={atom_stats['max_atoms']}\n")
        f.write("Atom ratio per sequence: "
                f"min={atom_stats['min_ratio']:.6f} "
                f"p50={atom_stats['p50_ratio']:.6f} "
                f"p90={atom_stats['p90_ratio']:.6f} "
                f"p95={atom_stats['p95_ratio']:.6f} "
                f"p99={atom_stats['p99_ratio']:.6f} "
                f"max={atom_stats['max_ratio']:.6f}\n")
        f.write("\n")

        # Text feature statistics (GRMT v4)
        f.write("TEXT FEATURE STATISTICS (GRMT v4)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Feature dimension: {TEXT_FEATURE_DIM} (FP16 per token)\n")
        f.write(f"Total tokens: {text_feature_stats['total_tokens']}\n")
        f.write(f"Tokens with features: {text_feature_stats['tokens_with_features']}\n")
        f.write(f"Feature coverage: {text_feature_stats['feature_coverage']:.4%}\n")
        f.write(f"Avg features per seq: {text_feature_stats['avg_features_per_seq']:.2f}\n")
        f.write(f"Features per seq: "
                f"min={text_feature_stats['min_features_per_seq']} "
                f"p50={text_feature_stats['p50_features_per_seq']:.1f} "
                f"p90={text_feature_stats['p90_features_per_seq']:.1f} "
                f"max={text_feature_stats['max_features_per_seq']}\n")
        f.write("Category distribution:\n")
        total_cats = sum(text_feature_stats['category_counts'].values())
        for cat, count in sorted(text_feature_stats['category_counts'].items(), key=lambda x: -x[1]):
            pct = (count / total_cats * 100) if total_cats else 0.0
            f.write(f"  {cat:12s}: {count:8d} ({pct:5.1f}%)\n")
        f.write("\n")
        
        # Individual samples
        f.write("INDIVIDUAL SAMPLES\n")
        f.write("=" * 80 + "\n")
        
        for i, sample in enumerate(samples_data, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SAMPLE {i}/{sample_count} - Sequence #{sample['index']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Tokens: {len(sample['tokens'])}\n")
            f.write(f"Classification: [{sample['class']}]\n")
            f.write(f"Score breakdown: {dict(sample['scores'])}\n")
            f.write(f"\nDecoded text (first 1500 chars):\n")
            f.write("-" * 80 + "\n")
            
            preview = sample['text'][:1500]
            f.write(preview)
            
            if len(sample['text']) > 1500:
                f.write(f"\n... (truncated, total {len(sample['text'])} chars)\n")
            
            f.write("\n" + "-" * 80 + "\n")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"✓ Analyzed {sample_count} sequences")
    print("\nClassification breakdown:")
    for cls in sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True):
        count = class_counts[cls]
        pct = (count / sample_count) * 100.0
        weight = 1.0
        if cls == "boilerplate/nav":
            weight = 0.4
        elif cls == "mixed_junk":
            weight = 0.3
        print(f"  {cls:20s}: {count:3d} ({pct:5.1f}%)  [loss weight: {weight:.1f}×]")

    print("\nAtom stats (all sequences):")
    print(f"  atom_token_range: [{ATOM_TOKEN_START}, {ATOM_TOKEN_END})")
    print(f"  total_atoms: {atom_stats['total_atoms']} total_tokens: {atom_stats['total_tokens']}")
    print(f"  overall_atom_ratio: {atom_stats['overall_atom_ratio']:.6f}")
    print(f"  avg_atoms_per_seq: {atom_stats['avg_atoms_per_seq']:.2f}")
    print(f"  atoms_per_seq p50: {atom_stats['p50_atoms']:.2f} p90: {atom_stats['p90_atoms']:.2f} max: {atom_stats['max_atoms']}")
    print(f"  atom_ratio p50: {atom_stats['p50_ratio']:.6f} p90: {atom_stats['p90_ratio']:.6f} max: {atom_stats['max_ratio']:.6f}")

    print("\nText feature stats (GRMT v4):")
    print(f"  feature_dim: {TEXT_FEATURE_DIM} (FP16 per token)")
    print(f"  tokens_with_features: {text_feature_stats['tokens_with_features']} / {text_feature_stats['total_tokens']}")
    print(f"  feature_coverage: {text_feature_stats['feature_coverage']:.4%}")
    print(f"  features_per_seq p50: {text_feature_stats['p50_features_per_seq']:.1f} p90: {text_feature_stats['p90_features_per_seq']:.1f} max: {text_feature_stats['max_features_per_seq']}")
    total_cats = sum(text_feature_stats['category_counts'].values())
    if total_cats > 0:
        print("  category_breakdown:")
        for cat, count in sorted(text_feature_stats['category_counts'].items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total_cats * 100
                print(f"    {cat}: {count} ({pct:.1f}%)")
    
    print(f"\n✓ Full report written to: {args.output}")


if __name__ == '__main__':
    main()
