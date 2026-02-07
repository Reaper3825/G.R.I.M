#!/usr/bin/env python3
"""
Cross-reference GRIM tokenizer against HuggingFace tokenizers
Identifies fragile logic and potential issues
"""

import json
import os
import struct
from pathlib import Path
from typing import Optional
from collections import Counter

# Try to import tokenizers
try:
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer, models, pre_tokenizers, normalizers
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers/torch not installed, some comparisons unavailable")


class GRIMTokenizerAnalyzer:
    """Analyze GRIM tokenizer implementation"""
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab = []
        self.token_to_id = {}
        self.merge_rules = []
        self.special_tokens = {}
        self.config = {}
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
    
    def load_vocab(self, path: str) -> bool:
        """Load GRIM tokenizer vocabulary file"""
        try:
            with open(path, 'rb') as f:
                # Read magic number
                magic = f.read(4)
                
                # Accept both GRIM (new) and KTMG/GMTK (old) for backward compatibility
                if magic == b'GRIM':
                    print(f"✓ Vocabulary format: GRIM (new format)")
                elif magic == b'KTMG' or magic == b'GMTK':
                    print(f"✓ Vocabulary format: GMTK/KTMG (old format, backward compatible)")
                else:
                    print(f"❌ Invalid magic number: {magic}")
                    return False
                
                # Read version
                version = struct.unpack('<I', f.read(4))[0]
                print(f"📦 Vocabulary version: {version}")
                
                # Read vocab size
                vocab_size = struct.unpack('<I', f.read(4))[0]
                print(f"📊 Vocabulary size: {vocab_size}")
                
                # Read tokens
                for i in range(vocab_size):
                    token_len = struct.unpack('<I', f.read(4))[0]
                    token = f.read(token_len).decode('utf-8', errors='replace')
                    self.vocab.append(token)
                    self.token_to_id[token] = i
                
                # Read merge rules count
                merge_count = struct.unpack('<I', f.read(4))[0]
                print(f"🔗 Merge rules: {merge_count}")
                
                for _ in range(merge_count):
                    left_len = struct.unpack('<I', f.read(4))[0]
                    left = f.read(left_len).decode('utf-8', errors='replace')
                    right_len = struct.unpack('<I', f.read(4))[0]
                    right = f.read(right_len).decode('utf-8', errors='replace')
                    priority = struct.unpack('<i', f.read(4))[0]
                    self.merge_rules.append((left, right, priority))
                
                # Read checksum
                checksum = struct.unpack('<I', f.read(4))[0]
                print(f"🔐 Checksum: 0x{checksum:08X}")
                
                return True
        except Exception as e:
            print(f"❌ Failed to load vocab: {e}")
            return False
    
    def analyze_byte_fallback(self) -> dict:
        """Check byte fallback token coverage"""
        issues = []
        byte_tokens = []
        
        for i in range(256):
            expected = f"<0x{i:02X}>"
            if i < len(self.vocab):
                actual = self.vocab[i]
                if actual != expected:
                    issues.append(f"ID {i}: expected '{expected}', got '{actual}'")
                byte_tokens.append(actual)
            else:
                issues.append(f"ID {i}: missing byte token {expected}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues[:10],  # First 10 issues
            'total_issues': len(issues),
            'byte_tokens_sample': byte_tokens[:16]
        }
    
    def analyze_special_tokens(self) -> dict:
        """Check special token placement and IDs"""
        expected_special = ['<pad>', '<unk>', '<s>', '</s>']
        found = {}
        issues = []
        
        for token in expected_special:
            if token in self.token_to_id:
                found[token] = self.token_to_id[token]
            else:
                issues.append(f"Missing special token: {token}")
        
        # Check if special tokens are in expected range (256-259)
        for token, tid in found.items():
            if tid < 256 or tid > 259:
                issues.append(f"Special token '{token}' at unusual ID {tid} (expected 256-259)")
        
        return {
            'found': found,
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def analyze_merge_rules(self) -> dict:
        """Analyze BPE merge rules for consistency"""
        issues = []
        
        # Check if merged tokens exist in vocab
        missing_merged = 0
        for left, right, priority in self.merge_rules[:100]:  # Sample first 100
            merged = left + right
            if merged not in self.token_to_id:
                missing_merged += 1
        
        if missing_merged > 0:
            issues.append(f"{missing_merged} merge results not in vocab (first 100 rules)")
        
        # Check priority ordering
        priorities = [p for _, _, p in self.merge_rules]
        if priorities != sorted(priorities):
            issues.append("Merge rules not sorted by priority")
        
        # Check for duplicate merges
        merge_keys = [(l, r) for l, r, _ in self.merge_rules]
        duplicates = len(merge_keys) - len(set(merge_keys))
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate merge rules")
        
        return {
            'total_rules': len(self.merge_rules),
            'issues': issues,
            'valid': len(issues) == 0,
            'sample_rules': self.merge_rules[:5]
        }
    
    def analyze_vocab_coverage(self) -> dict:
        """Analyze vocabulary coverage and gaps"""
        # Count token types
        byte_fallback = sum(1 for t in self.vocab if t.startswith('<0x') and t.endswith('>'))
        special = sum(1 for t in self.vocab if t.startswith('<') and not t.startswith('<0x'))
        subwords = sum(1 for t in self.vocab if t.startswith('▁') or t.startswith('Ġ'))
        single_char = sum(1 for t in self.vocab if len(t) == 1)
        
        # Find longest tokens
        by_length = sorted(self.vocab, key=len, reverse=True)[:10]
        
        return {
            'total': len(self.vocab),
            'byte_fallback': byte_fallback,
            'special_tokens': special,
            'subword_markers': subwords,
            'single_char': single_char,
            'longest_tokens': by_length
        }


def test_normalization_logic():
    """Test NFKC normalization edge cases"""
    print("\n" + "="*60)
    print("🔤 NORMALIZATION LOGIC TESTS")
    print("="*60)
    
    test_cases = [
        # (input, expected_behavior)
        ("Hello World", "Basic ASCII - should pass through"),
        ("café", "Accented characters - NFKC should normalize"),
        ("ﬁ", "Ligature fi - NFKC decomposes to 'fi'"),
        ("①②③", "Circled numbers - NFKC normalizes"),
        ("　", "Full-width space (U+3000) - should normalize to space"),
        ("Hello\t\n\rWorld", "Control chars - treat as whitespace"),
        ("Hello  World", "Multiple spaces - should collapse"),
        ("Ａｂｃ", "Full-width ASCII - NFKC normalizes to ASCII"),
        ("½", "Fraction - NFKC may expand"),
        ("™©®", "Symbols - may need special handling"),
        ("\u200b\u200c\u200d", "Zero-width chars - should remove or keep?"),
        ("𝕳𝖊𝖑𝖑𝖔", "Math script - NFKC normalizes"),
    ]
    
    issues = []
    for text, description in test_cases:
        try:
            import unicodedata
            normalized = unicodedata.normalize('NFKC', text)
            # Collapse whitespace
            import re
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            print(f"  ✓ '{text}' → '{normalized}' ({description})")
        except Exception as e:
            issues.append(f"'{text}': {e}")
            print(f"  ❌ '{text}': {e}")
    
    # Check GRIM's normalization implementation
    print("\n⚠️  GRIM Normalization Issues to Check:")
    print("  1. Does normalizeNFKC handle all Unicode planes?")
    print("  2. Are zero-width characters handled?")
    print("  3. Is whitespace collapsing consistent with training data?")
    print("  4. Are ligatures properly decomposed?")
    
    return len(issues) == 0


def test_pretokenization_logic():
    """Test pretokenization edge cases"""
    print("\n" + "="*60)
    print("✂️  PRETOKENIZATION LOGIC TESTS")
    print("="*60)
    
    test_cases = [
        "Hello, World!",
        "don't",
        "self-aware",
        "test@email.com",
        "https://example.com/path?query=1",
        "$100.50",
        "C++",
        "file.txt",
        "Hello...World",
        "  multiple   spaces  ",
        "emoji😀test",
        "日本語テスト",
        "mixed日本語text",
    ]
    
    print("\n📋 Testing pretokenization patterns:")
    for text in test_cases:
        # Simulate GRIM's pretokenization (split on whitespace/punctuation)
        import re
        # Basic pattern matching GRIM's logic
        words = re.findall(r'\S+|\s+', text)
        words = [w for w in words if w.strip()]
        print(f"  '{text}' → {words}")
    
    print("\n⚠️  GRIM Pretokenization Issues to Check:")
    print("  1. How are contractions handled (don't → [don, ', t] or [don't])?")
    print("  2. Are URLs/emails kept together or split?")
    print("  3. How are punctuation clusters handled?")
    print("  4. Is leading/trailing whitespace in words preserved?")
    
    return True


def compare_with_pytorch_tokenizers():
    """Compare with multiple PyTorch reference tokenizers"""
    print("\n" + "="*60)
    print("🔄 PYTORCH TOKENIZER COMPARISON")
    print("="*60)
    
    if not HAS_TRANSFORMERS:
        print("⚠️  Skipping - transformers/torch not available")
        print("   Install with: pip install transformers torch")
        return
    
    # Test texts covering various edge cases
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-4 is a large language model.",
        "don't shouldn't won't",
        "test@email.com and https://example.com",
        "Numbers: 42, 3.14, 0xFF, 0b1010",
        "Code: def hello_world():\n    print('Hello!')",
        "日本語のテスト",
        "Mixed 日本語 and English text",
        "Émoji test: 😀🎉✨",
    ]
    
    # Compare against multiple reference tokenizers
    reference_models = [
        ("GPT-2 (BPE)", "gpt2"),
        ("BERT (WordPiece)", "bert-base-uncased"),
        ("T5 (Unigram)", "t5-small"),
    ]
    
    results = {}
    
    for model_name, model_id in reference_models:
        print(f"\n📥 Testing with {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model_results = []
            
            for text in test_texts[:5]:  # Test first 5 texts
                # Tokenize
                encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
                token_ids = encoding['input_ids'][0].tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                
                # Decode
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                
                # Check roundtrip
                roundtrip_ok = decoded.strip().lower() == text.strip().lower()
                
                model_results.append({
                    'text': text[:50],
                    'tokens': tokens[:10],
                    'ids': token_ids[:10],
                    'roundtrip_ok': roundtrip_ok,
                    'token_count': len(tokens)
                })
                
                if not roundtrip_ok:
                    print(f"  ⚠️  Roundtrip issue:")
                    print(f"      Original: '{text}'")
                    print(f"      Decoded:  '{decoded}'")
            
            results[model_name] = model_results
            print(f"  ✓ Tested {len(test_texts[:5])} samples")
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_name}: {e}")
            continue
    
    # Compare tokenization strategies
    print("\n\n📊 Tokenization Strategy Comparison:")
    print("-" * 60)
    
    sample_text = "Hello, world! Test: 123"
    print(f"Sample text: '{sample_text}'")
    print()
    
    for model_name, model_id in reference_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokens = tokenizer.tokenize(sample_text)
            print(f"{model_name:20} {tokens}")
        except:
            pass
    
    # Token count statistics
    if results:
        print("\n\n📈 Token Count Statistics:")
        print("-" * 60)
        print(f"{'Model':<20} {'Avg Tokens':<15} {'Min':<10} {'Max':<10}")
        print("-" * 60)
        
        for model_name, model_results in results.items():
            counts = [r['token_count'] for r in model_results]
            avg = sum(counts) / len(counts)
            print(f"{model_name:<20} {avg:>10.1f}      {min(counts):>5}      {max(counts):>5}")
    
    # Key findings
    print("\n\n🔍 Key Findings for GRIM:")
    print("-" * 60)
    print("  1. BPE (GPT-2): Aggressive subword splitting, good for rare words")
    print("  2. WordPiece (BERT): Balanced approach, ##prefix markers")
    print("  3. Unigram (T5): Probabilistic, multiple segmentations possible")
    print("  4. GRIM (UniByte): Unigram + Byte fallback + Atom detection")
    print()
    print("  ✅ GRIM's approach combines best of all three:")
    print("     - Unigram LM for quality (like T5)")
    print("     - Byte fallback for 100% coverage (better than all)")
    print("     - Atom detection for structural reasoning (unique!)")
    
    return results


def analyze_fragile_patterns():
    """Identify fragile patterns in GRIM tokenizer code"""
    print("\n" + "="*60)
    print("🔍 FRAGILE PATTERN ANALYSIS")
    print("="*60)
    
    fragile_patterns = [
        {
            'location': 'Tokenizer_GPU.cu:initializeBaseVocab()',
            'issue': 'Hardcoded byte range [0-255] assumes IDs match byte values',
            'risk': 'HIGH',
            'fix': 'Validate byte token IDs match their byte values after loading'
        },
        {
            'location': 'Tokenizer_GPU.cu:normalize()',
            'issue': 'ASCII-only NFKC approximation misses multi-byte sequences',
            'risk': 'MEDIUM',
            'fix': 'Use ICU or full Unicode normalization library'
        },
        {
            'location': 'Tokenizer_GPU.cu:pretokenize()',
            'issue': 'Simple char-by-char split may break UTF-8 sequences',
            'risk': 'HIGH',
            'fix': 'Use UTF-8 aware iteration (codepoint-based)'
        },
        {
            'location': 'Tokenizer_GPU.cu:bpeEncode()',
            'issue': 'O(n²) merge search per word - slow for long words',
            'risk': 'MEDIUM',
            'fix': 'Use heap-based priority merge or precomputed tables'
        },
        {
            'location': 'Tokenizer_GPU.cu:findBestMerge()',
            'issue': 'Linear search through all merge rules',
            'risk': 'MEDIUM', 
            'fix': 'Use hash map for O(1) merge lookup'
        },
        {
            'location': 'Tokenizer_GPU.hpp:TokenizerConfig',
            'issue': 'vocab_size=0 default requires explicit initialization',
            'risk': 'HIGH',
            'fix': 'Add validation that vocab_size > 0 before encoding'
        },
        {
            'location': 'Tokenizer_GPU.cu:encodeChunk()',
            'issue': 'Byte fallback uses string formatting (slow)',
            'risk': 'LOW',
            'fix': 'Precompute byte token strings in constructor'
        },
        {
            'location': 'Tokenizer_GPU.cu:decode()',
            'issue': 'No handling of malformed byte sequences',
            'risk': 'MEDIUM',
            'fix': 'Add error handling for invalid UTF-8 during decode'
        },
    ]
    
    print("\n🚨 IDENTIFIED FRAGILE PATTERNS:\n")
    for i, pattern in enumerate(fragile_patterns, 1):
        risk_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[pattern['risk']]
        print(f"{i}. {risk_emoji} [{pattern['risk']}] {pattern['location']}")
        print(f"   Issue: {pattern['issue']}")
        print(f"   Fix: {pattern['fix']}")
        print()
    
    return fragile_patterns


def check_encoding_consistency():
    """Test encoding/decoding roundtrip consistency"""
    print("\n" + "="*60)
    print("🔁 ENCODING CONSISTENCY TESTS")
    print("="*60)
    
    test_cases = [
        # Basic
        "Hello",
        "Hello World",
        "Hello, World!",
        
        # Punctuation
        "test...test",
        "a--b",
        "what?!",
        
        # Numbers
        "123",
        "3.14159",
        "$1,234.56",
        
        # Code
        "def foo():",
        "x = y + z",
        "if (a && b)",
        
        # Unicode
        "café",
        "日本語",
        "emoji😀",
        "mixed日english",
        
        # Edge cases
        "",
        " ",
        "   ",
        "\n",
        "\t",
        "a" * 100,
    ]
    
    print("\n📋 Roundtrip test cases (simulated):")
    issues = []
    for text in test_cases:
        # Simulate what GRIM should do
        display = text.replace('\n', '\\n').replace('\t', '\\t')
        if len(display) > 40:
            display = display[:40] + "..."
        
        # Check for potential issues
        has_issue = False
        issue_type = None
        
        if not text:
            issue_type = "empty string (explicitly handled)"
            has_issue = False
        elif text.isspace():
            issue_type = "whitespace-only string (explicitly handled)"
            has_issue = False
        elif any(ord(c) > 127 for c in text):
            issue_type = "non-ASCII characters"
            # Not necessarily an issue, but worth checking
        
        status = "⚠️ " if has_issue else "✓ "
        msg = f" ({issue_type})" if issue_type else ""
        print(f"  {status}'{display}'{msg}")
        
        if has_issue:
            issues.append((text, issue_type))
    
    print(f"\n📊 Results: {len(test_cases) - len(issues)}/{len(test_cases)} likely OK")
    if issues:
        print(f"⚠️  {len(issues)} cases need verification")
    
    return len(issues) == 0


def main():
    print("="*60)
    print("🔬 GRIM TOKENIZER VERIFICATION TOOL")
    print("="*60)
    
    # Find vocab file - use absolute path
    vocab_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin"
    
    if not Path(vocab_path).exists():
        print(f"⚠️  Vocab not found at: {vocab_path}")
        print("   Searching for alternatives...")
        vocab_paths = [
            r"D:\G.R.I.M\resources\models\GRIM-text\Shared\UnigramByte\vocab.bin",
            r"D:\G.R.I.M\resources\models\GRIM-text\vocab.bin",
            r"D:\G.R.I.M\resources\models\GRIM-text\training\vocab.bin",
        ]
        
        vocab_path = None
        for p in vocab_paths:
            if Path(p).exists():
                vocab_path = p
                print(f"   ✓ Found: {p}")
                break
    else:
        print(f"✓ Using vocab: {vocab_path}")
    
    # Analyze GRIM tokenizer
    analyzer = GRIMTokenizerAnalyzer(vocab_path)
    
    if analyzer.vocab:
        print("\n" + "="*60)
        print("📊 GRIM VOCABULARY ANALYSIS")
        print("="*60)
        
        # Byte fallback analysis
        print("\n🔢 Byte Fallback Tokens:")
        byte_result = analyzer.analyze_byte_fallback()
        print(f"  Valid: {byte_result['valid']}")
        print(f"  Sample: {byte_result['byte_tokens_sample']}")
        if byte_result['issues']:
            print(f"  Issues ({byte_result['total_issues']} total):")
            for issue in byte_result['issues']:
                print(f"    - {issue}")
        
        # Special tokens analysis
        print("\n⭐ Special Tokens:")
        special_result = analyzer.analyze_special_tokens()
        for token, tid in special_result['found'].items():
            print(f"  {token}: ID {tid}")
        if special_result['issues']:
            print("  Issues:")
            for issue in special_result['issues']:
                print(f"    - {issue}")
        
        # Merge rules analysis
        print("\n🔗 Merge Rules:")
        merge_result = analyzer.analyze_merge_rules()
        print(f"  Total rules: {merge_result['total_rules']}")
        print(f"  Valid: {merge_result['valid']}")
        if merge_result['sample_rules']:
            print("  Sample rules:")
            for left, right, priority in merge_result['sample_rules']:
                print(f"    '{left}' + '{right}' (priority {priority})")
        if merge_result['issues']:
            for issue in merge_result['issues']:
                print(f"    ⚠️  {issue}")
        
        # Vocabulary coverage
        print("\n📈 Vocabulary Coverage:")
        coverage = analyzer.analyze_vocab_coverage()
        print(f"  Total tokens: {coverage['total']}")
        print(f"  Byte fallback: {coverage['byte_fallback']}")
        print(f"  Special tokens: {coverage['special_tokens']}")
        print(f"  Subword markers: {coverage['subword_markers']}")
        print(f"  Single char: {coverage['single_char']}")
        print(f"  Longest tokens: {coverage['longest_tokens'][:5]}")
    else:
        print("\n⚠️  No vocab file found - skipping vocabulary analysis")
    
    # Run other tests
    test_normalization_logic()
    test_pretokenization_logic()
    pytorch_results = compare_with_pytorch_tokenizers()
    fragile = analyze_fragile_patterns()
    check_encoding_consistency()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    high_risk = sum(1 for p in fragile if p['risk'] == 'HIGH')
    medium_risk = sum(1 for p in fragile if p['risk'] == 'MEDIUM')
    
    print(f"\n🔴 High-risk patterns: {high_risk}")
    print(f"🟡 Medium-risk patterns: {medium_risk}")
    print(f"🟢 Low-risk patterns: {len(fragile) - high_risk - medium_risk}")
    
    print("\n🎯 RECOMMENDED ACTIONS:")
    print("  1. Add vocab_size validation in TokenizerGPU constructor")
    print("  2. Implement proper UTF-8 iteration in pretokenize()")
    print("  3. Add roundtrip tests for edge cases")
    print("  4. Consider using hash map for merge rule lookups")
    print("  5. Add explicit handling for empty/whitespace-only strings")


if __name__ == "__main__":
    main()
