"""
Training Data Quality Inspector
Analyzes GRMT training data to measure actual diversity and quality metrics.
"""

import struct
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import json

class GRMTDataInspector:
    def __init__(self, grmt_path: str):
        self.grmt_path = Path(grmt_path)
        self.sequences = []
        self.token_sequences = []
        
    def load_grmt(self):
        """Load and parse GRMT binary format."""
        print(f"Loading GRMT file: {self.grmt_path}")
        
        with open(self.grmt_path, 'rb') as f:
            # Read header: magic (4), version (4), num_sequences (4), vocab_size (4)
            magic = f.read(4)
            if magic not in [b'GRMT', b'TRMG']:  # Handle both byte orders
                raise ValueError(f"Invalid GRMT magic: {magic}")
            
            version = struct.unpack('I', f.read(4))[0]
            num_sequences = struct.unpack('I', f.read(4))[0]  # 32-bit, not 64-bit!
            vocab_size = struct.unpack('I', f.read(4))[0]
            
            print(f"  Version: {version}")
            print(f"  Sequences: {num_sequences}")
            print(f"  Vocab size: {vocab_size}")
            
            # Read sequences
            for i in range(num_sequences):
                seq_len = struct.unpack('I', f.read(4))[0]
                tokens = struct.unpack(f'{seq_len}I', f.read(4 * seq_len))
                self.token_sequences.append(list(tokens))
                
                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1}/{num_sequences} sequences...")
        
        print(f"✓ Loaded {len(self.token_sequences)} sequences")
        
    def analyze_token_distribution(self):
        """Analyze token frequency distribution."""
        print("\n" + "="*70)
        print("TOKEN DISTRIBUTION ANALYSIS")
        print("="*70)
        
        all_tokens = []
        for seq in self.token_sequences:
            all_tokens.extend(seq)
        
        token_counts = Counter(all_tokens)
        unique_tokens = len(token_counts)
        total_tokens = len(all_tokens)
        
        print(f"Total tokens: {total_tokens:,}")
        print(f"Unique tokens: {unique_tokens:,}")
        print(f"Token coverage: {unique_tokens / total_tokens * 100:.2f}%")
        
        # Frequency analysis
        frequencies = sorted(token_counts.values(), reverse=True)
        
        # Top 10 most frequent
        top_10_sum = sum(frequencies[:10])
        top_100_sum = sum(frequencies[:100])
        
        print(f"\nFrequency concentration:")
        print(f"  Top 10 tokens: {top_10_sum / total_tokens * 100:.2f}% of all tokens")
        print(f"  Top 100 tokens: {top_100_sum / total_tokens * 100:.2f}% of all tokens")
        
        # Singletons (tokens appearing only once)
        singletons = sum(1 for count in token_counts.values() if count == 1)
        print(f"  Singleton tokens: {singletons:,} ({singletons / unique_tokens * 100:.2f}%)")
        
        # Check for Zipf's law deviation
        # In natural text, token frequency follows Zipf's law: f(k) ~ 1/k^s
        # Calculate how well data fits Zipf
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks[:1000])  # Use top 1000 for fitting
        log_freqs = np.log(frequencies[:1000])
        
        # Linear regression in log space
        coeffs = np.polyfit(log_ranks, log_freqs, 1)
        zipf_exponent = -coeffs[0]
        
        print(f"\nZipf's law analysis:")
        print(f"  Exponent: {zipf_exponent:.3f} (natural text ~1.0)")
        if abs(zipf_exponent - 1.0) > 0.3:
            print(f"  ⚠️ WARNING: Deviation from natural distribution!")
        
        return {
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens,
            'top_10_ratio': top_10_sum / total_tokens,
            'zipf_exponent': zipf_exponent,
            'singleton_ratio': singletons / unique_tokens
        }
    
    def analyze_sequence_diversity(self):
        """Analyze sequence-level diversity."""
        print("\n" + "="*70)
        print("SEQUENCE DIVERSITY ANALYSIS")
        print("="*70)
        
        # Convert to tuples for hashing
        seq_tuples = [tuple(seq) for seq in self.token_sequences]
        unique_sequences = len(set(seq_tuples))
        total_sequences = len(seq_tuples)
        
        print(f"Total sequences: {total_sequences:,}")
        print(f"Unique sequences: {unique_sequences:,}")
        print(f"Duplicate ratio: {(1 - unique_sequences / total_sequences) * 100:.2f}%")
        
        if unique_sequences < total_sequences:
            print(f"  ⚠️ WARNING: {total_sequences - unique_sequences} duplicate sequences!")
        
        # Sequence length distribution
        lengths = [len(seq) for seq in self.token_sequences]
        print(f"\nSequence length statistics:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
        print(f"  Std Dev: {np.std(lengths):.1f}")
        
        # Check for length clustering
        length_counter = Counter(lengths)
        most_common_length = length_counter.most_common(1)[0]
        print(f"  Most common length: {most_common_length[0]} ({most_common_length[1]} sequences, {most_common_length[1]/total_sequences*100:.1f}%)")
        
        return {
            'unique_sequences': unique_sequences,
            'total_sequences': total_sequences,
            'duplicate_ratio': 1 - unique_sequences / total_sequences,
            'length_mean': np.mean(lengths),
            'length_std': np.std(lengths)
        }
    
    def analyze_ngram_diversity(self):
        """Analyze n-gram repetition patterns."""
        print("\n" + "="*70)
        print("N-GRAM REPETITION ANALYSIS")
        print("="*70)
        
        for n in [2, 3, 4]:
            ngrams = []
            for seq in self.token_sequences:
                if len(seq) >= n:
                    for i in range(len(seq) - n + 1):
                        ngrams.append(tuple(seq[i:i+n]))
            
            if not ngrams:
                continue
                
            ngram_counts = Counter(ngrams)
            unique_ngrams = len(ngram_counts)
            total_ngrams = len(ngrams)
            
            # Most repeated n-grams
            most_common = ngram_counts.most_common(5)
            
            print(f"\n{n}-gram analysis:")
            print(f"  Total {n}-grams: {total_ngrams:,}")
            print(f"  Unique {n}-grams: {unique_ngrams:,}")
            print(f"  Repetition ratio: {(1 - unique_ngrams / total_ngrams) * 100:.2f}%")
            
            # Check if top n-grams dominate
            top_5_ratio = sum(count for _, count in most_common) / total_ngrams
            print(f"  Top 5 {n}-grams account for: {top_5_ratio * 100:.2f}% of all {n}-grams")
            
            if top_5_ratio > 0.1:  # If top 5 account for >10%
                print(f"  ⚠️ WARNING: High repetition detected!")
                print(f"  Most repeated {n}-grams:")
                for ngram, count in most_common:
                    print(f"    {ngram}: {count:,} times ({count/total_ngrams*100:.2f}%)")
    
    def compute_sequence_similarity_matrix(self, sample_size=100):
        """Compute pairwise sequence similarity to detect repetitive patterns."""
        print("\n" + "="*70)
        print("SEQUENCE SIMILARITY ANALYSIS")
        print("="*70)
        
        # Sample sequences to avoid O(n^2) computation on full dataset
        sample_size = min(sample_size, len(self.token_sequences))
        indices = np.random.choice(len(self.token_sequences), sample_size, replace=False)
        sample_seqs = [self.token_sequences[i] for i in indices]
        
        print(f"Analyzing {sample_size} random sequences...")
        
        similarities = []
        for i in range(len(sample_seqs)):
            for j in range(i + 1, len(sample_seqs)):
                seq1 = set(sample_seqs[i])
                seq2 = set(sample_seqs[j])
                
                # Jaccard similarity
                intersection = len(seq1 & seq2)
                union = len(seq1 | seq2)
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            max_sim = np.max(similarities)
            
            print(f"\nJaccard similarity (token overlap):")
            print(f"  Mean: {mean_sim:.4f}")
            print(f"  Std Dev: {std_sim:.4f}")
            print(f"  Max: {max_sim:.4f}")
            
            # High similarity indicates lack of diversity
            if mean_sim > 0.5:
                print(f"  ⚠️ WARNING: High mean similarity ({mean_sim:.4f}) indicates low diversity!")
            if max_sim > 0.9:
                print(f"  ⚠️ WARNING: Some sequences are nearly identical!")
            
            # Distribution analysis
            high_sim_count = sum(1 for s in similarities if s > 0.7)
            print(f"  High similarity pairs (>0.7): {high_sim_count} ({high_sim_count/len(similarities)*100:.2f}%)")
            
            return mean_sim, std_sim
        
        return None, None
    
    def detect_structural_patterns(self):
        """Detect structural patterns that might cause gradient alignment."""
        print("\n" + "="*70)
        print("STRUCTURAL PATTERN DETECTION")
        print("="*70)
        
        # Check for common prefixes
        prefix_lengths = [5, 10, 20]
        for prefix_len in prefix_lengths:
            prefixes = []
            for seq in self.token_sequences:
                if len(seq) >= prefix_len:
                    prefixes.append(tuple(seq[:prefix_len]))
            
            if prefixes:
                prefix_counts = Counter(prefixes)
                unique_prefixes = len(prefix_counts)
                total_sequences = len(prefixes)
                
                print(f"\nPrefix length {prefix_len}:")
                print(f"  Unique prefixes: {unique_prefixes}/{total_sequences}")
                print(f"  Diversity: {unique_prefixes/total_sequences*100:.2f}%")
                
                if unique_prefixes / total_sequences < 0.5:
                    print(f"  ⚠️ WARNING: Low prefix diversity - many sequences start similarly!")
                    
                    # Show most common prefixes
                    most_common = prefix_counts.most_common(3)
                    print(f"  Most common prefixes:")
                    for prefix, count in most_common:
                        print(f"    {prefix[:3]}... appears {count} times ({count/total_sequences*100:.1f}%)")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        
        self.load_grmt()
        
        token_stats = self.analyze_token_distribution()
        seq_stats = self.analyze_sequence_diversity()
        self.analyze_ngram_diversity()
        mean_sim, std_sim = self.compute_sequence_similarity_matrix()
        self.detect_structural_patterns()
        
        # Overall verdict
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        issues = []
        
        # Check token distribution
        if token_stats['zipf_exponent'] < 0.7 or token_stats['zipf_exponent'] > 1.3:
            issues.append(f"Unnatural token distribution (Zipf exponent: {token_stats['zipf_exponent']:.3f})")
        
        # Check sequence diversity
        if seq_stats['duplicate_ratio'] > 0.01:
            issues.append(f"High duplicate ratio: {seq_stats['duplicate_ratio']*100:.2f}%")
        
        # Check similarity
        if mean_sim and mean_sim > 0.5:
            issues.append(f"High sequence similarity: {mean_sim:.4f}")
        
        # Check coverage
        coverage_ratio = token_stats['unique_tokens'] / token_stats['total_tokens']
        if coverage_ratio > 0.5:
            issues.append(f"Poor token coverage: {coverage_ratio*100:.2f}% (too many rare tokens)")
        
        if issues:
            print("\n⚠️ DATA QUALITY ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print("\n→ Training data diversity appears LOW")
            print("→ This could explain 99.45% gradient alignment")
        else:
            print("\n✓ Data quality appears GOOD")
            print("→ Gradient alignment issue likely NOT caused by data diversity")
            print("→ May need to investigate model architecture or training dynamics")
        
        # Save metrics to JSON
        report = {
            'token_stats': token_stats,
            'sequence_stats': seq_stats,
            'similarity_mean': mean_sim,
            'similarity_std': std_sim,
            'issues': issues
        }
        
        report_path = Path('data_quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Detailed report saved to: {report_path}")


def main():
    import sys
    
    # Load training data path from config
    try:
        with open('ai_config.json', 'r') as f:
            config = json.load(f)
        
        grmt_path = config['paths']['grim_text']['training_data']
        print(f"Using training data from config: {grmt_path}")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Usage: python inspect_training_data_quality.py [path_to_training_data.grmt]")
        if len(sys.argv) > 1:
            grmt_path = sys.argv[1]
        else:
            return
    
    inspector = GRMTDataInspector(grmt_path)
    inspector.generate_report()


if __name__ == '__main__':
    main()
