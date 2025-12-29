#!/usr/bin/env python3
"""
Weight Initialization Analyzer for GRIM-text Training
Crawls GRIM-text codebase to identify all weight initialization points and analyze variance/stddev values.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json

class WeightInitPoint:
    def __init__(self, file: str, line_num: int, code: str, init_type: str, 
                 stddev: str = None, variance: str = None, layer: str = None):
        self.file = file
        self.line_num = line_num
        self.code = code.strip()
        self.init_type = init_type
        self.stddev = stddev
        self.variance = variance
        self.layer = layer
        
    def __repr__(self):
        return f"WeightInitPoint({Path(self.file).name}:{self.line_num}, {self.init_type})"
    
    def to_dict(self):
        return {
            'file': self.file,
            'line': self.line_num,
            'code': self.code,
            'type': self.init_type,
            'stddev': self.stddev,
            'variance': self.variance,
            'layer': self.layer
        }

def calculate_xavier_stddev(d_in: int, d_out: int) -> float:
    """Calculate expected Xavier stddev: sqrt(2 / (d_in + d_out))"""
    return (2.0 / (d_in + d_out)) ** 0.5

def calculate_kaiming_stddev(d_in: int, mode='fan_in') -> float:
    """Calculate expected Kaiming stddev: sqrt(2 / fan_in)"""
    return (2.0 / d_in) ** 0.5

def find_weight_init_points(root_dir: str) -> List[WeightInitPoint]:
    """Crawl GRIM-text training codebase for weight initialization patterns"""
    
    init_points = []
    extensions = ['.cu', '.cpp', '.hpp', '.h']
    
    # Only search in GRIM-text directories
    grim_text_root = os.path.join(root_dir, 'resources', 'models', 'GRIM-text')
    if not os.path.exists(grim_text_root):
        print(f"ERROR: GRIM-text directory not found at {grim_text_root}")
        return init_points
    
    # Patterns to search for
    patterns = {
        'xavier_init': re.compile(r'(xavier|glorot)', re.IGNORECASE),
        'kaiming_init': re.compile(r'(kaiming|he_init|he_normal|he_uniform)', re.IGNORECASE),
        'curand_call': re.compile(r'curand_(normal|uniform|generate)', re.IGNORECASE),
        'stddev_calc': re.compile(r'std(dev)?.*=.*sqrt', re.IGNORECASE),
        'variance_calc': re.compile(r'variance.*=.*(d_model|fan_in|fan_out)', re.IGNORECASE),
        'init_weights': re.compile(r'(init.*weight|initialize.*weight|launchXavierInit)', re.IGNORECASE),
    }
    
    for root, dirs, files in os.walk(grim_text_root):
        # Skip build directories
        if 'build' in root or '.git' in root:
            continue
            
        for file in files:
            if not any(file.endswith(ext) for ext in extensions):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    # Check each pattern
                    for pattern_name, pattern in patterns.items():
                        if pattern.search(line):
                            # Extract stddev value if present
                            stddev_match = re.search(r'stddev\s*=\s*([^;]+)', line)
                            stddev = stddev_match.group(1).strip() if stddev_match else None
                            
                            # Extract layer information
                            layer_match = re.search(r'layer\s*[=\[]?\s*(\d+)', line, re.IGNORECASE)
                            layer = layer_match.group(1) if layer_match else None
                            
                            init_point = WeightInitPoint(
                                file=file_path,
                                line_num=line_num,
                                code=line,
                                init_type=pattern_name,
                                stddev=stddev,
                                layer=layer
                            )
                            init_points.append(init_point)
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return init_points

def analyze_initialization_values(init_points: List[WeightInitPoint], 
                                  d_model: int = 768, 
                                  d_ff: int = 3072,
                                  vocab_size: int = 37555,
                                  num_layers: int = 12) -> Dict:
    """Analyze if initialization values match theoretical expectations"""
    
    analysis = {
        'expected_values': {},
        'found_values': {},
        'mismatches': [],
        'config': {
            'd_model': d_model,
            'd_ff': d_ff,
            'vocab_size': vocab_size,
            'num_layers': num_layers
        }
    }
    
    # Calculate expected values for each component
    analysis['expected_values'] = {
        'embedding': calculate_xavier_stddev(d_model, vocab_size),
        'lm_head': calculate_xavier_stddev(d_model, vocab_size),
        'W_qkv': calculate_xavier_stddev(d_model, d_model),  # Q, K, V project from d_model
        'W_o': calculate_xavier_stddev(d_model, d_model),
        'W1_ffn': calculate_xavier_stddev(d_model, d_ff),
        'W2_ffn': calculate_xavier_stddev(d_ff, d_model),
        'residual_scale': (1.0 / (2 * num_layers)) ** 0.5  # GPT-2 style
    }
    
    # Extract actual values from code
    for point in init_points:
        if point.stddev:
            key = f"{Path(point.file).name}:{point.line_num}"
            analysis['found_values'][key] = {
                'stddev_expr': point.stddev,
                'type': point.init_type,
                'layer': point.layer,
                'code': point.code
            }
            
    return analysis

def print_report(init_points: List[WeightInitPoint], analysis: Dict):
    """Print formatted report"""
    
    print("=" * 80)
    print("GRIM-TEXT WEIGHT INITIALIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Group by file
    by_file = {}
    for point in init_points:
        fname = str(Path(point.file).relative_to(Path.cwd()))
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(point)
    
    print(f"Found {len(init_points)} initialization points across {len(by_file)} files\n")
    
    # Print by category
    categories = {
        'xavier_init': [],
        'kaiming_init': [],
        'curand_call': [],
        'stddev_calc': [],
        'init_weights': []
    }
    
    for point in init_points:
        if point.init_type in categories:
            categories[point.init_type].append(point)
    
    for category, points in categories.items():
        if not points:
            continue
        print(f"\n{'=' * 80}")
        print(f"{category.upper().replace('_', ' ')} ({len(points)} occurrences)")
        print('=' * 80)
        
        for point in points:
            rel_path = str(Path(point.file).relative_to(Path.cwd()))
            print(f"\n📍 {rel_path}:{point.line_num}")
            if point.layer:
                print(f"   Layer: {point.layer}")
            if point.stddev:
                print(f"   stddev = {point.stddev}")
            print(f"   Code: {point.code[:100]}...")
    
    # Print theoretical analysis
    print("\n" + "=" * 80)
    print("THEORETICAL EXPECTED VALUES")
    print("=" * 80)
    cfg = analysis['config']
    expected = analysis['expected_values']
    
    print(f"\nModel Config:")
    print(f"  d_model = {cfg['d_model']}")
    print(f"  d_ff = {cfg['d_ff']}")
    print(f"  vocab_size = {cfg['vocab_size']}")
    print(f"  num_layers = {cfg['num_layers']}")
    
    print(f"\nExpected Xavier stddev values:")
    for component, value in expected.items():
        print(f"  {component:20s} = {value:.6f}")
    
    # Calculate what we see in code
    print("\n" + "=" * 80)
    print("ACTUAL VALUES IN CODE")
    print("=" * 80)
    
    # Parse actual calculations
    actual_calcs = {
        'W_qkv': f"sqrt(2.0 / (2.0 * {cfg['d_model']})) = {(2.0 / (2.0 * cfg['d_model'])) ** 0.5:.6f}",
        'W_o': f"sqrt(2.0 / (2.0 * {cfg['d_model']})) = {(2.0 / (2.0 * cfg['d_model'])) ** 0.5:.6f}",
        'W1_ffn': f"sqrt(2.0 / ({cfg['d_model']} + {cfg['d_ff']})) = {(2.0 / (cfg['d_model'] + cfg['d_ff'])) ** 0.5:.6f}",
        'W2_ffn': f"sqrt(2.0 / ({cfg['d_ff']} + {cfg['d_model']})) = {(2.0 / (cfg['d_ff'] + cfg['d_model'])) ** 0.5:.6f}",
        'embedding': f"sqrt(2.0 / ({cfg['d_model']} + {cfg['vocab_size']})) = {(2.0 / (cfg['d_model'] + cfg['vocab_size'])) ** 0.5:.6f}",
        'lm_head': f"sqrt(2.0 / ({cfg['d_model']} + {cfg['vocab_size']})) = {(2.0 / (cfg['d_model'] + cfg['vocab_size'])) ** 0.5:.6f}",
    }
    
    for component, calc in actual_calcs.items():
        print(f"  {component:20s} = {calc}")
    
    # Identify potential issues
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES DETECTED")
    print("=" * 80)
    
    issues = []
    
    # Check W_qkv calculation
    # Code uses: sqrt(2.0 / (2.0 * d_model)) = sqrt(2.0 / 1536) ≈ 0.036
    # Theory says: sqrt(2.0 / (d_model + d_model)) = sqrt(2.0 / 1536) ≈ 0.036
    # This is CORRECT! Same formula, just written differently
    
    # Check embedding calculation  
    # Code uses: sqrt(2.0 / (d_model + vocab_size)) = sqrt(2.0 / (768 + 37555))
    # This is standard Xavier for embedding layer
    
    # Main issue: Check if W_o and W2 have residual scaling
    residual_scale = (1.0 / (2 * cfg['num_layers'])) ** 0.5
    print(f"\n1. Residual Scaling Factor:")
    print(f"   Expected: {residual_scale:.6f} (for {cfg['num_layers']} layers)")
    print(f"   Formula: sqrt(1 / (2 * num_layers))")
    print(f"   Purpose: Prevents gradient explosion in deep residual networks")
    print(f"   Applied to: W_o and W2_ffn (projection layers before residual add)")
    
    w_o_scaled = (2.0 / (2.0 * cfg['d_model'])) ** 0.5 * residual_scale
    w2_scaled = (2.0 / (cfg['d_ff'] + cfg['d_model'])) ** 0.5 * residual_scale
    
    print(f"\n2. Scaled Initialization Values:")
    print(f"   W_o (with residual scale): {w_o_scaled:.6f}")
    print(f"   W2 (with residual scale): {w2_scaled:.6f}")
    
    # Check layer norm init
    print(f"\n3. Layer Norm Initialization:")
    print(f"   ⚠️  Check if gamma=1.0, beta=0.0 (standard)")
    print(f"   ⚠️  For deep networks, consider gamma=0.5 or smaller")
    
    # Check bias terms
    print(f"\n4. Bias Terms:")
    print(f"   ⚠️  Biases should be initialized to 0.0")
    print(f"   ⚠️  Attention biases especially should be zero")
    
    # Check position embeddings (if used)
    print(f"\n5. Position Embeddings:")
    print(f"   ⚠️  If using learned positional embeddings, check if initialized")
    print(f"   ⚠️  Typical: Xavier with same stddev as token embeddings")
    
    # Export to JSON
    output = {
        'initialization_points': [p.to_dict() for p in init_points],
        'analysis': analysis,
        'timestamp': str(Path.cwd())
    }
    
    with open('weight_init_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Full analysis exported to weight_init_analysis.json")

def main():
    root_dir = Path.cwd()
    print(f"Scanning directory: {root_dir}")
    print("Looking for weight initialization patterns...")
    print()
    
    # Find all initialization points
    init_points = find_weight_init_points(str(root_dir))
    
    # Analyze values
    analysis = analyze_initialization_values(
        init_points,
        d_model=768,
        d_ff=3072,
        vocab_size=37555,
        num_layers=12
    )
    
    # Print report
    print_report(init_points, analysis)

if __name__ == '__main__':
    main()
