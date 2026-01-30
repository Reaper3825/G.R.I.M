#!/usr/bin/env python3
"""
GRIM-text Training Signal Diagnostic Tool

Analyzes training logs to correlate and diagnose:
- LogitSignal: Mode collapse detection (unique_argmax, top2_margin)
- HiddenCosine: Hidden state similarity (position collapse)
- LMHeadNorm: Weight norm evolution
- Token277Trace: Gradient direction vs weight changes
- Token277 POST-OPT: Weight norm increase despite negative gradients

Usage:
    python diagnose_training_signals.py [log_file] [--batch N] [--last N] [--summary]
"""

import re
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

# ============================================================================
# Data Classes for Parsed Log Entries
# ============================================================================

@dataclass
class LogitSignal:
    batch: int
    logit_mean: float
    logit_max: float
    logit_min: float
    avg_max_logit: float
    top2_margin: float
    unique_argmax: int
    top_argmax: Dict[str, int] = field(default_factory=dict)

@dataclass
class HiddenCosine:
    batch: int
    cosine_pairs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    avg_cos: float = 0.0

@dataclass
class LMHeadNorm:
    batch: int
    token_norms: Dict[str, float] = field(default_factory=dict)

@dataclass
class Token277Trace:
    batch: int
    grad_norm: float
    grad_sum: float
    grad_mean: float
    weight_norm: float
    weight_mean: float
    target_count: int
    total_tokens: int
    ratio_pct: float
    is_negative_grad: bool

@dataclass
class Token277PostOpt:
    batch: int
    pre_norm: float
    post_norm: float
    delta_norm: float
    delta_mean: float
    norm_increased: bool

@dataclass
class GradTrace:
    batch: int
    total: float = 0.0
    emb_lm_tied: float = 0.0
    num: float = 0.0
    attn: float = 0.0
    ffn: float = 0.0
    rms: float = 0.0

@dataclass
class LossStats:
    batch: int
    loss_mean: float
    loss_sum: float
    valid_tokens: int
    masked_tokens: int
    total_tokens: int

# ============================================================================
# Data Classes for training_run.log (detailed C++ diagnostics)
# ============================================================================

@dataclass
class GradAttrib:
    """Gradient attribution for Token 277 from CUDA backward pass"""
    batch: int
    lm_head_sum: float
    lm_head_norm: float
    lm_head_mean: float
    raw_emb_sum: float
    raw_emb_norm: float
    raw_emb_mean: float
    cosine_lm_emb: float
    final_sum: float
    final_norm: float
    final_mean: float
    direction: str  # "LM wants DECREASE" or "LM wants INCREASE"

@dataclass
class UpdateProbeValues:
    """AdamW update diagnostic from optimizer step"""
    group: str
    step: int
    w_before: List[float]
    w_after: List[float]
    grad: List[float]
    update: List[float]

@dataclass
class UpdateProbeState:
    """AdamW optimizer state (momentum/variance)"""
    group: str
    step: int
    offset: int
    token_id: int
    m_state: List[float]
    v_state: List[float]

@dataclass
class FlashAttnFwdLSE:
    """FlashAttention forward pass LSE (Log-Sum-Exp) statistics"""
    call: int
    seqlen: int
    batch: int
    n_heads: int
    total_elems: int
    nan_count: int = 0
    inf_count: int = 0
    lse_min: float = 0.0
    lse_max: float = 0.0
    has_anomaly: bool = False

@dataclass
class Issue96PosEmb:
    """Issue #96 position embedding diagnostic"""
    batch: int
    skipped: bool
    reason: str

@dataclass 
class FABwdOut:
    """FlashAttention backward output gradients"""
    call: int
    dq_nan: int
    dq_inf: int
    dq_max: float
    dq_rms: float
    dk_nan: int
    dk_inf: int
    dk_max: float
    dk_rms: float
    dv_nan: int
    dv_inf: int
    dv_max: float
    dv_rms: float

@dataclass
class LossComponents:
    """Loss breakdown by component"""
    text_ce: float
    text_weight: float
    numeric: float
    numeric_weight: float

@dataclass
class GradExport:
    """Per-layer gradient export"""
    name: str
    elements: int
    norm: float

@dataclass
class FAFwdIn:
    """FlashAttention forward input (Q/K/V)"""
    call: int
    q_nan: int
    q_inf: int
    q_first: float
    k_nan: int
    k_inf: int
    k_first: float
    v_nan: int
    v_inf: int
    v_first: float

@dataclass
class FAFwdOut:
    """FlashAttention forward output"""
    call: int
    nan: int
    inf: int
    first: float

@dataclass
class GELUBwd:
    """GELU backward gradient stats"""
    call: int
    numel: int
    max_grad: float
    rms: float

@dataclass
class RMSBwd:
    """RMSNorm backward gradient stats"""
    call: int
    numel: int
    max_grad: float

@dataclass
class Issue93Fwd:
    """Issue93 forward layer diagnostics"""
    layer: int
    stage: str  # layer_input, flash_attn_out, attn_out_flat, proj_out_W_o, residual1, ffn_out, layer_output
    min_val: float
    max_val: float

@dataclass
class LossBwdOut:
    """Loss backward output stats"""
    num_tokens: int
    vocab: int
    valid: int

# ============================================================================
# Simple String Parsers (no regex needed)
# ============================================================================

def parse_kv(line: str) -> Dict[str, str]:
    """Parse key=value pairs from a line"""
    result = {}
    parts = line.split()
    for part in parts:
        if '=' in part:
            k, v = part.split('=', 1)
            result[k] = v
    return result

def parse_range(s: str) -> Tuple[float, float]:
    """Parse [min, max] or range=[min,max] format"""
    s = s.strip('[]')
    if ',' in s:
        parts = s.split(',')
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except (ValueError, IndexError):
            return 0.0, 0.0
    return 0.0, 0.0

def parse_top_argmax(s: str) -> Dict[str, int]:
    """Parse top_argmax=[tok277:37,tok258:13] format"""
    result = {}
    s = s.strip('[]')
    for item in s.split(','):
        if ':' in item:
            tok, count = item.split(':')
            result[tok.strip()] = int(count)
    return result

def parse_token_norms(s: str) -> Dict[str, float]:
    """Parse tok277:0.1957,tok258:0.1876 format"""
    result = {}
    for item in s.split(','):
        if ':' in item:
            tok, norm = item.split(':')
            result[tok.strip()] = float(norm)
    return result

def parse_cosine_pairs(s: str) -> Dict[Tuple[int, int], float]:
    """Parse (0,1):0.914,(0,10):0.599 format"""
    result = {}
    s = s.strip('[]')
    for item in s.split(','):
        item = item.strip()
        if item.startswith('(') and ':' in item:
            pair_part, val = item.rsplit(':', 1)
            pair_part = pair_part.strip('()')
            if ',' in pair_part:
                i, j = pair_part.split(',')
                result[(int(i), int(j))] = float(val)
    return result

def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated float list like '0.001, -0.002, 0.003'"""
    s = s.strip('[]')
    if not s:
        return []
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def get_float(kv: Dict[str, str], key: str, default: float = 0.0) -> float:
    """Get float from parsed key-value dict"""
    try:
        return float(kv.get(key, default))
    except (ValueError, TypeError):
        return default

def get_int(kv: Dict[str, str], key: str, default: int = 0) -> int:
    """Get int from parsed key-value dict"""
    try:
        return int(kv.get(key, default))
    except (ValueError, TypeError):
        return default

def detect_encoding(filepath: str) -> str:
    """Detect file encoding by checking BOM"""
    with open(filepath, 'rb') as f:
        header = f.read(4)
        if header[:2] == b'\xff\xfe':
            return 'utf-16-le'
        elif header[:2] == b'\xfe\xff':
            return 'utf-16-be'
        elif header[:3] == b'\xef\xbb\xbf':
            return 'utf-8-sig'
        return 'utf-8'


def parse_runtime_log(filepath: str) -> Dict[str, List]:
    """Parse training_run.log using simple string operations"""
    data = {
        'grad_attrib': [],
        'update_probe_values': [],
        'update_probe_state': [],
        'fa_lse_summary': [],
        'fa_bwd_out': [],
        'issue91': [],
        'issue93': [],
        'issue94': [],
        'issue96': [],
    }
    
    current_grad_attrib = None
    encoding = detect_encoding(filepath)
    
    with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # GRAD_ATTRIB parsing (multi-line)
            if '[GRAD_ATTRIB]' in line and 'TOKEN_277' in line:
                if current_grad_attrib:
                    data['grad_attrib'].append(current_grad_attrib)
                kv = parse_kv(line)
                current_grad_attrib = GradAttrib(
                    batch=get_int(kv, 'batch'),
                    lm_head_sum=0, lm_head_norm=0, lm_head_mean=0,
                    raw_emb_sum=0, raw_emb_norm=0, raw_emb_mean=0,
                    cosine_lm_emb=0, final_sum=0, final_norm=0, final_mean=0,
                    direction=""
                )
                continue
            
            if current_grad_attrib:
                if 'LM_HEAD_ONLY:' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('sum='):
                            current_grad_attrib.lm_head_sum = float(p[4:])
                        elif p.startswith('norm='):
                            current_grad_attrib.lm_head_norm = float(p[5:])
                        elif p.startswith('mean='):
                            current_grad_attrib.lm_head_mean = float(p[5:])
                    continue
                elif 'RAW_EMBEDDING:' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('sum='):
                            current_grad_attrib.raw_emb_sum = float(p[4:])
                        elif p.startswith('norm='):
                            current_grad_attrib.raw_emb_norm = float(p[5:])
                        elif p.startswith('mean='):
                            current_grad_attrib.raw_emb_mean = float(p[5:])
                    continue
                elif 'COSINE(lm,emb):' in line:
                    val = line.split('COSINE(lm,emb):')[1].strip().split()[0]
                    current_grad_attrib.cosine_lm_emb = float(val)
                    continue
                elif 'FINAL_GRADIENT:' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('sum='):
                            current_grad_attrib.final_sum = float(p[4:])
                        elif p.startswith('norm='):
                            current_grad_attrib.final_norm = float(p[5:])
                        elif p.startswith('mean='):
                            current_grad_attrib.final_mean = float(p[5:])
                    continue
                elif 'DIRECTION:' in line:
                    current_grad_attrib.direction = line.split('DIRECTION:')[1].strip()
                    data['grad_attrib'].append(current_grad_attrib)
                    current_grad_attrib = None
                    continue
            
            # UpdateProbeValues
            if '[UpdateProbeValues]' in line:
                # group='embedding_lm_head_tied' step=0 w_before[0:4]=[...] ...
                group = line.split("group='")[1].split("'")[0] if "group='" in line else ""
                step = int(line.split("step=")[1].split()[0]) if "step=" in line else 0
                w_before = parse_float_list(line.split("w_before[0:4]=")[1].split("]")[0] + "]") if "w_before" in line else []
                w_after = parse_float_list(line.split("w_after[0:4]=")[1].split("]")[0] + "]") if "w_after" in line else []
                grad = parse_float_list(line.split("grad[0:4]=")[1].split("]")[0] + "]") if "grad[0:4]=" in line else []
                update = parse_float_list(line.split("update[0:4]=")[1].split("]")[0] + "]") if "update[0:4]=" in line else []
                data['update_probe_values'].append(UpdateProbeValues(
                    group=group, step=step, w_before=w_before, w_after=w_after, grad=grad, update=update
                ))
                continue
            
            # UpdateProbeState
            if '[UpdateProbeState]' in line:
                group = line.split("group='")[1].split("'")[0] if "group='" in line else ""
                step = int(line.split("step=")[1].split()[0]) if "step=" in line else 0
                offset = int(line.split("offset=")[1].split()[0]) if "offset=" in line else 0
                token_id = int(line.split("(token ")[1].split(")")[0]) if "(token " in line else 0
                m_state = parse_float_list(line.split("m_state[0:4]=")[1].split("]")[0] + "]") if "m_state" in line else []
                v_state = parse_float_list(line.split("v_state[0:4]=")[1].split("]")[0] + "]") if "v_state" in line else []
                data['update_probe_state'].append(UpdateProbeState(
                    group=group, step=step, offset=offset, token_id=token_id, m_state=m_state, v_state=v_state
                ))
                continue
            
            # FA-FWD-LSE-SUMMARY or standalone softmax_lse line
            # Format 1: [FA-FWD-LSE-SUMMARY] nan=0 inf=0 range=[min,max] mean=X
            # Format 2:     softmax_lse: nan=0 inf=0 range=[-542.99, 881.25]
            if '[FA-FWD-LSE-SUMMARY]' in line or 'softmax_lse:' in line:
                kv = parse_kv(line)
                # Extract range=[...] - find the closing bracket to handle space after comma
                lse_range = '[0,0]'
                if 'range=' in line:
                    start = line.index('range=') + 6
                    end = line.index(']', start) + 1
                    lse_range = line[start:end]
                lse_min, lse_max = parse_range(lse_range)
                data['fa_lse_summary'].append({
                    'nan': get_int(kv, 'nan'),
                    'inf': get_int(kv, 'inf'),
                    'lse_min': lse_min,
                    'lse_max': lse_max,
                    'lse_mean': get_float(kv, 'mean'),
                    'has_anomaly': abs(lse_max - lse_min) > 100,
                })
                continue
            
            # FA-BWD-OUT (dQ/dK/dV gradients)
            if '[FA-BWD-OUT]' in line:
                kv = parse_kv(line)
                # dQ: max=X rms=Y, dK: max=X rms=Y, dV: max=X rms=Y
                entry = {'call': get_int(kv, 'call')}
                for grad_type in ['dQ', 'dK', 'dV']:
                    if f'{grad_type}:' in line:
                        after = line.split(f'{grad_type}:')[1]
                        parts = after.split()
                        for p in parts[:2]:  # max and rms are first two
                            if p.startswith('max='):
                                entry[f'{grad_type}_max'] = float(p[4:])
                            elif p.startswith('rms='):
                                entry[f'{grad_type}_rms'] = float(p[4:])
                data['fa_bwd_out'].append(entry)
                continue
            
            # Issue91 embedding diagnostics
            # Note: rms_input entries have stats on the NEXT line after the header
            # Format: [Issue91-FWD-rms_input] layer=0 batch=0 tokens=7168 d_model=768:
            #         min=-0.215516 max=0.842303 mean=0.000013 std=0.176688 rms=0.176688
            if '[Issue91-' in line:
                tag = line.split(']')[0].split('[')[1]
                kv = parse_kv(line)
                # Handle gamma entries which use gamma_mean/gamma_rms instead of min/max
                if 'gamma_mean' in kv:
                    # Gamma entries: use gamma values as min/max (they should all be ~1.0)
                    gamma_val = get_float(kv, 'gamma_mean')
                    gamma_rms = get_float(kv, 'gamma_rms')
                    data['issue91'].append({
                        'tag': tag,
                        'layer': get_int(kv, 'layer', -1),
                        'min': gamma_val,  # Use gamma_mean for range display
                        'max': gamma_rms,  # Use gamma_rms for range display
                        'mean': gamma_val,
                        'rms': gamma_rms,
                    })
                elif 'min=' in line and 'max=' in line:
                    # Stats on same line (e.g., EMB-AFTER-SB, W_qkv)
                    data['issue91'].append({
                        'tag': tag,
                        'layer': get_int(kv, 'layer', -1),
                        'min': get_float(kv, 'min'),
                        'max': get_float(kv, 'max'),
                        'mean': get_float(kv, 'mean'),
                        'rms': get_float(kv, 'rms'),
                    })
                else:
                    # Header line with stats on next line (e.g., rms_input)
                    # Store partial entry, will be completed when next line is read
                    data['_issue91_pending'] = {
                        'tag': tag,
                        'layer': get_int(kv, 'layer', -1),
                    }
                continue
            
            # Check for continuation of Issue91 entry (stats on separate line)
            if data.get('_issue91_pending') and 'min=' in line and '[' not in line:
                pending = data.pop('_issue91_pending')
                kv = parse_kv(line)
                pending['min'] = get_float(kv, 'min')
                pending['max'] = get_float(kv, 'max')
                pending['mean'] = get_float(kv, 'mean')
                pending['rms'] = get_float(kv, 'rms')
                data['issue91'].append(pending)
                continue
            
            # Issue93 forward layer diagnostics
            if '[Issue93-FWD-' in line:
                tag = line.split(']')[0].split('[Issue93-FWD-')[1]
                kv = parse_kv(line)
                # Extract layer from layer=X
                data['issue93'].append({
                    'stage': tag,
                    'layer': get_int(kv, 'layer'),
                    'min': get_float(kv, 'min'),
                    'max': get_float(kv, 'max'),
                })
                continue
            
            # Issue94 RMSNorm diagnostics
            if '[Issue94-RMSNorm-' in line:
                tag = line.split(']')[0].split('[Issue94-RMSNorm-')[1]
                kv = parse_kv(line)
                # per_row_rms: min=X max=Y
                rms_min, rms_max = 0.0, 0.0
                if 'per_row_rms:' in line:
                    after = line.split('per_row_rms:')[1]
                    parts = after.split()
                    for p in parts:
                        if p.startswith('min='):
                            rms_min = float(p[4:])
                        elif p.startswith('max='):
                            rms_max = float(p[4:])
                data['issue94'].append({
                    'stage': tag,
                    'layer': get_int(kv, 'layer'),
                    'rms_min': rms_min,
                    'rms_max': rms_max,
                })
                continue
            
            # Issue96 position embedding skip
            if '[Issue96-POSEMB]' in line:
                data['issue96'].append({
                    'skipped': 'SKIPPED' in line,
                    'reason': line.split(']')[1].strip(),
                })
                continue
    
    if current_grad_attrib:
        data['grad_attrib'].append(current_grad_attrib)
    
    return data


def parse_log_file(filepath: str) -> Dict[str, List]:
    """Parse all relevant entries from log file (Python-side training log)"""
    data = {
        'logit_signal': [],
        'hidden_cosine': [],
        'lm_head_norm': [],
        'token277_trace': [],
        'token277_post_opt': [],
        'grad_trace': [],
        'loss_stats': [],
    }
    
    current_batch = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # LogitSignal
            if '[LogitSignal]' in line:
                kv = parse_kv(line)
                # top_argmax=[tok277:37,tok258:13]
                top_argmax_str = line.split('top_argmax=')[1].split()[0] if 'top_argmax=' in line else '[]'
                data['logit_signal'].append(LogitSignal(
                    batch=get_int(kv, 'batch'),
                    logit_mean=get_float(kv, 'logit_mean'),
                    logit_max=get_float(kv, 'logit_max'),
                    logit_min=get_float(kv, 'logit_min'),
                    avg_max_logit=get_float(kv, 'avg_max_logit'),
                    top2_margin=get_float(kv, 'top2_margin'),
                    unique_argmax=get_int(kv, 'unique_argmax'),
                    top_argmax=parse_top_argmax(top_argmax_str)
                ))
                current_batch = get_int(kv, 'batch')
                continue
            
            # HiddenCosine
            if '[HiddenCosine]' in line:
                kv = parse_kv(line)
                cos_str = line.split('cos(h_i,h_j)=')[1].split(']')[0] + ']' if 'cos(h_i,h_j)=' in line else '[]'
                data['hidden_cosine'].append(HiddenCosine(
                    batch=get_int(kv, 'batch'),
                    cosine_pairs=parse_cosine_pairs(cos_str),
                    avg_cos=get_float(kv, 'avg_cos')
                ))
                continue
            
            # LMHeadNorm
            if '[LMHeadNorm]' in line:
                kv = parse_kv(line)
                norms_str = line.split('||W[v]||=')[1].split(']')[0] + ']' if '||W[v]||=' in line else '[]'
                data['lm_head_norm'].append(LMHeadNorm(
                    batch=get_int(kv, 'batch'),
                    token_norms=parse_token_norms(norms_str.strip('[]'))
                ))
                continue
            
            # Token277Trace
            if '[Token277Trace]' in line:
                kv = parse_kv(line)
                # Parse grad_row277: norm=X sum=Y mean=Z
                grad_norm, grad_sum, grad_mean = 0.0, 0.0, 0.0
                if 'grad_row277:' in line:
                    after = line.split('grad_row277:')[1]
                    parts = after.split('|')[0].split()
                    for p in parts:
                        if p.startswith('norm='): grad_norm = float(p[5:])
                        elif p.startswith('sum='): grad_sum = float(p[4:])
                        elif p.startswith('mean='): grad_mean = float(p[5:])
                # Parse weight_row277: norm=X mean=Y
                weight_norm, weight_mean = 0.0, 0.0
                if 'weight_row277:' in line:
                    after = line.split('weight_row277:')[1]
                    parts = after.split('|')[0].split()
                    for p in parts:
                        if p.startswith('norm='): weight_norm = float(p[5:])
                        elif p.startswith('mean='): weight_mean = float(p[5:])
                # Parse targets: 277_count=X/Y ratio=Z%
                target_count, total_tokens, ratio_pct = 0, 0, 0.0
                if '277_count=' in line:
                    count_part = line.split('277_count=')[1].split()[0]
                    if '/' in count_part:
                        target_count, total_tokens = int(count_part.split('/')[0]), int(count_part.split('/')[1])
                if 'ratio=' in line:
                    ratio_part = line.split('ratio=')[1].split('%')[0]
                    ratio_pct = float(ratio_part)
                data['token277_trace'].append(Token277Trace(
                    batch=get_int(kv, 'batch'),
                    grad_norm=grad_norm,
                    grad_sum=grad_sum,
                    grad_mean=grad_mean,
                    weight_norm=weight_norm,
                    weight_mean=weight_mean,
                    target_count=target_count,
                    total_tokens=total_tokens,
                    ratio_pct=ratio_pct,
                    is_negative_grad='NEGATIVE_GRAD' in line
                ))
                continue
            
            # Token277 POST-OPT
            if '[Token277] POST-OPT' in line:
                kv = parse_kv(line)
                data['token277_post_opt'].append(Token277PostOpt(
                    batch=get_int(kv, 'batch'),
                    pre_norm=get_float(kv, 'pre_norm'),
                    post_norm=get_float(kv, 'post_norm'),
                    delta_norm=get_float(kv, 'delta_norm'),
                    delta_mean=get_float(kv, 'delta_mean'),
                    norm_increased='NORM_INCREASED' in line
                ))
                continue
            
            # GradTrace COMPUTED COMPONENTS
            if '[GradTrace] COMPUTED COMPONENTS:' in line:
                kv = parse_kv(line)
                data['grad_trace'].append(GradTrace(
                    batch=current_batch,
                    total=get_float(kv, 'total'),
                    emb_lm_tied=get_float(kv, 'emb_lm_tied'),
                    num=get_float(kv, 'num'),
                    attn=get_float(kv, 'attn'),
                    ffn=get_float(kv, 'ffn'),
                    rms=get_float(kv, 'rms')
                ))
                continue
            
            # LossStats
            if '[LossStats]' in line:
                kv = parse_kv(line)
                data['loss_stats'].append(LossStats(
                    batch=get_int(kv, 'batch'),
                    loss_mean=get_float(kv, 'loss_mean'),
                    loss_sum=get_float(kv, 'loss_sum'),
                    valid_tokens=get_int(kv, 'valid_tokens'),
                    masked_tokens=get_int(kv, 'masked_tokens'),
                    total_tokens=get_int(kv, 'total_tokens')
                ))
                continue
    
    return data

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_mode_collapse(logit_signals: List[LogitSignal]) -> Dict:
    """Detect mode collapse from LogitSignal data"""
    if not logit_signals:
        return {'status': 'NO_DATA'}
    
    results = {
        'unique_argmax_trend': [],
        'tok277_dominance': [],
        'top2_margin_trend': [],
        'collapse_detected': False,
        'collapse_batch': None,
    }
    
    for ls in logit_signals:
        results['unique_argmax_trend'].append((ls.batch, ls.unique_argmax))
        
        total = sum(ls.top_argmax.values())
        tok277_count = ls.top_argmax.get('tok277', 0)
        tok277_pct = (tok277_count / total * 100) if total > 0 else 0
        results['tok277_dominance'].append((ls.batch, tok277_pct))
        results['top2_margin_trend'].append((ls.batch, ls.top2_margin))
        
        # Detect collapse: unique_argmax <= 2 AND tok277 > 70%
        if ls.unique_argmax <= 2 and tok277_pct > 70 and not results['collapse_detected']:
            results['collapse_detected'] = True
            results['collapse_batch'] = ls.batch
    
    return results

def analyze_hidden_collapse(hidden_cosines: List[HiddenCosine]) -> Dict:
    """Detect hidden state collapse from cosine similarities"""
    if not hidden_cosines:
        return {'status': 'NO_DATA'}
    
    results = {
        'avg_cos_trend': [],
        'high_similarity_batches': [],
        'collapse_threshold': 0.7,
    }
    
    for hc in hidden_cosines:
        results['avg_cos_trend'].append((hc.batch, hc.avg_cos))
        
        if hc.avg_cos > results['collapse_threshold']:
            results['high_similarity_batches'].append(hc.batch)
    
    return results

def analyze_weight_paradox(token277_traces: List[Token277Trace], 
                           token277_post_opts: List[Token277PostOpt]) -> Dict:
    """
    Detect the weight paradox: negative gradient but weight norm increasing.
    This indicates AdamW/optimizer dynamics overriding gradient direction.
    """
    results = {
        'paradox_batches': [],
        'grad_direction_consistent': True,
        'weight_norm_trend': [],
        'grad_sum_trend': [],
    }
    
    # Index post-opt data by batch
    post_opt_by_batch = {p.batch: p for p in token277_post_opts}
    
    for trace in token277_traces:
        batch = trace.batch
        results['grad_sum_trend'].append((batch, trace.grad_sum))
        results['weight_norm_trend'].append((batch, trace.weight_norm))
        
        # Check for paradox: negative gradient but norm increased
        post_opt = post_opt_by_batch.get(batch)
        if post_opt and trace.is_negative_grad and post_opt.norm_increased:
            results['paradox_batches'].append({
                'batch': batch,
                'grad_sum': trace.grad_sum,
                'pre_norm': post_opt.pre_norm,
                'post_norm': post_opt.post_norm,
                'delta': post_opt.delta_norm,
            })
    
    return results

def correlate_batch(batch: int, data: Dict, runtime_data: Dict = None) -> Dict:
    """Get all signals for a specific batch"""
    result = {'batch': batch}
    
    for ls in data['logit_signal']:
        if ls.batch == batch:
            result['logit_signal'] = {
                'logit_mean': ls.logit_mean,
                'logit_max': ls.logit_max,
                'logit_min': ls.logit_min,
                'avg_max_logit': ls.avg_max_logit,
                'top2_margin': ls.top2_margin,
                'unique_argmax': ls.unique_argmax,
                'top_argmax': ls.top_argmax,
            }
            break
    
    for hc in data['hidden_cosine']:
        if hc.batch == batch:
            result['hidden_cosine'] = {
                'cosine_pairs': {f"({k[0]},{k[1]})": v for k, v in hc.cosine_pairs.items()},
                'avg_cos': hc.avg_cos,
            }
            break
    
    for ln in data['lm_head_norm']:
        if ln.batch == batch:
            result['lm_head_norm'] = ln.token_norms
            break
    
    for t in data['token277_trace']:
        if t.batch == batch:
            result['token277_trace'] = {
                'grad_norm': t.grad_norm,
                'grad_sum': t.grad_sum,
                'grad_mean': t.grad_mean,
                'weight_norm': t.weight_norm,
                'is_negative_grad': t.is_negative_grad,
                'target_ratio': f"{t.ratio_pct}%",
            }
            break
    
    for p in data['token277_post_opt']:
        if p.batch == batch:
            result['token277_post_opt'] = {
                'pre_norm': p.pre_norm,
                'post_norm': p.post_norm,
                'delta_norm': p.delta_norm,
                'norm_increased': p.norm_increased,
            }
            break
    
    for g in data['grad_trace']:
        if g.batch == batch:
            result['grad_components'] = {
                'total': g.total,
                'emb_lm_tied': g.emb_lm_tied,
                'attn': g.attn,
                'ffn': g.ffn,
                'rms': g.rms,
            }
            break
    
    for ls in data['loss_stats']:
        if ls.batch == batch:
            result['loss'] = {
                'loss_mean': ls.loss_mean,
                'valid_tokens': ls.valid_tokens,
            }
            break
    
    # Add runtime data if available
    if runtime_data:
        # GRAD_ATTRIB data
        if batch in runtime_data.get('grad_attrib_by_batch', {}):
            ga = runtime_data['grad_attrib_by_batch'][batch]
            result['grad_attrib'] = {
                'direction': ga.direction,
                'lm_head': {'sum': ga.lm_head_sum, 'norm': ga.lm_head_norm, 'mean': ga.lm_head_mean},
                'raw_embedding': {'sum': ga.raw_emb_sum, 'norm': ga.raw_emb_norm, 'mean': ga.raw_emb_mean},
                'raw_emb_is_zero': ga.raw_emb_norm == 0.0,
            }
        
        # UpdateProbe data (keyed by optimizer step, not batch)
        # Each optimizer step covers 2 micro-batches for grad_accum=2
        opt_step = batch // 2  # Approximate mapping
        if opt_step in runtime_data.get('update_probe_values_by_step', {}):
            upv = runtime_data['update_probe_values_by_step'][opt_step]
            result['update_probe'] = {
                'step': upv.step,
                'm_277': upv.m_277,
                'v_277': upv.v_277,
                'm_hat_277': upv.m_hat_277,
                'v_hat_277': upv.v_hat_277,
                'update_277': upv.update_277,
            }
        
        # FA-LSE data (if available for this batch)
        for lse in runtime_data.get('fa_lse', []):
            if lse.batch == batch:
                result['flash_attn_lse'] = {
                    'lse_min': lse.lse_min,
                    'lse_max': lse.lse_max,
                    'lse_mean': lse.lse_mean,
                    'anomaly': lse.anomaly,
                }
                break
    
    return result

# ============================================================================
# Report Generation
# ============================================================================

def print_batch_report(batch_data: Dict):
    """Print detailed report for a single batch"""
    batch = batch_data.get('batch', '?')
    print(f"\n{'='*70}")
    print(f"  BATCH {batch}")
    print(f"{'='*70}")
    
    # Loss
    if 'loss' in batch_data:
        loss = batch_data['loss']
        print(f"\nLOSS: {loss['loss_mean']:.4f} (tokens={loss['valid_tokens']})")
    
    # Logit Signal Analysis
    if 'logit_signal' in batch_data:
        ls = batch_data['logit_signal']
        print(f"\nLOGIT SIGNAL:")
        print(f"   logit_mean={ls['logit_mean']:.4f}  logit_range=[{ls['logit_min']:.2f}, {ls['logit_max']:.2f}]")
        print(f"   avg_max_logit={ls['avg_max_logit']:.4f}  top2_margin={ls['top2_margin']:.4f}")
        print(f"   unique_argmax={ls['unique_argmax']}")
        print(f"   top_argmax: {ls['top_argmax']}")
        
        # Calculate tok277 dominance
        total = sum(ls['top_argmax'].values())
        tok277 = ls['top_argmax'].get('tok277', 0)
        if total > 0:
            pct = tok277 / total * 100
            print(f"   tok277_pct: {pct:.1f}%")
    
    # Hidden Cosine Analysis
    if 'hidden_cosine' in batch_data:
        hc = batch_data['hidden_cosine']
        print(f"\nHIDDEN STATE SIMILARITY:")
        print(f"   avg_cos={hc['avg_cos']:.4f}")
        print(f"   pairs: {hc['cosine_pairs']}")
    
    # LM Head Norm
    if 'lm_head_norm' in batch_data:
        ln = batch_data['lm_head_norm']
        print(f"\nLM HEAD WEIGHT NORMS:")
        for tok, norm in ln.items():
            print(f"   ||W[{tok}]|| = {norm:.4f}")
    
    # Token 277 Gradient Trace
    if 'token277_trace' in batch_data:
        t = batch_data['token277_trace']
        print(f"\nTOKEN 277 GRADIENT:")
        print(f"   grad: norm={t['grad_norm']:.4f}  sum={t['grad_sum']:.4f}  mean={t['grad_mean']:.6f}")
        print(f"   weight: norm={t['weight_norm']:.4f}")
        print(f"   in targets: {t['target_ratio']}")
        print(f"   is_negative_grad: {t['is_negative_grad']}")
    
    # Token 277 Post-Optimizer
    if 'token277_post_opt' in batch_data:
        p = batch_data['token277_post_opt']
        print(f"\nTOKEN 277 POST-OPTIMIZER:")
        print(f"   pre_norm={p['pre_norm']:.6f}  post_norm={p['post_norm']:.6f}")
        print(f"   delta_norm={p['delta_norm']:.6f}  norm_increased={p['norm_increased']}")
    
    # Gradient Components
    if 'grad_components' in batch_data:
        g = batch_data['grad_components']
        print(f"\nGRADIENT COMPONENTS:")
        print(f"   total={g['total']:.4f}  emb_lm={g['emb_lm_tied']:.4f}  attn={g['attn']:.4f}  ffn={g['ffn']:.4f}  rms={g['rms']:.4f}")
    
    # GRAD_ATTRIB (from training_run.log)
    if 'grad_attrib' in batch_data:
        ga = batch_data['grad_attrib']
        print(f"\nGRAD ATTRIBUTION (C++ runtime):")
        print(f"   direction: {ga['direction']}")
        lm = ga['lm_head']
        print(f"   LM_HEAD:       sum={lm['sum']:+.6f}  norm={lm['norm']:.6f}  mean={lm['mean']:+.9f}")
        re = ga['raw_embedding']
        print(f"   RAW_EMBEDDING: sum={re['sum']:+.6f}  norm={re['norm']:.6f}  mean={re['mean']:+.9f}")
    
    # UpdateProbe (optimizer state)
    if 'update_probe' in batch_data:
        up = batch_data['update_probe']
        print(f"\nADAMW STATE (step {up['step']}):")
        print(f"   m[277]={up['m_277']:+.9f}  v[277]={up['v_277']:.9f}")
        print(f"   m_hat[277]={up['m_hat_277']:+.9f}  v_hat[277]={up['v_hat_277']:.9f}")
        print(f"   update[277]={up['update_277']:+.9f}")
    
    # Flash Attention LSE
    if 'flash_attn_lse' in batch_data:
        lse = batch_data['flash_attn_lse']
        print(f"\nFLASH ATTENTION LSE:")
        print(f"   lse_range=[{lse['lse_min']:.2f}, {lse['lse_max']:.2f}]  mean={lse['lse_mean']:.2f}")

def print_summary_report(data: Dict, runtime_data: Dict = None):
    """Print summary analysis across all batches"""
    print(f"\n{'='*70}")
    print(f"  TRAINING SIGNAL SUMMARY")
    print(f"{'='*70}")
    
    # Mode collapse analysis
    mc = analyze_mode_collapse(data['logit_signal'])
    print(f"\nMODE COLLAPSE:")
    print(f"   collapse_detected: {mc.get('collapse_detected', False)}")
    if mc.get('collapse_detected'):
        print(f"   collapse_batch: {mc['collapse_batch']}")
    
    if mc.get('unique_argmax_trend'):
        unique_vals = [x[1] for x in mc['unique_argmax_trend']]
        print(f"   unique_argmax: min={min(unique_vals)} max={max(unique_vals)} avg={sum(unique_vals)/len(unique_vals):.1f}")
    
    if mc.get('tok277_dominance'):
        dom_vals = [x[1] for x in mc['tok277_dominance']]
        print(f"   tok277_dominance: min={min(dom_vals):.1f}% max={max(dom_vals):.1f}% avg={sum(dom_vals)/len(dom_vals):.1f}%")
    
    # Hidden state collapse
    hc = analyze_hidden_collapse(data['hidden_cosine'])
    print(f"\nHIDDEN STATE SIMILARITY:")
    if hc.get('avg_cos_trend'):
        cos_vals = [x[1] for x in hc['avg_cos_trend']]
        print(f"   avg_cosine: min={min(cos_vals):.3f} max={max(cos_vals):.3f} avg={sum(cos_vals)/len(cos_vals):.3f}")
        if hc.get('high_similarity_batches'):
            print(f"   high_similarity_batches (>{hc['collapse_threshold']}): {len(hc['high_similarity_batches'])} batches")
    
    # Weight paradox
    wp = analyze_weight_paradox(data['token277_trace'], data['token277_post_opt'])
    print(f"\nWEIGHT PARADOX (negative grad + weight increases):")
    print(f"   paradox_count: {len(wp['paradox_batches'])}")
    if wp['paradox_batches']:
        for p in wp['paradox_batches'][:5]:
            print(f"   batch={p['batch']}: grad_sum={p['grad_sum']:.4f} delta_norm={p['delta']:+.6f}")
    
    # Loss trend
    if data['loss_stats']:
        losses = [(ls.batch, ls.loss_mean) for ls in data['loss_stats']]
        loss_vals = [x[1] for x in losses]
        print(f"\nLOSS:")
        print(f"   range: [{min(loss_vals):.4f}, {max(loss_vals):.4f}]")
        print(f"   first={loss_vals[0]:.4f}  last={loss_vals[-1]:.4f}")
    
    # Runtime data analysis
    if runtime_data:
        print(f"\n{'='*70}")
        print(f"  RUNTIME LOG DATA (training_run.log)")
        print(f"{'='*70}")
        
        # GRAD_ATTRIB analysis
        grad_attribs = runtime_data.get('grad_attrib', [])
        if grad_attribs:
            print(f"\nGRAD ATTRIBUTION ({len(grad_attribs)} entries):")
            zero_emb_count = sum(1 for ga in grad_attribs if ga.raw_emb_norm == 0.0)
            print(f"   raw_embedding_zero: {zero_emb_count}/{len(grad_attribs)}")
            
            # Direction distribution
            directions = [ga.direction for ga in grad_attribs]
            decrease = sum(1 for d in directions if 'DECREASE' in d)
            increase = sum(1 for d in directions if 'INCREASE' in d)
            print(f"   direction: DECREASE={decrease} INCREASE={increase}")
        
        # FA-LSE analysis
        fa_lse = runtime_data.get('fa_lse_summary', [])
        if fa_lse:
            print(f"\nFLASH ATTENTION LSE ({len(fa_lse)} entries):")
            anomalies = [lse for lse in fa_lse if lse.get('has_anomaly', False)]
            print(f"   anomaly_count: {len(anomalies)}")
            all_mins = [lse['lse_min'] for lse in fa_lse]
            all_maxs = [lse['lse_max'] for lse in fa_lse]
            print(f"   lse_min: [{min(all_mins):.1f}, {max(all_mins):.1f}]")
            print(f"   lse_max: [{min(all_maxs):.1f}, {max(all_maxs):.1f}]")
        
        # Issue96 position embeddings
        pos_embs = runtime_data.get('issue96_pos_emb', [])
        if pos_embs:
            print(f"\nPOSITION EMBEDDINGS ({len(pos_embs)} entries):")
            skipped = [p for p in pos_embs if 'SKIPPED' in p.status.upper()]
            learned = [p for p in pos_embs if 'LEARNED' in p.status.upper() or p.pos_emb_var > 0]
            print(f"   skipped: {len(skipped)}")
            print(f"   learned: {len(learned)}")
            if learned:
                print(f"   pos_emb_var: [{min(p.pos_emb_var for p in learned):.4f}, {max(p.pos_emb_var for p in learned):.4f}]")
        
        # UpdateProbe summary
        update_probes = runtime_data.get('update_probe_values', [])
        if update_probes:
            print(f"\nADAMW UPDATES ({len(update_probes)} steps):")
            # Group by parameter group and show aggregate stats
            groups = {}
            for up in update_probes:
                g = up.group
                if g not in groups:
                    groups[g] = {'grads': [], 'updates': []}
                if up.grad:
                    groups[g]['grads'].extend(up.grad)
                if up.update:
                    groups[g]['updates'].extend(up.update)
            for g, data in sorted(groups.items()):
                if data['grads'] and data['updates']:
                    avg_grad = sum(abs(x) for x in data['grads']) / len(data['grads'])
                    avg_upd = sum(abs(x) for x in data['updates']) / len(data['updates'])
                    max_grad = max(abs(x) for x in data['grads'])
                    max_upd = max(abs(x) for x in data['updates'])
                    print(f"   {g}: avg_grad={avg_grad:.2e} max_grad={max_grad:.2e} avg_update={avg_upd:.2e} max_update={max_upd:.2e}")
        
        # UpdateProbeState (momentum/variance)
        update_states = runtime_data.get('update_probe_state', [])
        if update_states:
            print(f"\nADAMW M/V STATE ({len(update_states)} entries):")
            # Group by parameter group and show aggregate stats
            groups = {}
            for us in update_states:
                g = us.group
                if g not in groups:
                    groups[g] = {'m': [], 'v': [], 'steps': set()}
                if us.m_state:
                    groups[g]['m'].extend(us.m_state)
                if us.v_state:
                    groups[g]['v'].extend(us.v_state)
                groups[g]['steps'].add(us.step)
            for g, data in sorted(groups.items()):
                if data['m'] and data['v']:
                    avg_m = sum(abs(x) for x in data['m']) / len(data['m'])
                    avg_v = sum(abs(x) for x in data['v']) / len(data['v'])
                    max_m = max(abs(x) for x in data['m'])
                    max_v = max(abs(x) for x in data['v'])
                    print(f"   {g}: steps={len(data['steps'])} avg_m={avg_m:.2e} max_m={max_m:.2e} avg_v={avg_v:.2e} max_v={max_v:.2e}")
        
        # FA-BWD-OUT (FlashAttention backward gradients)
        fa_bwd = runtime_data.get('fa_bwd_out', [])
        if fa_bwd:
            print(f"\nFLASH ATTN BACKWARD ({len(fa_bwd)} calls):")
            dq_maxs = [e.get('dQ_max', 0) for e in fa_bwd if 'dQ_max' in e]
            dk_maxs = [e.get('dK_max', 0) for e in fa_bwd if 'dK_max' in e]
            dv_maxs = [e.get('dV_max', 0) for e in fa_bwd if 'dV_max' in e]
            if dq_maxs:
                print(f"   dQ_max: [{min(dq_maxs):.6f}, {max(dq_maxs):.6f}]")
            if dk_maxs:
                print(f"   dK_max: [{min(dk_maxs):.6f}, {max(dk_maxs):.6f}]")
            if dv_maxs:
                print(f"   dV_max: [{min(dv_maxs):.6f}, {max(dv_maxs):.6f}]")
        
        # Issue91 embedding diagnostics
        issue91 = runtime_data.get('issue91', [])
        if issue91:
            print(f"\nEMBEDDING DIAGNOSTICS ({len(issue91)} entries):")
            tags = set(e['tag'] for e in issue91)
            for tag in sorted(tags)[:5]:
                entries = [e for e in issue91 if e['tag'] == tag]
                if entries:
                    mins = [e['min'] for e in entries if e['min'] is not None]
                    maxs = [e['max'] for e in entries if e['max'] is not None]
                    if mins and maxs:
                        print(f"   {tag}: range=[{min(mins):.4f}, {max(maxs):.4f}]")
        
        # Issue93 forward layer diagnostics
        issue93 = runtime_data.get('issue93', [])
        if issue93:
            print(f"\nFORWARD LAYER DIAGNOSTICS ({len(issue93)} entries):")
            stages = set(e['stage'] for e in issue93)
            for stage in sorted(stages)[:5]:
                entries = [e for e in issue93 if e['stage'] == stage]
                mins = [e['min'] for e in entries if e['min'] is not None]
                maxs = [e['max'] for e in entries if e['max'] is not None]
                if mins and maxs:
                    print(f"   {stage}: range=[{min(mins):.4f}, {max(maxs):.4f}]")
        
        # Issue94 RMSNorm diagnostics
        issue94 = runtime_data.get('issue94', [])
        if issue94:
            print(f"\nRMSNORM DIAGNOSTICS ({len(issue94)} entries):")
            stages = set(e['stage'] for e in issue94)
            for stage in sorted(stages)[:5]:
                entries = [e for e in issue94 if e['stage'] == stage]
                rms_mins = [e['rms_min'] for e in entries if e['rms_min'] > 0]
                rms_maxs = [e['rms_max'] for e in entries if e['rms_max'] > 0]
                if rms_mins and rms_maxs:
                    print(f"   {stage}: rms=[{min(rms_mins):.4f}, {max(rms_maxs):.4f}]")
        
        # Issue96 position embeddings
        issue96 = runtime_data.get('issue96', [])
        if issue96:
            print(f"\nPOSITION EMBEDDING STATUS ({len(issue96)} entries):")
            skipped = sum(1 for e in issue96 if e.get('skipped', False))
            active = len(issue96) - skipped
            print(f"   active: {active}  skipped: {skipped}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='GRIM-text Training Signal Diagnostic')
    parser.add_argument('log_file', nargs='?', default=None, help='Path to training log file')
    parser.add_argument('--runtime', '-r', type=str, default=None, help='Path to training_run.log (C++ runtime log)')
    parser.add_argument('--batch', '-b', type=int, help='Analyze specific batch')
    parser.add_argument('--last', '-l', type=int, default=10, help='Analyze last N batches')
    parser.add_argument('--summary', '-s', action='store_true', help='Show summary analysis')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    parser.add_argument('--output', '-o', type=str, default=None, help='Save output to file (auto-generates name if not specified)')
    parser.add_argument('--save', action='store_true', help='Auto-save to logs directory with timestamp')
    args = parser.parse_args()
    
    # Find log file
    log_file = args.log_file
    log_dir = Path('d:/G.R.I.M/resources/models/GRIM-text/training/logs')
    
    if not log_file:
        # Try to find the most recent training log
        if log_dir.exists():
            logs = sorted(log_dir.glob('training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
            # Skip training_run.log
            logs = [l for l in logs if l.name != 'training_run.log']
            if logs:
                log_file = str(logs[0])
                print(f"Using most recent log: {log_file}")
    
    if not log_file or not Path(log_file).exists():
        print("Error: No log file found. Please specify a log file path.")
        sys.exit(1)
    
    # Find runtime log
    runtime_log = args.runtime
    if not runtime_log:
        # Try to find training_run.log in the same directory or logs dir
        potential_paths = [
            Path(log_file).parent / 'training_run.log',
            log_dir / 'training_run.log',
        ]
        for p in potential_paths:
            if p.exists():
                runtime_log = str(p)
                print(f"Found runtime log: {runtime_log}")
                break
    
    # Determine output file
    output_file = None
    if args.output:
        output_file = Path(args.output)
    elif args.save:
        # Auto-generate filename based on source log
        source_name = Path(log_file).stem
        timestamp = source_name.replace('training_', '') if source_name.startswith('training_') else str(int(time.time() * 10000000))
        output_file = log_dir / f"diagnostic_{timestamp}.txt"
    
    # Set up output - either to file or stdout
    import io
    output_buffer = io.StringIO() if output_file else None
    
    def output(msg=''):
        """Print to both stdout and buffer if saving"""
        print(msg)
        if output_buffer:
            output_buffer.write(msg + '\n')
    
    output(f"Parsing {log_file}...")
    data = parse_log_file(log_file)
    
    output(f"Parsed: {len(data['logit_signal'])} LogitSignal, "
          f"{len(data['hidden_cosine'])} HiddenCosine, "
          f"{len(data['lm_head_norm'])} LMHeadNorm, "
          f"{len(data['token277_trace'])} Token277Trace, "
          f"{len(data['token277_post_opt'])} Token277PostOpt")
    
    # Parse runtime log if available
    runtime_data = None
    if runtime_log and Path(runtime_log).exists():
        output(f"Parsing runtime log {runtime_log}...")
        runtime_data = parse_runtime_log(runtime_log)
        output(f"Parsed: {len(runtime_data.get('grad_attrib', []))} GradAttrib, "
              f"{len(runtime_data.get('update_probe_values', []))} UpdateProbe, "
              f"{len(runtime_data.get('fa_lse_summary', []))} FA-LSE")
    
    # Capture print output for file saving
    import sys as _sys
    _original_stdout = _sys.stdout
    if output_file:
        _sys.stdout = io.StringIO()
    
    if args.batch:
        # Single batch analysis
        batch_data = correlate_batch(args.batch, data, runtime_data)
        if args.json:
            print(json.dumps(batch_data, indent=2))
        else:
            print_batch_report(batch_data)
    
    elif args.summary:
        # Summary analysis
        if args.json:
            result = {
                'mode_collapse': analyze_mode_collapse(data['logit_signal']),
                'hidden_collapse': analyze_hidden_collapse(data['hidden_cosine']),
                'weight_paradox': analyze_weight_paradox(data['token277_trace'], data['token277_post_opt']),
            }
            print(json.dumps(result, indent=2, default=str))
        else:
            print_summary_report(data, runtime_data)
    
    else:
        # Last N batches
        if data['logit_signal']:
            batches = sorted(set(ls.batch for ls in data['logit_signal']))[-args.last:]
            for batch in batches:
                batch_data = correlate_batch(batch, data, runtime_data)
                if args.json:
                    print(json.dumps(batch_data, indent=2))
                else:
                    print_batch_report(batch_data)
        
        # Always show summary
        if not args.json:
            print_summary_report(data, runtime_data)
    
    # Save to file if requested
    if output_file:
        report_content = _sys.stdout.getvalue()
        _sys.stdout = _original_stdout
        
        # Print to console too
        print(report_content)
        
        # Combine header and report
        full_content = output_buffer.getvalue() + report_content
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(full_content, encoding='utf-8')
        print(f"\nSaved diagnostic report to: {output_file}")

if __name__ == '__main__':
    main()
