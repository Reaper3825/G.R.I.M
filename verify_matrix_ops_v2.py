#!/usr/bin/env python3
"""
GRIM Model Verification Tool v2 - Mathematically Rigorous

=============================================================================
CUBLAS GEMM SEMANTICS (THE TRUTH)
=============================================================================

cuBLAS is COLUMN-MAJOR. Always. The API computes:

    C_cm = α * op(A_cm) @ op(B_cm) + β * C_cm

Where:
    - op(X) = X if CUBLAS_OP_N, X^T if CUBLAS_OP_T
    - A_cm is column-major with shape (lda, *) where * depends on op
    - If trans_A = N: op(A) has shape [M, K], so A_cm is stored as [lda, K] with lda >= M
    - If trans_A = T: op(A) has shape [M, K], so A_cm is stored as [lda, M] with lda >= K
    - Similarly for B, C

LEADING DIMENSION RULES (column-major):
    - lda = stride between columns of A in memory
    - For OP_N: A is [M, K], lda >= M (M rows, K cols, stride = lda)
    - For OP_T: A is [K, M], lda >= K (K rows, M cols, stride = lda)
    
    Memory layout for column-major A[M, K] with lda:
        A[i, j] is at offset: i + j * lda
        
ROW-MAJOR INTERPRETATION:
    If your data is actually row-major R[rows, cols], it appears to cuBLAS as:
        R_rm[rows, cols] ↔ R_cm[cols, rows] (same memory, different interpretation)
        
    So a row-major matrix R[M, N] with row stride N appears as column-major [N, M].
    
    To verify: check that ldX >= required_stride for the operation.
    
=============================================================================
"""

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from enum import Enum
import math


# =============================================================================
# Core Data Structures
# =============================================================================

class Layout(Enum):
    ROW_MAJOR = "row"
    COL_MAJOR = "col"

class TransposeOp(Enum):
    N = "CUBLAS_OP_N"  # No transpose
    T = "CUBLAS_OP_T"  # Transpose

@dataclass
class CacheOperation:
    """A memory cache operation (cudaMemcpy DeviceToDevice)"""
    file: str
    line_num: int
    src_ptr: str  # Source pointer
    dst_ptr: str  # Destination pointer (cached name)
    format: str   # Tensor format (BHSD, BSM, etc.)
    size_bytes: str  # Size expression
    context: str  # Surrounding code context
    is_forward: bool = True  # True if in forward pass
    cached_tensor_name: str = ""  # e.g., "cached_Q", "cached_attn_bhsd"
    
@dataclass
class TensorUsage:
    """Usage of a cached tensor"""
    file: str
    line_num: int
    tensor_name: str  # e.g., "cached_Q[layer]"
    operation: str  # "GEMM", "Flash_Attention", "convertTensor", etc.
    expected_format: str  # Format expected by this operation
    context: str  # Code snippet
    is_backward: bool = False
    
@dataclass
class FormatMismatch:
    """Detected format mismatch between cached and usage"""
    cached_op: CacheOperation
    usage: TensorUsage
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    suggested_fix: str = ""
    redundant_conversion: bool = False  # True if this triggers an extra conversion

@dataclass
class DataFlowIssue:
    """Data flow inefficiency or architectural problem"""
    issue_type: str  # "redundant_conversion", "suboptimal_cache_point", "format_mismatch"
    severity: str  # "CRITICAL", "MAJOR", "MINOR"
    description: str
    files_involved: List[str]
    line_numbers: List[int]
    impact: str  # Description of performance/correctness impact
    suggested_fix: str
    evidence: List[str]  # Code snippets or analysis results

@dataclass
class SymbolicDim:
    """A symbolic dimension that can be a constant or expression"""
    expr: str
    value: Optional[int] = None
    
    def __str__(self):
        if self.value is not None:
            return str(self.value)
        return self.expr
    
    def __eq__(self, other):
        if isinstance(other, SymbolicDim):
            if self.value is not None and other.value is not None:
                return self.value == other.value
            return self.expr == other.expr
        if isinstance(other, int):
            return self.value == other
        return False
    
    def evaluate(self, context: Dict[str, int]) -> Optional[int]:
        """
        Try to evaluate the symbolic expression given a context of known values.
        
        Examples:
            SymbolicDim("batch_size * seq_len").evaluate({"batch_size": 4, "seq_len": 512}) -> 2048
            SymbolicDim("3 * d_model").evaluate({"d_model": 768}) -> 2304
        """
        if self.value is not None:
            return self.value
        
        # Try to evaluate the expression
        expr = self.expr.lower()
        
        # Replace known symbols with their values
        eval_expr = expr
        for symbol, val in context.items():
            # Replace whole words only (avoid replacing "d_model" in "embed_d_model")
            eval_expr = re.sub(r'\b' + re.escape(symbol.lower()) + r'\b', str(val), eval_expr)
        
        # Try to evaluate as a Python expression (safe eval with limited scope)
        try:
            # Only allow arithmetic and known safe operations
            allowed_names = {"__builtins__": {}}
            result = eval(eval_expr, allowed_names)
            if isinstance(result, (int, float)):
                return int(result)
        except:
            pass
        
        return None

@dataclass
class TensorShape:
    """Symbolic tensor shape with layout information"""
    dims: Tuple[SymbolicDim, ...]
    layout: Layout = Layout.ROW_MAJOR
    
    @property
    def ndim(self):
        return len(self.dims)
    
    @property
    def rows(self):
        """First dimension for 2D"""
        return self.dims[0] if len(self.dims) >= 1 else None
    
    @property
    def cols(self):
        """Second dimension for 2D"""
        return self.dims[1] if len(self.dims) >= 2 else None
    
    def __str__(self):
        dim_str = ", ".join(str(d) for d in self.dims)
        return f"[{dim_str}] ({self.layout.value})"

@dataclass
class TensorInfo:
    """Full tensor metadata"""
    name: str
    shape: TensorShape
    semantic_role: str  # "weight", "activation", "gradient", "bias"
    
@dataclass
class GEMMCall:
    """Parsed cuBLAS GEMM call with full semantics"""
    file: str
    line_num: int
    raw_call: str
    
    # cuBLAS parameters (column-major semantics)
    trans_a: TransposeOp
    trans_b: TransposeOp
    m: SymbolicDim  # Rows of op(A) and C in column-major
    n: SymbolicDim  # Cols of op(B) and C in column-major
    k: SymbolicDim  # Cols of op(A) = Rows of op(B) (contraction dim)
    
    # Pointers and strides (column-major leading dimensions)
    ptr_a: str
    lda: SymbolicDim  # Column stride for A
    ptr_b: str  
    ldb: SymbolicDim  # Column stride for B
    ptr_c: str
    ldc: SymbolicDim  # Column stride for C
    
    # Scalars
    alpha: str
    beta: str
    
    # Context
    comment: str = ""
    preceding_comments: str = ""
    
@dataclass
class GEMMVerification:
    """Verification result for a GEMM call"""
    call: GEMMCall
    
    # Column-major shapes as cuBLAS sees them
    # These are the STORAGE shapes, not the mathematical shapes
    storage_a: Tuple[SymbolicDim, SymbolicDim]  # (physical_rows, physical_cols) in col-major
    storage_b: Tuple[SymbolicDim, SymbolicDim]
    storage_c: Tuple[SymbolicDim, SymbolicDim]
    
    # Mathematical operation shapes (what cuBLAS computes)
    # C[M,N] = op(A)[M,K] @ op(B)[K,N]
    math_m: SymbolicDim
    math_n: SymbolicDim
    math_k: SymbolicDim
    
    # Leading dimension requirements
    lda_required: SymbolicDim
    ldb_required: SymbolicDim
    ldc_required: SymbolicDim
    
    # Operation semantics
    effective_op: str  # Mathematical description
    op_category: str  # "forward", "weight_grad", "activation_grad", "unknown"
    
    # Verification status
    is_valid: bool
    ld_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    analysis: str = ""


# =============================================================================
# Known Model Configuration
# =============================================================================

GRIM_CONFIG = {
    # Model dimensions
    "d_model": 768,
    "d_ff": 3072,  # 4 * d_model
    "num_heads": 12,
    "head_dim": 64,  # d_model / num_heads
    "vocab_size": 13811,
    "max_seq_len": 2048,
    "num_layers": 12,
    
    # All tensors in GRIM are ROW-MAJOR
    "default_layout": Layout.ROW_MAJOR,
}

# Known tensor shapes (name pattern -> shape template)
KNOWN_TENSORS = {
    # Embeddings
    r"embedding|token_embed": ("vocab_size", "d_model"),
    r"position_embed": ("max_seq_len", "d_model"),
    
    # QKV projection
    r"W_qkv|w_qkv": ("3*d_model", "d_model"),  # Fused QKV
    r"W_q|w_q": ("d_model", "d_model"),
    r"W_k|w_k": ("d_model", "d_model"),
    r"W_v|w_v": ("d_model", "d_model"),
    r"W_o|w_o|W_out|w_out": ("d_model", "d_model"),
    
    # FFN
    r"W1|w1|ffn.*w1|ffnW1": ("d_ff", "d_model"),  # Up projection
    r"W2|w2|ffn.*w2|ffnW2": ("d_model", "d_ff"),  # Down projection
    
    # LM Head
    r"lm_head|weights": ("vocab_size", "d_model"),
    
    # Activations (per-token)
    r"encoder_output|hidden|ln.*output": ("total_tokens", "d_model"),
    r"ffn_output|ffn_hidden": ("total_tokens", "d_ff"),
    r"logits": ("total_tokens", "vocab_size"),
    r"qkv_out": ("total_tokens", "3*d_model"),
    
    # Gradients match their forward counterparts
    r"grad_.*weight|.*_grads": None,  # Shape matches weight
    r"grad_.*input|grad_.*output": None,  # Shape matches activation
}


# =============================================================================
# cuBLAS GEMM Analysis (Correct Semantics)
# =============================================================================

def analyze_gemm_call(call: GEMMCall) -> GEMMVerification:
    """
    Analyze a cuBLAS GEMM call with CORRECT column-major semantics.
    
    cuBLAS computes (in column-major):
        C[M, N] = α * op(A)[M, K] @ op(B)[K, N] + β * C[M, N]
    
    Where for matrix X with leading dimension ldx:
        - If trans = N: X is stored as [ldx, cols], we use [rows, cols] where ldx >= rows
        - If trans = T: X is stored as [ldx, rows], we use [cols, rows]^T = [rows, cols]
    
    Leading dimension requirements:
        - trans_A = N: lda >= M (A stored as column-major [lda, K])
        - trans_A = T: lda >= K (A stored as column-major [lda, M], transposed to [M, K])
        - trans_B = N: ldb >= K (B stored as column-major [ldb, N])
        - trans_B = T: ldb >= N (B stored as column-major [ldb, K], transposed to [K, N])
        - ldc >= M always (C stored as column-major [ldc, N])
    """
    issues = []
    warnings = []
    
    m = call.m
    n = call.n
    k = call.k
    
    # ==========================================================================
    # LEADING DIMENSION REQUIREMENTS (the core correctness check)
    # ==========================================================================
    
    # For A: op(A) must be [M, K]
    if call.trans_a == TransposeOp.N:
        # A is column-major [lda, K], no transpose, so A[i,j] at i + j*lda
        # We need rows = M, so lda >= M
        lda_required = m
        storage_a = (call.lda, k)  # [lda, K] physical storage
    else:  # trans_A = T
        # A is column-major [lda, M], transposed to get [M, K]
        # Original has K rows, so lda >= K
        lda_required = k
        storage_a = (call.lda, m)  # [lda, M] physical storage
    
    # For B: op(B) must be [K, N]
    if call.trans_b == TransposeOp.N:
        # B is column-major [ldb, N], no transpose
        # We need rows = K, so ldb >= K
        ldb_required = k
        storage_b = (call.ldb, n)  # [ldb, N] physical storage
    else:  # trans_B = T
        # B is column-major [ldb, K], transposed to get [K, N]
        # Original has N rows, so ldb >= N
        ldb_required = n
        storage_b = (call.ldb, k)  # [ldb, K] physical storage
    
    # For C: always [M, N] in column-major, so C[i,j] at i + j*ldc
    # Need rows = M, so ldc >= M
    ldc_required = m
    storage_c = (call.ldc, n)  # [ldc, N] physical storage
    
    # ==========================================================================
    # VERIFY LEADING DIMENSIONS
    # ==========================================================================
    
    ld_valid = True
    
    def check_ld(ld_actual: SymbolicDim, ld_req: SymbolicDim, name: str) -> bool:
        nonlocal ld_valid
        if ld_actual.value is not None and ld_req.value is not None:
            if ld_actual.value < ld_req.value:
                issues.append(f"{name}={ld_actual.value} but need >= {ld_req.value}")
                ld_valid = False
                return False
        return True
    
    check_ld(call.lda, lda_required, "lda")
    check_ld(call.ldb, ldb_required, "ldb")
    check_ld(call.ldc, ldc_required, "ldc")
    
    # ==========================================================================
    # INFER OPERATION CATEGORY
    # ==========================================================================
    
    op_category = infer_operation_category_v2(call)
    
    # ==========================================================================
    # BUILD EFFECTIVE OPERATION STRING
    # ==========================================================================
    
    op_a = "^T" if call.trans_a == TransposeOp.T else ""
    op_b = "^T" if call.trans_b == TransposeOp.T else ""
    
    # cuBLAS: C[M,N] = op(A)[M,K] @ op(B)[K,N]
    effective_op = f"C[{m}, {n}] = op(A){op_a}[{m}, {k}] @ op(B){op_b}[{k}, {n}]"
    
    # ==========================================================================
    # BUILD ANALYSIS
    # ==========================================================================
    
    analysis_lines = [
        f"cuBLAS GEMM: trans_A={call.trans_a.name}, trans_B={call.trans_b.name}",
        f"  M={m}, N={n}, K={k}",
        f"  A: ptr={call.ptr_a}, lda={call.lda} (required >= {lda_required})",
        f"  B: ptr={call.ptr_b}, ldb={call.ldb} (required >= {ldb_required})",
        f"  C: ptr={call.ptr_c}, ldc={call.ldc} (required >= {ldc_required})",
        f"  α={call.alpha}, β={call.beta}",
        f"Category: {op_category}",
    ]
    
    return GEMMVerification(
        call=call,
        storage_a=storage_a,
        storage_b=storage_b,
        storage_c=storage_c,
        math_m=m,
        math_n=n,
        math_k=k,
        lda_required=lda_required,
        ldb_required=ldb_required,
        ldc_required=ldc_required,
        effective_op=effective_op,
        op_category=op_category,
        is_valid=len(issues) == 0,
        ld_valid=ld_valid,
        issues=issues,
        warnings=warnings,
        analysis='\n'.join(analysis_lines)
    )


def infer_operation_category_v2(call: GEMMCall) -> str:
    """
    Infer operation category using multiple signals, not just shapes.
    
    Signals used:
    1. β value: β=0 suggests forward/overwrite, β=1 suggests gradient accumulation
    2. Transpose pattern: forward often uses OP_T/OP_N or OP_N/OP_N
    3. Pointer names: fallback heuristic
    4. Dimension patterns: M,N,K relationships
    """
    
    # Parse beta value
    beta_str = call.beta.strip()
    beta_is_zero = any(x in beta_str for x in ['&zero', '0.0', '0.f', '&beta_zero'])
    beta_is_one = any(x in beta_str for x in ['&one', '1.0', '1.f', '&beta_one'])
    
    # Transpose patterns
    trans_pattern = (call.trans_a.name, call.trans_b.name)
    
    # Pointer name analysis
    ptr_a = call.ptr_a.lower()
    ptr_b = call.ptr_b.lower()
    ptr_c = call.ptr_c.lower()
    
    # Check for gradient indicators
    is_grad_output = 'grad' in ptr_c or 'd_' in ptr_c or '_grad' in ptr_c
    has_grad_input = 'grad' in ptr_a or 'grad' in ptr_b or 'd_' in ptr_a or 'd_' in ptr_b
    
    # Check for weight indicators
    is_weight_output = any(w in ptr_c for w in ['weight', 'w_', 'w1', 'w2', 'wq', 'wk', 'wv', 'wo'])
    has_weight_input = any(w in ptr_a or w in ptr_b for w in ['weight', 'w_', 'w1', 'w2', 'wq', 'wk', 'wv', 'wo'])
    
    # Decision logic
    
    # Weight gradient: computing dW = X^T @ dY or similar
    # Pattern: β=1 (accumulate), one input is activation, one is gradient
    if is_weight_output or (is_grad_output and beta_is_one and has_grad_input):
        return "weight_grad"
    
    # Activation gradient: computing dX = dY @ W or similar
    # Pattern: β=0 (overwrite), output is gradient, weight is input
    if is_grad_output and has_weight_input and not is_weight_output:
        return "activation_grad"
    
    # Forward pass: computing Y = X @ W or similar
    # Pattern: β=0 (overwrite), no gradients involved
    if beta_is_zero and not is_grad_output and not has_grad_input:
        return "forward"
    
    # Gradient accumulation
    if beta_is_one:
        return "gradient_accum"
    
    # Unknown
    return "unknown"


# =============================================================================
# Reshape/Permutation Verification (with stride analysis)
# =============================================================================

@dataclass
class ReshapeOp:
    """Tensor reshape/permutation operation with stride verification"""
    file: str
    line_num: int
    
    src_format: str  # e.g., "BHSD"
    dst_format: str  # e.g., "BSHD"
    
    # Dimension values
    dims: Dict[str, SymbolicDim]
    
    # Stride analysis
    src_strides: Optional[List[int]] = None
    dst_strides: Optional[List[int]] = None
    
    # Verification
    is_bijective: bool = True
    is_contiguous_preserving: bool = True
    preserves_elements: bool = True
    issues: List[str] = field(default_factory=list)


def verify_reshape_with_strides(src_fmt: str, dst_fmt: str, 
                                 dims: Dict[str, int]) -> Tuple[bool, bool, List[str]]:
    """
    Verify reshape is correct using index algebra and stride analysis.
    
    For a tensor with format BHSD and dims B=2, H=12, S=1024, D=64:
        - Strides (row-major): [H*S*D, S*D, D, 1] = [786432, 64, 64, 1]
        - Element at [b,h,s,d] is at offset: b*786432 + h*64 + s*64 + d
        
    Transpose BHSD -> BSHD:
        - New strides: [S*H*D, H*D, D, 1] = [786432, 768, 64, 1]
        - Element at [b,s,h,d] is at offset: b*786432 + s*768 + h*64 + d
        
    This is valid iff the same elements are accessed (bijective).
    """
    issues = []
    
    src_chars = list(src_fmt)
    dst_chars = list(dst_fmt)
    
    # Check permutation validity
    if sorted(src_chars) != sorted(dst_chars):
        issues.append(f"Not a permutation: {src_fmt} -> {dst_fmt}")
        return False, False, issues
    
    # Check for duplicates
    if len(src_chars) != len(set(src_chars)):
        issues.append(f"Duplicate axes in source: {src_fmt}")
        return False, False, issues
    
    if len(dst_chars) != len(set(dst_chars)):
        issues.append(f"Duplicate axes in destination: {dst_fmt}")
        return False, False, issues
    
    # Compute permutation
    try:
        perm = [src_chars.index(c) for c in dst_chars]
    except ValueError:
        issues.append(f"Incompatible formats: {src_fmt} -> {dst_fmt}")
        return False, False, issues
    
    # Check stride compatibility if we have concrete dimensions
    is_contiguous = True
    if all(c in dims for c in src_chars):
        # Compute source strides (row-major: rightmost is 1)
        src_strides = []
        stride = 1
        for c in reversed(src_chars):
            src_strides.insert(0, stride)
            stride *= dims.get(c, 1)
        
        # Compute destination strides after permutation
        dst_strides = [src_strides[perm[i]] for i in range(len(perm))]
        
        # Check if destination is contiguous (strides decrease left to right)
        for i in range(len(dst_strides) - 1):
            if dst_strides[i] < dst_strides[i + 1]:
                # Not contiguous - may need explicit copy
                is_contiguous = False
                issues.append(f"Non-contiguous after transpose (strides: {dst_strides})")
                break
    
    return True, is_contiguous, issues


# =============================================================================
# Attention Verification (Structural)
# =============================================================================
# =============================================================================

@dataclass
class AttentionBlock:
    """Full attention block with all components"""
    file: str
    line_start: int
    line_end: int
    
    # Components found
    has_qkv_projection: bool = False
    has_scale_factor: bool = False
    has_softmax: bool = False
    has_causal_mask: bool = False
    has_dropout: bool = False
    uses_flash_attention: bool = False
    
    # Shapes
    qkv_shape: Optional[str] = None  # Should be (B, H, S, D) or similar
    score_shape: Optional[str] = None  # Should be (B, H, S, S)
    
    # Scale factor
    scale_value: Optional[str] = None
    scale_correct: bool = False  # 1/sqrt(d_k)
    
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def verify_attention_block(file_path: str, lines: List[str], start_line: int) -> Optional[AttentionBlock]:
    """
    Verify a complete attention mechanism, not just individual operations.
    
    Standard attention:
        1. QKV projection: X @ W_qkv -> Q, K, V
        2. Reshape to heads: [B, S, 3*D] -> [B, H, S, D//H] for Q, K, V
        3. Scores = Q @ K^T / sqrt(d_k)
        4. Apply causal mask (optional)
        5. Attention weights = softmax(scores)
        6. Apply dropout (optional)
        7. Output = weights @ V
        8. Reshape and project: [B, H, S, D//H] -> [B, S, D] @ W_o
    
    Flash Attention fuses steps 3-7 into a single kernel.
    """
    block = AttentionBlock(
        file=file_path,
        line_start=start_line,
        line_end=min(start_line + 200, len(lines))
    )
    
    context = '\n'.join(lines[start_line:block.line_end])
    
    # FILTER OUT FALSE POSITIVES: Skip if this is clearly not attention code
    false_positive_patterns = [
        r'logger\.log|std::cout|printf|fprintf',  # Logging statements
        r'std::to_string',  # String formatting
        r'clip_min.*clip_max',  # Quantization config
        r'parseModuleLogLevel|ModuleLogOverride',  # Logging config
        r'Xavier|Glorot.*initialization',  # Weight initialization
        r'shuffle.*epoch|batching|curriculum',  # Training orchestration
    ]
    if any(re.search(pattern, context) for pattern in false_positive_patterns):
        return None  # Skip this block - not attention code
    
    # Check for Flash Attention
    if re.search(r'flash.*attention|FlashAttention', context, re.IGNORECASE):
        block.uses_flash_attention = True
        # Flash attention handles scale/softmax/mask internally
        block.has_scale_factor = True
        block.has_softmax = True
    
    # Look for QKV projection
    qkv_match = re.search(r'(W_qkv|w_qkv|qkv.*proj)', context, re.IGNORECASE)
    if qkv_match:
        block.has_qkv_projection = True
    
    # Look for scale factor
    scale_patterns = [
        r'1\.0f?\s*/\s*sqrtf?\s*\(\s*(?:static_cast<float>\s*\()?\s*(\w+)',
        r'rsqrtf?\s*\(\s*(?:static_cast<float>\s*\()?\s*(\w+)',
        r'scale\s*=\s*([^;]+)',
        r'\*\s*scale\b',
    ]
    for pattern in scale_patterns:
        match = re.search(pattern, context)
        if match:
            block.has_scale_factor = True
            if match.lastindex and match.lastindex >= 1:
                block.scale_value = match.group(1)
            break
    
    # Verify scale factor value
    if block.scale_value:
        if 'head_dim' in block.scale_value.lower() or 'd_head' in block.scale_value.lower():
            block.scale_correct = True
        elif block.scale_value == '64':  # head_dim = d_model/num_heads = 768/12 = 64
            block.scale_correct = True
        elif '8.0' in block.scale_value or '8' == block.scale_value.strip():
            # sqrt(64) = 8, so 1/8 is correct
            block.scale_correct = True
    
    # Look for softmax
    if re.search(r'softmax|Softmax', context, re.IGNORECASE):
        block.has_softmax = True
    
    # Look for causal mask
    if re.search(r'causal|mask.*-inf|-1e9|k_idx\s*>\s*q_idx|triu|triangular', context, re.IGNORECASE):
        block.has_causal_mask = True
    
    # Look for dropout
    if re.search(r'dropout|Dropout', context, re.IGNORECASE):
        block.has_dropout = True
    
    # Look for attention score shape comment
    score_shape_match = re.search(r'\[.*batch.*head.*seq.*seq.*\]|\(B,\s*H,\s*S,\s*S\)', context, re.IGNORECASE)
    if score_shape_match:
        block.score_shape = score_shape_match.group(0)
    
    # Verify completeness
    if not block.uses_flash_attention:
        if not block.has_scale_factor:
            block.issues.append("Missing scale factor (1/sqrt(d_k))")
        elif not block.scale_correct:
            block.warnings.append(f"Scale factor '{block.scale_value}' - verify it equals 1/sqrt(head_dim)")
        
        if not block.has_softmax:
            block.issues.append("Missing softmax operation")
    
    return block


def verify_reshape_bijective(src_fmt: str, dst_fmt: str) -> Tuple[bool, List[str]]:
    """
    Verify reshape is bijective using index algebra.
    
    For BHSD -> BSHD:
        src_index = b*H*S*D + h*S*D + s*D + d
        dst_index = b*S*H*D + s*H*D + h*D + d
        
    This is bijective iff the mapping is a permutation of axes.
    """
    issues = []
    
    src_chars = list(src_fmt)
    dst_chars = list(dst_fmt)
    
    # Must have same characters (permutation)
    if sorted(src_chars) != sorted(dst_chars):
        issues.append(f"Not a permutation: {src_fmt} -> {dst_fmt} have different axes")
        return False, issues
    
    # Check for duplicates
    if len(src_chars) != len(set(src_chars)):
        issues.append(f"Source format has duplicate axes: {src_fmt}")
        return False, issues
    
    if len(dst_chars) != len(set(dst_chars)):
        issues.append(f"Destination format has duplicate axes: {dst_fmt}")
        return False, issues
    
    return True, issues


def compute_permutation(src_fmt: str, dst_fmt: str) -> List[int]:
    """Compute the axis permutation from src to dst format"""
    try:
        return [src_fmt.index(c) for c in dst_fmt]
    except ValueError:
        # Formats are incompatible (not a pure transpose)
        return []


# =============================================================================
# Initializer Verification
# =============================================================================

@dataclass
class InitializerVerification:
    """Weight initialization verification"""
    file: str
    line_num: int
    
    init_type: str  # "xavier_uniform", "xavier_normal", "he", "normal"
    fan_in: Optional[int]
    fan_out: Optional[int]
    
    # Expected values
    expected_std: Optional[float]
    expected_bound: Optional[float]
    
    # Actual values found
    actual_std: Optional[str]
    actual_bound: Optional[str]
    
    is_correct: bool = False
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def verify_initializer(init_type: str, fan_in: Optional[int], fan_out: Optional[int],
                       actual_value: Optional[str]) -> InitializerVerification:
    """
    Verify weight initialization follows proper formulas.
    
    Xavier Uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    Xavier Normal: N(0, sqrt(2/(fan_in+fan_out)))
    He Normal: N(0, sqrt(2/fan_in))
    He Uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in))
    """
    result = InitializerVerification(
        file="", line_num=0,
        init_type=init_type,
        fan_in=fan_in,
        fan_out=fan_out,
        expected_std=None,
        expected_bound=None,
        actual_std=actual_value,
        actual_bound=actual_value,
    )
    
    if fan_in is None:
        result.warnings.append("fan_in not specified")
    if fan_out is None and 'xavier' in init_type.lower():
        result.warnings.append("fan_out not specified for Xavier init")
    
    if fan_in is not None and fan_out is not None:
        if 'xavier' in init_type.lower():
            if 'uniform' in init_type.lower():
                result.expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
            else:  # normal
                result.expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        elif 'he' in init_type.lower() or 'kaiming' in init_type.lower():
            if 'uniform' in init_type.lower():
                result.expected_bound = math.sqrt(6.0 / fan_in)
            else:  # normal
                result.expected_std = math.sqrt(2.0 / fan_in)
    
    return result


# =============================================================================
# Main Verifier Class
# =============================================================================

class GRIMVerifier:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.config = GRIM_CONFIG.copy()
        
        self.gemm_calls: List[GEMMCall] = []
        self.gemm_results: List[GEMMVerification] = []
        self.attention_blocks: List[AttentionBlock] = []
        self.reshape_ops: List[ReshapeOp] = []
        self.cache_ops: List[CacheOperation] = []
        self.tensor_usages: List[TensorUsage] = []
        self.format_mismatches: List[FormatMismatch] = []
        self.dataflow_issues: List[DataFlowIssue] = []
        
    def scan_files(self) -> None:
        """Scan all CUDA/C++ files"""
        print(f"🔍 Scanning {self.model_dir}...")
        
        for ext in ['*.cu', '*.cpp', '*.hpp', '*.cuh']:
            for file_path in self.model_dir.rglob(ext):
                self._analyze_file(file_path)
        
        print(f"✓ Found {len(self.gemm_calls)} cuBLAS GEMM calls")
        print(f"✓ Found {len(self.attention_blocks)} attention blocks")
        print(f"✓ Found {len(self.reshape_ops)} reshape operations")
        print(f"✓ Found {len(self.cache_ops)} cache operations")
        print(f"✓ Found {len(self.tensor_usages)} tensor usages")
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"⚠ Could not read {file_path}: {e}")
            return
        
        rel_path = str(file_path.relative_to(self.model_dir))
        
        # Find GEMM calls
        self._find_gemm_calls(rel_path, lines)
        
        # Find attention blocks
        self._find_attention_blocks(rel_path, lines)
        
        # Find reshape operations
        self._find_reshape_ops(rel_path, lines)
        
        # Find cache operations
        self._find_cache_ops(rel_path, lines)
        
        # Find tensor usages
        self._find_tensor_usages(rel_path, lines)
    
    def _find_gemm_calls(self, file_path: str, lines: List[str]) -> None:
        """
        Find and parse all cublasSgemm calls with robust multi-line/macro support.
        
        Handles:
        - Multi-line calls
        - Comments (// and /* */)
        - Macros that expand to cublasSgemm
        - Preprocessor directives
        """
        
        content = '\n'.join(lines)
        
        # First pass: remove block comments (but preserve line count)
        content_no_block_comments = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), content, flags=re.DOTALL)
        
        # Pattern for cublasSgemm call (allow whitespace/newlines before paren)
        pattern = r'cublas[SD]gemm\s*\('
        
        for match in re.finditer(pattern, content_no_block_comments):
            start_pos = match.start()
            line_num = content[:start_pos].count('\n')
            
            # Extract full call with robust paren matching
            call_text = self._extract_call_robust(content_no_block_comments, start_pos)
            if call_text:
                gemm = self._parse_gemm_call(file_path, line_num + 1, call_text, lines)
                if gemm:
                    self.gemm_calls.append(gemm)
    
    def _extract_call_robust(self, content: str, start_pos: int) -> Optional[str]:
        """
        Extract complete function call with robust handling of:
        - Nested parentheses
        - String literals (that might contain parens)
        - Character literals
        - Line comments
        """
        # Find function name start
        func_start = start_pos
        while func_start > 0 and (content[func_start - 1].isalnum() or content[func_start - 1] == '_'):
            func_start -= 1
        
        # Find opening paren (skip whitespace)
        paren_start = start_pos
        while paren_start < len(content) and content[paren_start] != '(':
            paren_start += 1
        
        if paren_start >= len(content):
            return None
        
        # Now extract until matching close paren, handling strings and comments
        depth = 1
        pos = paren_start + 1
        in_string = False
        in_char = False
        escape_next = False
        
        while pos < len(content) and depth > 0:
            char = content[pos]
            
            # Handle escape sequences
            if escape_next:
                escape_next = False
                pos += 1
                continue
            
            if char == '\\':
                escape_next = True
                pos += 1
                continue
            
            # Handle string literals
            if char == '"' and not in_char:
                in_string = not in_string
                pos += 1
                continue
            
            # Handle character literals
            if char == "'" and not in_string:
                in_char = not in_char
                pos += 1
                continue
            
            # Skip if inside string or char literal
            if in_string or in_char:
                pos += 1
                continue
            
            # Handle line comments (// until end of line)
            if char == '/' and pos + 1 < len(content) and content[pos + 1] == '/':
                # Skip to end of line
                while pos < len(content) and content[pos] != '\n':
                    pos += 1
                pos += 1
                continue
            
            # Handle parentheses
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            pos += 1
        
        if depth == 0:
            return content[func_start:pos]
        
        return None
    
    def _parse_gemm_call(self, file_path: str, line_num: int, 
                          call_text: str, lines: List[str]) -> Optional[GEMMCall]:
        """
        Parse a cuBLAS GEMM call into structured form.
        
        Handles multi-line calls, comments, and various formatting styles.
        """
        
        # Remove all comments (line and block)
        call_clean = re.sub(r'//.*$', '', call_text, flags=re.MULTILINE)
        call_clean = re.sub(r'/\*.*?\*/', ' ', call_clean, flags=re.DOTALL)
        
        # Normalize whitespace (collapse multiple spaces/newlines)
        call_clean = re.sub(r'\s+', ' ', call_clean)
        
        # Find arguments
        paren_start = call_clean.find('(')
        paren_end = call_clean.rfind(')')
        if paren_start == -1 or paren_end == -1:
            return None
        
        args_str = call_clean[paren_start+1:paren_end]
        args = self._split_args_robust(args_str)
        
        # cublasSgemm signature:
        # (handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        if len(args) < 14:
            # Try to be lenient - sometimes there are extra commas or formatting issues
            # Filter out empty args
            args = [a for a in args if a.strip()]
            if len(args) < 14:
                return None
        
        try:
            trans_a = self._parse_transpose(args[1])
            trans_b = self._parse_transpose(args[2])
            m = self._parse_dim(args[3])
            n = self._parse_dim(args[4])
            k = self._parse_dim(args[5])
            # args[6] is &alpha
            ptr_a = args[7].strip()
            lda = self._parse_dim(args[8])
            ptr_b = args[9].strip()
            ldb = self._parse_dim(args[10])
            # args[11] is &beta
            ptr_c = args[12].strip()
            ldc = self._parse_dim(args[13])
            
            # Get preceding comments for context
            preceding = '\n'.join(lines[max(0, line_num-10):line_num])
            
            return GEMMCall(
                file=file_path,
                line_num=line_num,
                raw_call=call_text,
                trans_a=trans_a,
                trans_b=trans_b,
                m=m, n=n, k=k,
                ptr_a=ptr_a, lda=lda,
                ptr_b=ptr_b, ldb=ldb,
                ptr_c=ptr_c, ldc=ldc,
                alpha=args[6].strip(),
                beta=args[11].strip(),
                preceding_comments=preceding
            )
        except Exception as e:
            return None
    
    def _split_args_robust(self, args_str: str) -> List[str]:
        """
        Split arguments respecting:
        - Nested parentheses (function calls)
        - String literals
        - Character literals
        - Template angle brackets
        """
        args = []
        current = []
        paren_depth = 0
        angle_depth = 0
        in_string = False
        in_char = False
        escape_next = False
        
        i = 0
        while i < len(args_str):
            char = args_str[i]
            
            # Handle escape sequences
            if escape_next:
                current.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                current.append(char)
                escape_next = True
                i += 1
                continue
            
            # Handle string literals
            if char == '"' and not in_char:
                in_string = not in_string
                current.append(char)
                i += 1
                continue
            
            # Handle character literals
            if char == "'" and not in_string:
                in_char = not in_char
                current.append(char)
                i += 1
                continue
            
            # If inside string or char, just append
            if in_string or in_char:
                current.append(char)
                i += 1
                continue
            
            # Handle depth counters
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == '<':
                angle_depth += 1
                current.append(char)
            elif char == '>':
                angle_depth -= 1
                current.append(char)
            elif char == ',' and paren_depth == 0 and angle_depth == 0:
                # Top-level comma - split here
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
            
            i += 1
        
        if current:
            args.append(''.join(current).strip())
        
        return args
    
    def _parse_transpose(self, arg: str) -> TransposeOp:
        """Parse transpose operation"""
        arg = arg.strip()
        if 'OP_T' in arg:
            return TransposeOp.T
        return TransposeOp.N
    
    def _parse_dim(self, arg: str) -> SymbolicDim:
        """Parse a dimension argument and try to evaluate it"""
        arg = arg.strip()
        
        # Try to evaluate as integer constant
        try:
            val = int(arg)
            return SymbolicDim(arg, val)
        except ValueError:
            pass
        
        # Build evaluation context from known config
        eval_context = {
            'd_model': self.config['d_model'],
            'dmodel': self.config['d_model'],
            'd_ff': self.config['d_ff'],
            'dff': self.config['d_ff'],
            'vocab_size': self.config['vocab_size'],
            'vocabsize': self.config['vocab_size'],
            'num_heads': self.config['num_heads'],
            'numheads': self.config['num_heads'],
            'head_dim': self.config['head_dim'],
            'headdim': self.config['head_dim'],
            'max_seq_len': self.config['max_seq_len'],
            'maxseqlen': self.config['max_seq_len'],
        }
        
        # Create SymbolicDim and try to evaluate
        dim = SymbolicDim(arg, None)
        evaluated = dim.evaluate(eval_context)
        
        if evaluated is not None:
            dim.value = evaluated
        
        return dim
    
    def _find_attention_blocks(self, file_path: str, lines: List[str]) -> None:
        """Find and verify attention mechanism blocks"""
        
        content = '\n'.join(lines)
        
        # Look for attention-related patterns
        attention_starts = []
        for pattern in [r'attention', r'AttentionScores', r'Q.*K.*V', r'qkv']:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count('\n')
                attention_starts.append(line_num)
        
        # Deduplicate and verify each block
        seen_starts = set()
        for start in sorted(attention_starts):
            # Skip if too close to a previous start
            if any(abs(start - s) < 50 for s in seen_starts):
                continue
            
            block = verify_attention_block(file_path, lines, start)
            if block and (block.has_qkv_projection or block.has_scale_factor or block.has_softmax):
                self.attention_blocks.append(block)
                seen_starts.add(start)
    
    def _find_reshape_ops(self, file_path: str, lines: List[str]) -> None:
        """Find tensor reshape operations"""
        
        content = '\n'.join(lines)
        
        # Pattern for convertTensor ACTUAL CALLS (handles multi-line)
        # Match: convertTensor(...TensorFormat::XXX...TensorFormat::YYY...)
        pattern = r'convertTensor\s*\(\s*[^;]*?TensorFormat::(\w+)\s*,\s*[^;]*?TensorFormat::(\w+)'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            line_num = content[:match.start()].count('\n')
            src_fmt = match.group(1)
            dst_fmt = match.group(2)
            
            # Get the actual line to check context
            line_text = lines[line_num] if line_num < len(lines) else ""
            
            # Skip if this is inside a function signature/definition
            # Look at the 3 lines before the match for context
            context_start = max(0, line_num - 3)
            context_lines = lines[context_start:line_num + 1]
            context = ' '.join(context_lines)
            
            # Skip function definitions (have return type or 'void' keyword)
            if re.search(r'\b(void|inline|static|__device__|__host__)\s+\w+\s*\(', context):
                continue
            
            # Skip if inside a dispatcher condition (if statement checking formats)
            if 'if (srcFmt ==' in context or 'if (dstFmt ==' in context:
                continue
            
            # Use stride-aware verification with GRIM dimensions
            grim_dims = {
                'B': 1,  # Batch (variable)
                'H': self.config['num_heads'],  # 12
                'S': self.config['max_seq_len'],  # 2048 (variable)
                'D': self.config['head_dim'],  # 64
                'M': self.config['d_model'],  # 768
            }
            
            is_bij, is_contig, issues = verify_reshape_with_strides(src_fmt, dst_fmt, grim_dims)
            
            self.reshape_ops.append(ReshapeOp(
                file=file_path,
                line_num=line_num + 1,
                src_format=src_fmt,
                dst_format=dst_fmt,
                dims={k: SymbolicDim(k, v) for k, v in grim_dims.items()},
                is_bijective=is_bij,
                is_contiguous_preserving=is_contig,
                preserves_elements=is_bij,
                issues=issues
            ))
    
    def _find_cache_ops(self, file_path: str, lines: List[str]) -> None:
        """Find cudaMemcpy DeviceToDevice operations (caching)"""
        content = '\n'.join(lines)
        
        # Pattern for cudaMemcpy/cudaMemcpyAsync with DeviceToDevice
        cache_pattern = r'cudaMemcpy(?:Async)?\s*\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*cudaMemcpyDeviceToDevice'
        
        for match in re.finditer(cache_pattern, content):
            line_num = content[:match.start()].count('\n')
            dst_ptr = match.group(1).strip()
            src_ptr = match.group(2).strip()
            size_expr = match.group(3).strip()
            
            # Get context (5 lines before and after)
            ctx_start = max(0, line_num - 5)
            ctx_end = min(len(lines), line_num + 6)
            context = '\n'.join(lines[ctx_start:ctx_end])
            
            # Determine if forward or backward
            is_forward = 'backward' not in file_path.lower() and 'Backward' not in context
            
            # Try to infer format from variable names or comments
            tensor_format = self._infer_format_from_context(dst_ptr, src_ptr, context)
            
            # Extract cached tensor name
            cached_name = self._extract_cache_name(dst_ptr)
            
            self.cache_ops.append(CacheOperation(
                file=file_path,
                line_num=line_num + 1,
                src_ptr=src_ptr,
                dst_ptr=dst_ptr,
                format=tensor_format,
                size_bytes=size_expr,
                context=context,
                is_forward=is_forward,
                cached_tensor_name=cached_name
            ))
    
    def _find_tensor_usages(self, file_path: str, lines: List[str]) -> None:
        """Find usages of cached tensors"""
        content = '\n'.join(lines)
        
        # Look for common cached tensor patterns
        cached_patterns = [
            r'cached_Q\[',
            r'cached_K\[',
            r'cached_V\[',
            r'cached_attn_bhsd\[',
            r'cached_attn_raw_outputs?\[',
            r'cached_attn_outputs?\[',
            r'cached_ln\d*_outputs?\[',
            r'cached_ffn_[a-z_]+\[',
        ]
        
        for pattern in cached_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n')
                
                # Get context (3 lines before and after)
                ctx_start = max(0, line_num - 3)
                ctx_end = min(len(lines), line_num + 4)
                context = '\n'.join(lines[ctx_start:ctx_end])
                
                # Extract full tensor reference
                line = lines[line_num]
                match_start = match.start() - content[:line_num].rfind('\n') - 1 if line_num > 0 else match.start()
                
                # Find end of tensor reference (until ; or , or ))
                end_idx = match_start
                while end_idx < len(line) and line[end_idx] not in [';', '\n']:
                    end_idx += 1
                tensor_ref = line[match_start:end_idx].strip().rstrip(',').rstrip(')')
                
                # Determine operation type
                operation = self._infer_operation_from_context(context)
                expected_format = self._infer_expected_format(operation, context)
                
                is_backward = 'backward' in file_path.lower() or 'Backward' in context
                
                self.tensor_usages.append(TensorUsage(
                    file=file_path,
                    line_num=line_num + 1,
                    tensor_name=tensor_ref,
                    operation=operation,
                    expected_format=expected_format,
                    context=context,
                    is_backward=is_backward
                ))
    
    def _infer_format_from_context(self, dst_ptr: str, src_ptr: str, context: str) -> str:
        """Infer tensor format from pointer names and context"""
        combined = f"{dst_ptr} {src_ptr} {context}".lower()
        
        if 'bhsd' in combined or 'd_attn_bhsd' in combined:
            return 'BHSD'
        elif 'bshd' in combined:
            return 'BSHD'
        elif 'bsm' in combined or '_workspace' in combined or 'attn_raw' in combined:
            return 'BSM'
        elif 'd_q' in combined or 'd_k' in combined or 'd_v' in combined:
            # Q, K, V are typically in BHSD for Flash Attention
            return 'BHSD'
        
        # Check for size expressions that hint at format
        if 'num_heads' in combined and 'head_dim' in combined:
            if 'batch' in combined:
                return 'BHSD'
        
        return 'UNKNOWN'
    
    def _extract_cache_name(self, dst_ptr: str) -> str:
        """Extract the cache variable name from destination pointer"""
        # Remove common prefixes/access patterns
        name = dst_ptr.strip()
        
        # Handle array indexing: cached_Q[layer] -> cached_Q
        if '[' in name:
            name = name[:name.index('[')]
        
        # Handle member access: training_state_.cached_Q -> cached_Q
        if '.' in name or '->' in name:
            parts = name.replace('->', '.').split('.')
            name = parts[-1]
        
        return name.strip()
    
    def _infer_operation_from_context(self, context: str) -> str:
        """Infer what operation is using the cached tensor"""
        ctx_lower = context.lower()
        
        if 'flashattention' in ctx_lower or 'flash_attention' in ctx_lower:
            return 'Flash_Attention'
        elif 'sgemm' in ctx_lower or 'cublas' in ctx_lower:
            return 'GEMM'
        elif 'converttensor' in ctx_lower:
            return 'convertTensor'
        elif 'reshape' in ctx_lower:
            return 'reshape'
        elif 'quantize' in ctx_lower:
            return 'quantize'
        elif 'gradient' in ctx_lower or 'grad_' in ctx_lower:
            return 'gradient_computation'
        
        return 'UNKNOWN'
    
    def _infer_expected_format(self, operation: str, context: str) -> str:
        """Infer what format an operation expects"""
        if operation == 'Flash_Attention':
            return 'BHSD'  # Flash Attention uses BHSD
        elif operation == 'GEMM':
            # GEMMs typically work with flattened BSM format
            return 'BSM'
        elif operation == 'convertTensor':
            # Extract target format from convertTensor call
            match = re.search(r'TensorFormat::(\w+).*TensorFormat::(\w+)', context)
            if match:
                return match.group(2)  # Destination format
        
        return 'UNKNOWN'
    
    def _analyze_dataflow(self) -> None:
        """Analyze data flow to detect format mismatches and inefficiencies"""
        print(f"\n🔬 Analyzing data flow patterns...")
        
        # Build a map of cached tensors and their formats
        cache_map: Dict[str, List[CacheOperation]] = {}
        for cache_op in self.cache_ops:
            name = cache_op.cached_tensor_name
            if name not in cache_map:
                cache_map[name] = []
            cache_map[name].append(cache_op)
        
        # For each cached tensor, find its usages
        for tensor_name, cache_list in cache_map.items():
            # Find usages that match this tensor
            relevant_usages = [
                u for u in self.tensor_usages 
                if tensor_name in u.tensor_name
            ]
            
            if not relevant_usages:
                continue
            
            # Check for format mismatches
            for cache_op in cache_list:
                for usage in relevant_usages:
                    if cache_op.format != 'UNKNOWN' and usage.expected_format != 'UNKNOWN':
                        if cache_op.format != usage.expected_format:
                            # Format mismatch detected!
                            severity = "ERROR" if cache_op.is_forward and usage.is_backward else "WARNING"
                            
                            self.format_mismatches.append(FormatMismatch(
                                cached_op=cache_op,
                                usage=usage,
                                severity=severity,
                                message=f"Cached in {cache_op.format} but used as {usage.expected_format}",
                                suggested_fix=f"Cache in {usage.expected_format} format or add conversion",
                                redundant_conversion=True
                            ))
        
        # Detect redundant conversions
        self._detect_redundant_conversions()
        
        # Detect suboptimal cache points
        self._detect_suboptimal_caching()
        
        print(f"   Found {len(self.format_mismatches)} format mismatches")
        print(f"   Found {len(self.dataflow_issues)} data flow issues")
    
    def _detect_redundant_conversions(self) -> None:
        """Detect cases where we convert A->B, cache B, then convert B->C in backward"""
        # Look for patterns like:
        # Forward: BHSD -> BSM (reshape) -> cache BSM
        # Backward: load BSM -> BSM to BHSD (convert back)
        
        for cache_op in self.cache_ops:
            if not cache_op.is_forward:
                continue
            
            # Find reshapes that produce this cached format
            relevant_reshapes = [
                r for r in self.reshape_ops
                if r.dst_format == cache_op.format and 
                r.file == cache_op.file and
                abs(r.line_num - cache_op.line_num) < 20
            ]
            
            # Find backward usages that convert this back
            backward_usages = [
                u for u in self.tensor_usages
                if cache_op.cached_tensor_name in u.tensor_name and
                u.is_backward and
                u.expected_format != cache_op.format
            ]
            
            if relevant_reshapes and backward_usages:
                # We have a round-trip conversion!
                orig_format = relevant_reshapes[0].src_format
                intermediate_format = cache_op.format
                final_format = backward_usages[0].expected_format
                
                if orig_format == final_format:
                    # Circular conversion detected
                    self.dataflow_issues.append(DataFlowIssue(
                        issue_type="redundant_conversion",
                        severity="MAJOR",
                        description=f"Tensor converted {orig_format}->{intermediate_format} then cached, "
                                  f"but backward pass converts back {intermediate_format}->{final_format}",
                        files_involved=[cache_op.file],
                        line_numbers=[relevant_reshapes[0].line_num, cache_op.line_num, backward_usages[0].line_num],
                        impact="Wastes GPU memory bandwidth with redundant conversions in hot path",
                        suggested_fix=f"Cache in original {orig_format} format to eliminate backward conversion",
                        evidence=[
                            f"Forward reshape: {relevant_reshapes[0].file}:{relevant_reshapes[0].line_num}",
                            f"Cache operation: {cache_op.file}:{cache_op.line_num}",
                            f"Backward usage: {backward_usages[0].file}:{backward_usages[0].line_num}"
                        ]
                    ))
    
    def _detect_suboptimal_caching(self) -> None:
        """Detect cases where caching happens at a suboptimal point"""
        # Look for attention output caching patterns
        
        # Find attention-related cache operations
        attn_caches = [
            c for c in self.cache_ops
            if 'attn' in c.cached_tensor_name.lower() and
            'output' in c.cached_tensor_name.lower()
        ]
        
        for cache_op in attn_caches:
            # Check if this is caching after a reshape that could be avoided
            recent_reshapes = [
                r for r in self.reshape_ops
                if r.file == cache_op.file and
                r.line_num < cache_op.line_num and
                cache_op.line_num - r.line_num < 15
            ]
            
            if recent_reshapes:
                # Find if there's a backward usage that expects the original format
                backward_usages = [
                    u for u in self.tensor_usages
                    if cache_op.cached_tensor_name in u.tensor_name and
                    u.is_backward
                ]
                
                for reshape in recent_reshapes:
                    for usage in backward_usages:
                        if usage.expected_format == reshape.src_format:
                            # Cache should happen BEFORE the reshape!
                            self.dataflow_issues.append(DataFlowIssue(
                                issue_type="suboptimal_cache_point",
                                severity="MAJOR",
                                description=f"Caching {cache_op.cached_tensor_name} after reshape "
                                          f"({reshape.src_format}->{reshape.dst_format}), "
                                          f"but backward needs {usage.expected_format} format",
                                files_involved=[cache_op.file],
                                line_numbers=[reshape.line_num, cache_op.line_num, usage.line_num],
                                impact="Forces redundant format conversion in backward pass",
                                suggested_fix=f"Move cache operation before reshape to preserve {reshape.src_format} format",
                                evidence=[
                                    f"Reshape: {reshape.file}:{reshape.line_num} ({reshape.src_format}->{reshape.dst_format})",
                                    f"Cache: {cache_op.file}:{cache_op.line_num} (format: {cache_op.format})",
                                    f"Backward usage: {usage.file}:{usage.line_num} (expects: {usage.expected_format})"
                                ]
                            ))
        
        # NEW: Detect temp conversions for single operations (indicates format mismatch)
        self._detect_temp_conversions()
    
    def _detect_temp_conversions(self) -> None:
        """Detect temporary format conversions that indicate architectural issues"""
        # Look for patterns like:
        # 1. cudaMalloc temp buffer
        # 2. convertTensor(cached_X, temp, FORMAT_A, FORMAT_B)
        # 3. Use temp in operation
        # 4. cudaFree temp
        # This indicates the cached format doesn't match the operation's needs
        
        # EXCLUDE patterns with persistent buffer reuse:
        # - Comments mentioning "reuse" or "persistent"
        # - Assignments like "temp = existing_buffer" (no cudaMalloc)
        # - Comments explaining "eliminates cudaMalloc" or "OPTIMIZATION"
        
        for reshape in self.reshape_ops:
            # Check if this reshape involves a cached tensor
            try:
                file_path = self.model_dir / reshape.file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Get context around the reshape (30 lines before and after for better analysis)
                start_line = max(0, reshape.line_num - 30)
                end_line = min(len(lines), reshape.line_num + 30)
                context = ''.join(lines[start_line:end_line])
                
                # Check if this is a persistent buffer reuse pattern
                is_persistent_reuse = any([
                    'reuse' in context.lower() and 'buffer' in context.lower(),
                    'persistent' in context.lower(),
                    'eliminates' in context.lower() and 'cudamalloc' in context.lower(),
                    'optimization' in context.upper(),
                    # Check for assignment without cudaMalloc: "float* temp = existing_buffer;"
                    re.search(r'float\s*\*\s*\w+\s*=\s*\w+(?!.*cudaMalloc)', context, re.IGNORECASE)
                ])
                
                # Look for temp allocation pattern (cudaMalloc + temp variable)
                has_temp_alloc = 'cudaMalloc' in context and ('temp' in context.lower() or '_temp' in context)
                # Exclude if cudaMalloc is only in comments
                if has_temp_alloc:
                    # Check if cudaMalloc appears only in comments
                    context_lines = context.split('\n')
                    actual_malloc = False
                    for line in context_lines:
                        # Skip comment-only lines
                        if 'cudaMalloc' in line:
                            # Check if it's in a comment
                            comment_idx = line.find('//')
                            malloc_idx = line.find('cudaMalloc')
                            if comment_idx == -1 or malloc_idx < comment_idx:
                                # cudaMalloc appears before comment or no comment
                                actual_malloc = True
                                break
                    has_temp_alloc = actual_malloc
                
                if has_temp_alloc and not is_persistent_reuse:
                    # Check if source is a cached tensor
                    cached_sources = [c.cached_tensor_name for c in self.cache_ops]
                    for cached_name in cached_sources:
                        if cached_name in context and 'cached_' in context:
                            # We found a temp conversion of cached data!
                            # This suggests the cache format doesn't match operational needs
                            
                            # Find what operation uses this temp
                            operation = "UNKNOWN"
                            if 'sgemm' in context.lower() or 'cublas' in context.lower():
                                operation = "GEMM (weight gradient)"
                            elif 'flash' in context.lower():
                                operation = "Flash Attention"
                            
                            self.dataflow_issues.append(DataFlowIssue(
                                issue_type="temp_format_conversion",
                                severity="MAJOR",
                                description=f"Temporary buffer allocated to convert cached tensor "
                                          f"from {reshape.src_format} to {reshape.dst_format} for {operation}",
                                files_involved=[reshape.file],
                                line_numbers=[reshape.line_num],
                                impact="Extra memory allocation + conversion in hot path (backward pass). "
                                      "Indicates cached format doesn't match operational requirements.",
                                suggested_fix=f"Consider: (1) Cache in {reshape.dst_format} format if this is the "
                                            f"primary usage, OR (2) Redesign {operation} to work with {reshape.src_format} format, "
                                            f"OR (3) Use persistent buffer instead of temp allocation",
                                evidence=[
                                    f"Conversion: {reshape.file}:{reshape.line_num} ({reshape.src_format}->{reshape.dst_format})",
                                    f"Pattern: temp cudaMalloc + convertTensor + operation",
                                    f"Context shows: {operation}"
                                ]
                            ))
                            break
            except Exception:
                pass  # Skip if file can't be read
    
    def verify_all(self) -> None:
        """Run verification on all found operations"""
        print(f"\n🔬 Verifying {len(self.gemm_calls)} GEMM calls...")
        
        for call in self.gemm_calls:
            result = analyze_gemm_call(call)
            self.gemm_results.append(result)
        
        # Summarize
        valid_gemm = sum(1 for r in self.gemm_results if r.is_valid)
        print(f"   ✅ Valid: {valid_gemm}/{len(self.gemm_results)}")
        
        # Analyze data flow
        self._analyze_dataflow()
    def print_report(self) -> None:
        """Print verification report"""
        print("\n" + "="*80)
        print("GRIM MODEL VERIFICATION REPORT v2")
        print("="*80)
        
        print("\n📋 VERIFICATION METHODOLOGY:")
        print("   • cuBLAS uses COLUMN-MAJOR storage")
        print("   • Leading dimensions are COLUMN strides (not row counts)")
        print("   • op(A)[M,K] @ op(B)[K,N] = C[M,N]")
        print("   • lda >= M if OP_N, lda >= K if OP_T")
        print("   • Category inferred from β value + pointer names")
        
        # GEMM Summary
        print("\n" + "-"*80)
        print("GEMM OPERATIONS (cuBLAS column-major semantics)")
        print("-"*80)
        
        by_category = {}
        for r in self.gemm_results:
            cat = r.op_category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)
        
        for cat, results in sorted(by_category.items()):
            valid = sum(1 for r in results if r.is_valid)
            print(f"\n{cat.upper()}: {valid}/{len(results)} valid")
            
            for r in results:
                status = "✅" if r.is_valid else "❌"
                print(f"   {status} {r.call.file}:{r.call.line_num}")
                print(f"      {r.effective_op}")
                
                if r.issues:
                    for issue in r.issues:
                        print(f"      ❌ {issue}")
                if r.warnings:
                    for warn in r.warnings:
                        print(f"      ⚠️  {warn}")
        
        # Attention Summary
        if self.attention_blocks:
            print("\n" + "-"*80)
            print("ATTENTION MECHANISMS")
            print("-"*80)
            
            for block in self.attention_blocks:
                flash = "⚡ Flash" if block.uses_flash_attention else "📊 Standard"
                status = "✅" if not block.issues else "❌"
                
                print(f"\n{status} {block.file}:{block.line_start} ({flash})")
                print(f"   QKV: {'✓' if block.has_qkv_projection else '✗'}")
                print(f"   Scale: {'✓' if block.has_scale_factor else '✗'} ({block.scale_value or 'not found'})")
                print(f"   Softmax: {'✓' if block.has_softmax else '✗'}")
                print(f"   Causal: {'✓' if block.has_causal_mask else '✗'}")
                print(f"   Dropout: {'✓' if block.has_dropout else '✗'}")
                
                for issue in block.issues:
                    print(f"   ❌ {issue}")
                for warn in block.warnings:
                    print(f"   ⚠️  {warn}")
        
        # Reshape Summary
        if self.reshape_ops:
            print("\n" + "-"*80)
            print("TENSOR RESHAPES (with stride analysis)")
            print("-"*80)
            
            for op in self.reshape_ops:
                bij_status = "✅" if op.is_bijective else "❌"
                contig_status = "📦" if op.is_contiguous_preserving else "⚠️ "
                
                perm = compute_permutation(op.src_format, op.dst_format)
                perm_str = str(perm) if perm else "N/A (not a pure permutation)"
                
                print(f"   {bij_status}{contig_status} {op.file}:{op.line_num}")
                print(f"      {op.src_format} → {op.dst_format}")
                print(f"      Permutation: {perm_str}")
                if not op.is_contiguous_preserving:
                    print(f"      ⚠️  Result is non-contiguous (may need explicit copy)")
                
                for issue in op.issues:
                    print(f"      ❌ {issue}")
        
        # Data Flow Analysis Summary
        if self.format_mismatches or self.dataflow_issues:
            print("\n" + "-"*80)
            print("DATA FLOW ANALYSIS (Format Mismatches & Inefficiencies)")
            print("-"*80)
            
            # Format mismatches
            if self.format_mismatches:
                print(f"\n🔄 FORMAT MISMATCHES: {len(self.format_mismatches)} found")
                
                for mismatch in self.format_mismatches:
                    severity_icon = "🔴" if mismatch.severity == "ERROR" else "🟡"
                    print(f"\n{severity_icon} {mismatch.severity}: {mismatch.message}")
                    print(f"   Cached: {mismatch.cached_op.file}:{mismatch.cached_op.line_num}")
                    print(f"      Variable: {mismatch.cached_op.cached_tensor_name}")
                    print(f"      Format: {mismatch.cached_op.format}")
                    print(f"      Phase: {'Forward' if mismatch.cached_op.is_forward else 'Backward'}")
                    print(f"   Used: {mismatch.usage.file}:{mismatch.usage.line_num}")
                    print(f"      Operation: {mismatch.usage.operation}")
                    print(f"      Expected Format: {mismatch.usage.expected_format}")
                    print(f"      Phase: {'Backward' if mismatch.usage.is_backward else 'Forward'}")
                    if mismatch.redundant_conversion:
                        print(f"   ⚠️  This triggers redundant conversion in hot path!")
                    if mismatch.suggested_fix:
                        print(f"   💡 Fix: {mismatch.suggested_fix}")
            
            # Data flow issues
            if self.dataflow_issues:
                print(f"\n🔍 DATA FLOW ISSUES: {len(self.dataflow_issues)} found")
                
                # Sort by severity
                severity_order = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
                sorted_issues = sorted(self.dataflow_issues, key=lambda x: severity_order.get(x.severity, 99))
                
                for issue in sorted_issues:
                    severity_icon = {"CRITICAL": "🔴", "MAJOR": "🟠", "MINOR": "🟡"}.get(issue.severity, "⚪")
                    print(f"\n{severity_icon} {issue.severity} - {issue.issue_type.upper()}")
                    print(f"   {issue.description}")
                    print(f"   Impact: {issue.impact}")
                    print(f"   💡 Suggested Fix: {issue.suggested_fix}")
                    print(f"   Evidence:")
                    for evidence in issue.evidence:
                        print(f"      • {evidence}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        valid_gemm = sum(1 for r in self.gemm_results if r.is_valid)
        valid_reshapes = sum(1 for r in self.reshape_ops if r.is_bijective)
        critical_issues = sum(1 for i in self.dataflow_issues if i.severity == "CRITICAL")
        major_issues = sum(1 for i in self.dataflow_issues if i.severity == "MAJOR")
        
        print(f"GEMM Operations: {valid_gemm}/{len(self.gemm_results)} valid")
        print(f"Reshapes: {valid_reshapes}/{len(self.reshape_ops)} bijective")
        print(f"Format Mismatches: {len(self.format_mismatches)} detected")
        print(f"Data Flow Issues: {critical_issues} CRITICAL, {major_issues} MAJOR")
        
        if critical_issues > 0 or major_issues > 0:
            print(f"\n⚠️  ATTENTION REQUIRED: Found {critical_issues + major_issues} serious data flow issues")
        elif len(self.format_mismatches) > 0:
            print(f"\n⚠️  Found {len(self.format_mismatches)} format mismatches to review")
        else:
            print(f"\n✅ No critical data flow issues detected")


def main():
    parser = argparse.ArgumentParser(description="GRIM Model Verification Tool v2")
    parser.add_argument(
        "--model-dir",
        default="resources/models/GRIM-text",
        help="Path to model directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed analysis"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    model_dir = script_dir / args.model_dir
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    
    print("="*80)
    print("GRIM Model Verification Tool v2 - Mathematically Rigorous")
    print("="*80)
    
    verifier = GRIMVerifier(str(model_dir))
    verifier.scan_files()
    verifier.verify_all()
    verifier.print_report()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
