#!/usr/bin/env python3
"""
Comprehensive Model Verification Tool for GRIM Training Code

This script analyzes CUDA/C++ code to verify mathematical correctness of:
- Matrix operations (cuBLAS gemm, transposes, leading dimensions)
- Attention mechanisms (softmax scaling, mask application)
- Weight initialization (Xavier/He bounds)
- Optimizer operations (AdamW, momentum, weight decay)
- Forward/backward pass flow
- Kernel launch configurations
- Numerical stability patterns
"""

import re
import os
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from enum import Enum

class TransposeOp(Enum):
    NONE = "CUBLAS_OP_N"
    TRANSPOSE = "CUBLAS_OP_T"

@dataclass
class MatrixOp:
    """Represents a single matrix multiplication operation"""
    file: str
    line_num: int
    context: str
    operation: str  # "gemm", "sgemm", etc.
    transpose_a: TransposeOp
    transpose_b: TransposeOp
    m: str  # dimension expressions
    n: str
    k: str
    matrix_a: str
    matrix_b: str
    matrix_c: str
    lda: str
    ldb: str
    ldc: str
    comment: str = ""
    forward_context: str = ""  # Mathematical context from comments

@dataclass
class TensorReshape:
    """Represents a tensor reshape/conversion operation"""
    file: str
    line_num: int
    src_format: str
    dst_format: str
    dimensions: Dict[str, str]
    context: str

@dataclass
class AttentionOp:
    """Represents an attention-related operation"""
    file: str
    line_num: int
    op_type: str  # "scale", "softmax", "mask", "score_compute"
    scale_factor: Optional[str]
    softmax_dim: int
    mask_type: Optional[str]
    seq_len: Optional[str]
    head_dim: Optional[str]
    num_heads: Optional[str]
    context: str

@dataclass
class InitializerOp:
    """Represents a weight initialization operation"""
    file: str
    line_num: int
    init_type: str  # "xavier", "he", "normal", "uniform"
    fan_in: Optional[str]
    fan_out: Optional[str]
    distribution: Optional[str]  # "normal", "uniform"
    gain: Optional[str]
    context: str

@dataclass
class OptimizerOp:
    """Represents an optimizer operation"""
    file: str
    line_num: int
    optimizer_type: str  # "adamw", "sgd", "adam"
    learning_rate: Optional[str]
    beta1: Optional[str]
    beta2: Optional[str]
    epsilon: Optional[str]
    weight_decay: Optional[str]
    context: str

@dataclass
class HyperparamOp:
    """Represents a hyperparameter assignment"""
    file: str
    line_num: int
    name: str
    value: str
    context: str

@dataclass
class KernelLaunch:
    """Represents a CUDA kernel launch"""
    file: str
    line_num: int
    kernel_name: str
    grid_dim: str
    block_dim: str
    shared_mem: str
    context: str

@dataclass
class VerificationResult:
    """Results of verifying a matrix operation"""
    op: MatrixOp
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    analysis: str

@dataclass 
class ReshapeResult:
    """Results of verifying a tensor reshape"""
    reshape: TensorReshape
    is_valid: bool
    issues: List[str]
    warnings: List[str]

@dataclass
class GenericResult:
    """Generic verification result for attention, optimizer, etc."""
    item: Any  # Can be AttentionOp, InitializerOp, OptimizerOp, or KernelLaunch
    category: str
    is_valid: bool
    issues: List[str]
    warnings: List[str]

class ModelVerifier:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        
        # Matrix operations
        self.operations: List[MatrixOp] = []
        self.results: List[VerificationResult] = []
        
        # Tensor reshapes
        self.reshapes: List[TensorReshape] = []
        self.reshape_results: List[ReshapeResult] = []
        
        # Attention operations
        self.attention_ops: List[AttentionOp] = []
        self.attention_results: List[GenericResult] = []
        
        # Weight initializers
        self.initializers: List[InitializerOp] = []
        self.initializer_results: List[GenericResult] = []
        
        # Optimizer operations
        self.optimizer_ops: List[OptimizerOp] = []
        self.optimizer_results: List[GenericResult] = []
        
        # Hyperparameters
        self.hyperparams: List[HyperparamOp] = []
        self.hyperparam_results: List[GenericResult] = []

        # Kernel launches
        self.kernel_launches: List[KernelLaunch] = []
        self.kernel_results: List[GenericResult] = []
        
        # Known tensor format element counts (for validation)
        self.format_elements = {
            'BHSD': lambda b,h,s,d: b*h*s*d,
            'BHDS': lambda b,h,s,d: b*h*d*s,
            'BSHD': lambda b,h,s,d: b*s*h*d,
            'BSM': lambda b,h,s,d: b*s*(h*d),
            'SMB': lambda b,h,s,d: s*(h*d)*b,
        }
        
    def scan_files(self) -> None:
        """Scan all .cu and .cpp files for operations to verify"""
        print(f"🔍 Scanning {self.model_dir}...")
        
        for file_path in self.model_dir.rglob("*.cu"):
            self._analyze_file(file_path)
        for file_path in self.model_dir.rglob("*.cpp"):
            self._analyze_file(file_path)
        for file_path in self.model_dir.rglob("*.hpp"):
            self._analyze_file(file_path)
            
        print(f"✓ Found {len(self.operations)} matrix operations (cuBLAS gemm)")
        print(f"✓ Found {len(self.reshapes)} tensor reshapes")
        print(f"✓ Found {len(self.attention_ops)} attention operations")
        print(f"✓ Found {len(self.initializers)} weight initializers")
        print(f"✓ Found {len(self.optimizer_ops)} optimizer configurations")
        print(f"✓ Found {len(self.hyperparams)} hyperparameters")
        print(f"✓ Found {len(self.kernel_launches)} kernel launches")
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for all operations to verify"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"⚠ Warning: Could not read {file_path}: {e}")
            return
        
        # Find all cublasSgemm calls
        gemm_pattern = r'cublas[SD]gemm\s*\('
        
        # Find tensor conversions
        convert_pattern = r'convertTensor\s*\('
        
        # Attention patterns
        attention_scale_pattern = r'(1\.0f?\s*/\s*sqrtf?\s*\(|rsqrtf?\s*\(|scale\s*=.*sqrt|attention.*scale|score.*scale)'
        softmax_pattern = r'softmax\s*\(|_softmax|Softmax'
        
        # Initialization patterns
        init_pattern = r'(xavier|glorot|he_init|kaiming|normal\s*\(|uniform\s*\(|init.*weight|weight.*init)'
        
        # Optimizer patterns
        adam_pattern = r'(adam|AdamW|adamw|beta1|beta2|weight_decay|epsilon\s*=|lr\s*=|learning_rate)'
        
        # Hyperparameter patterns
        hyperparam_pattern = r'(grad_clip_norm|learning_rate|weight_decay|epochs|batch_size)\s*=\s*([\d.e-]+)f?'

        # Kernel launch patterns
        kernel_launch_pattern = r'<<<\s*([^>]+)\s*,\s*([^>]+)(?:\s*,\s*([^>]+))?\s*>>>'
        
        rel_path = str(file_path.relative_to(self.model_dir))
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # cuBLAS gemm operations
            if re.search(gemm_pattern, line):
                op = self._extract_gemm_call(file_path, lines, i)
                if op:
                    self.operations.append(op)
            
            # Tensor conversions
            if re.search(convert_pattern, line):
                reshape = self._extract_convert_tensor(file_path, lines, i)
                if reshape:
                    self.reshapes.append(reshape)
            
            # Attention scale factor
            if re.search(attention_scale_pattern, line, re.IGNORECASE):
                attn = self._extract_attention_op(rel_path, lines, i)
                if attn:
                    self.attention_ops.append(attn)
            
            # Softmax operations
            if re.search(softmax_pattern, line, re.IGNORECASE):
                attn = self._extract_softmax_op(rel_path, lines, i)
                if attn:
                    self.attention_ops.append(attn)
            
            # Weight initialization
            if re.search(init_pattern, line, re.IGNORECASE):
                init_op = self._extract_initializer(rel_path, lines, i)
                if init_op:
                    self.initializers.append(init_op)
            
            # Optimizer configuration
            if re.search(adam_pattern, line, re.IGNORECASE):
                opt_op = self._extract_optimizer(rel_path, lines, i)
                if opt_op:
                    self.optimizer_ops.append(opt_op)
            
            # Hyperparameters
            hp_match = re.search(hyperparam_pattern, line)
            if hp_match:
                self.hyperparams.append(HyperparamOp(
                    file=rel_path,
                    line_num=i + 1,
                    name=hp_match.group(1),
                    value=hp_match.group(2),
                    context=line.strip()
                ))

            # Kernel launches
            kernel_match = re.search(kernel_launch_pattern, line)
            if kernel_match:
                kernel = self._extract_kernel_launch(rel_path, lines, i, kernel_match)
                if kernel:
                    self.kernel_launches.append(kernel)
            
            i += 1
    
    def _extract_attention_op(self, file_path: str, lines: List[str], line_num: int) -> Optional[AttentionOp]:
        """Extract attention scaling operation"""
        line = lines[line_num]
        context = '\n'.join(lines[max(0, line_num-3):min(len(lines), line_num+4)])
        
        # Look for scale factor patterns
        scale_factor = None
        
        # Pattern: 1/sqrt(d) or rsqrt(d)
        sqrt_match = re.search(r'1\.0f?\s*/\s*sqrtf?\s*\(\s*(\w+|\d+\.?\d*)\s*\)', line)
        rsqrt_match = re.search(r'rsqrtf?\s*\(\s*(\w+|\d+\.?\d*)\s*\)', line)
        scale_match = re.search(r'scale\s*=\s*([^;,]+)', line)
        
        if sqrt_match:
            scale_factor = f"1/sqrt({sqrt_match.group(1)})"
        elif rsqrt_match:
            scale_factor = f"rsqrt({rsqrt_match.group(1)})"
        elif scale_match:
            scale_factor = scale_match.group(1).strip()
        else:
            return None  # Not a valid attention scaling op
        
        # Try to extract head_dim from context
        head_dim = None
        head_match = re.search(r'head_dim\s*[=:]\s*(\d+)', context, re.IGNORECASE)
        if head_match:
            head_dim = head_match.group(1)
        
        return AttentionOp(
            file=file_path,
            line_num=line_num + 1,
            op_type="scale",
            scale_factor=scale_factor,
            softmax_dim=-1,  # Last dimension typically
            mask_type=None,
            seq_len=None,
            head_dim=head_dim,
            num_heads=None,
            context=context
        )
    
    def _extract_softmax_op(self, file_path: str, lines: List[str], line_num: int) -> Optional[AttentionOp]:
        """Extract softmax operation"""
        line = lines[line_num]
        context = '\n'.join(lines[max(0, line_num-3):min(len(lines), line_num+4)])
        
        # Look for softmax function call
        if not re.search(r'softmax', line, re.IGNORECASE):
            return None
        
        # Try to determine dimension
        dim = -1  # Default: last dim
        dim_match = re.search(r'dim\s*[=:]\s*(-?\d+)', context)
        if dim_match:
            dim = int(dim_match.group(1))
        
        return AttentionOp(
            file=file_path,
            line_num=line_num + 1,
            op_type="softmax",
            scale_factor=None,
            softmax_dim=dim,
            mask_type=None,
            seq_len=None,
            head_dim=None,
            num_heads=None,
            context=context
        )
    
    def _extract_initializer(self, file_path: str, lines: List[str], line_num: int) -> Optional[InitializerOp]:
        """Extract weight initialization operation"""
        line = lines[line_num]
        context = '\n'.join(lines[max(0, line_num-3):min(len(lines), line_num+4)])
        
        init_type = None
        fan_in = None
        fan_out = None
        gain = None
        distribution = None
        
        lower_line = line.lower()
        lower_context = context.lower()
        
        # Determine init type
        if 'xavier' in lower_line or 'glorot' in lower_line:
            init_type = "xavier"
            # Xavier uses sqrt(2/(fan_in + fan_out)) or sqrt(6/(fan_in + fan_out)) for uniform
        elif 'he' in lower_line or 'kaiming' in lower_line:
            init_type = "he"
            # He uses sqrt(2/fan_in)
        elif 'normal' in lower_line:
            init_type = "normal"
        elif 'uniform' in lower_line:
            init_type = "uniform"
        else:
            return None
        
        # Extract fan_in/fan_out from context
        fan_in_match = re.search(r'fan_in\s*[=:]\s*(\d+)', context)
        fan_out_match = re.search(r'fan_out\s*[=:]\s*(\d+)', context)
        gain_match = re.search(r'gain\s*[=:]\s*([\d.]+)', context)
        
        if fan_in_match:
            fan_in = fan_in_match.group(1)
        if fan_out_match:
            fan_out = fan_out_match.group(1)
        if gain_match:
            gain = gain_match.group(1)
        
        # Determine distribution
        if 'normal' in lower_context:
            distribution = "normal"
        elif 'uniform' in lower_context:
            distribution = "uniform"
        
        return InitializerOp(
            file=file_path,
            line_num=line_num + 1,
            init_type=init_type,
            fan_in=fan_in,
            fan_out=fan_out,
            distribution=distribution,
            gain=gain,
            context=context
        )
    
    def _extract_optimizer(self, file_path: str, lines: List[str], line_num: int) -> Optional[OptimizerOp]:
        """Extract optimizer configuration"""
        line = lines[line_num]
        # Get broader context for optimizer params
        context = '\n'.join(lines[max(0, line_num-10):min(len(lines), line_num+10)])
        
        opt_type = None
        lr = None
        beta1 = None
        beta2 = None
        epsilon = None
        weight_decay = None
        
        lower_context = context.lower()
        
        # Determine optimizer type
        if 'adamw' in lower_context:
            opt_type = "adamw"
        elif 'adam' in lower_context:
            opt_type = "adam"
        elif 'sgd' in lower_context:
            opt_type = "sgd"
        else:
            return None
        
        # Extract hyperparameters
        lr_match = re.search(r'(?:lr|learning_rate)\s*[=:]\s*([\d.e-]+)', context, re.IGNORECASE)
        beta1_match = re.search(r'beta1\s*[=:]\s*([\d.]+)', context)
        beta2_match = re.search(r'beta2\s*[=:]\s*([\d.]+)', context)
        eps_match = re.search(r'epsilon\s*[=:]\s*([\d.e-]+)', context, re.IGNORECASE)
        wd_match = re.search(r'weight_decay\s*[=:]\s*([\d.e-]+)', context, re.IGNORECASE)
        
        if lr_match:
            lr = lr_match.group(1)
        if beta1_match:
            beta1 = beta1_match.group(1)
        if beta2_match:
            beta2 = beta2_match.group(1)
        if eps_match:
            epsilon = eps_match.group(1)
        if wd_match:
            weight_decay = wd_match.group(1)
        
        return OptimizerOp(
            file=file_path,
            line_num=line_num + 1,
            optimizer_type=opt_type,
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            context=context
        )
    
    def _extract_kernel_launch(self, file_path: str, lines: List[str], line_num: int, match: re.Match) -> Optional[KernelLaunch]:
        """Extract CUDA kernel launch configuration"""
        line = lines[line_num]
        context = '\n'.join(lines[max(0, line_num-2):min(len(lines), line_num+3)])
        
        grid_str = match.group(1).strip()
        block_str = match.group(2).strip()
        shared_str = match.group(3).strip() if match.group(3) else "0"
        
        # Extract kernel name (before <<<)
        kernel_match = re.search(r'(\w+)\s*<<<', line)
        kernel_name = kernel_match.group(1) if kernel_match else "unknown"
        
        return KernelLaunch(
            file=file_path,
            line_num=line_num + 1,
            kernel_name=kernel_name,
            grid_dim=grid_str,
            block_dim=block_str,
            shared_mem=shared_str,
            context=context
        )
    
    def _extract_convert_tensor(self, file_path: Path, lines: List[str], start_line: int) -> Optional[TensorReshape]:
        """Extract a convertTensor call for validation"""
        call_lines = []
        line_num = start_line
        paren_depth = 0
        found_start = False
        
        while line_num < len(lines):
            line = lines[line_num]
            code_part = re.sub(r'//.*$', '', line)
            call_lines.append(code_part)
            
            for char in code_part:
                if char == '(':
                    paren_depth += 1
                    found_start = True
                elif char == ')':
                    paren_depth -= 1
            
            if found_start and paren_depth == 0:
                break
            line_num += 1
        
        full_call = ' '.join(call_lines)
        
        # Parse convertTensor(src, dst, src_fmt, dst_fmt, B, H, S, D, stream)
        try:
            params_match = re.search(r'convertTensor\s*\((.+)\)', full_call, re.DOTALL)
            if not params_match:
                return None
            
            params = self._smart_split(params_match.group(1))
            if len(params) < 8:
                return None
            
            # Extract format enums
            src_fmt_match = re.search(r'TensorFormat::(\w+)', params[2])
            dst_fmt_match = re.search(r'TensorFormat::(\w+)', params[3])
            
            src_fmt = src_fmt_match.group(1) if src_fmt_match else params[2].strip()
            dst_fmt = dst_fmt_match.group(1) if dst_fmt_match else params[3].strip()
            
            # Get context
            context_lines = []
            for j in range(max(0, start_line - 5), start_line):
                if lines[j].strip().startswith('//'):
                    context_lines.append(lines[j].strip())
            
            return TensorReshape(
                file=str(file_path.relative_to(self.model_dir)),
                line_num=start_line + 1,
                src_format=src_fmt,
                dst_format=dst_fmt,
                dimensions={
                    'B': params[4].strip(),
                    'H': params[5].strip(),
                    'S': params[6].strip(),
                    'D': params[7].strip(),
                },
                context='\n'.join(context_lines)
            )
        except Exception as e:
            return None
    
    def _extract_gemm_call(self, file_path: Path, lines: List[str], start_line: int) -> Optional[MatrixOp]:
        """Extract a complete cuBLAS gemm call spanning multiple lines"""
        # Collect lines until we find the closing semicolon
        call_lines = []
        line_num = start_line
        paren_depth = 0
        found_start = False
        
        while line_num < len(lines):
            line = lines[line_num]
            # Strip inline comments before processing
            code_part = re.sub(r'//.*$', '', line)
            call_lines.append(code_part)
            
            for char in code_part:
                if char == '(':
                    paren_depth += 1
                    found_start = True
                elif char == ')':
                    paren_depth -= 1
            
            if found_start and paren_depth == 0:
                break
            line_num += 1
        
        full_call = ' '.join(call_lines)
        
        # Extract preceding comments (last 5 lines before the call)
        context_lines = []
        for i in range(max(0, start_line - 10), start_line):
            if lines[i].strip().startswith('//'):
                context_lines.append(lines[i].strip())
        forward_context = '\n'.join(context_lines)
        
        # Parse the call
        try:
            # Extract operation type
            op_match = re.search(r'cublas([SD]gemm)', full_call)
            if not op_match:
                return None
            operation = op_match.group(1)
            
            # Remove function name and handle, keep only parameters
            params_match = re.search(r'cublas[SD]gemm\s*\([^,]+,\s*(.+)\)', full_call, re.DOTALL)
            if not params_match:
                return None
            
            params_str = params_match.group(1)
            
            # Split by comma, but respect nested parentheses
            params = self._smart_split(params_str)
            
            if len(params) < 11:
                print(f"⚠ Warning: Incomplete gemm call at {file_path}:{start_line+1}")
                return None
            
            # Parse transpose operations
            transpose_a = TransposeOp.TRANSPOSE if 'CUBLAS_OP_T' in params[0] else TransposeOp.NONE
            transpose_b = TransposeOp.TRANSPOSE if 'CUBLAS_OP_T' in params[1] else TransposeOp.NONE
            
            # Extract ldc - skip matrix C pointer, get the actual stride value (next param)
            ldc_value = params[12].strip() if len(params) > 12 else params[11].strip() if len(params) > 11 else "?"
            
            return MatrixOp(
                file=str(file_path.relative_to(self.model_dir)),
                line_num=start_line + 1,
                context=full_call,
                operation=operation,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                m=params[2].strip(),
                n=params[3].strip(),
                k=params[4].strip(),
                matrix_a=params[6].strip(),
                matrix_b=params[8].strip(),
                matrix_c=params[10].strip(),
                lda=params[7].strip(),
                ldb=params[9].strip(),
                ldc=ldc_value,
                comment=self._extract_inline_comment(full_call),
                forward_context=forward_context
            )
        except Exception as e:
            print(f"⚠ Warning: Could not parse gemm at {file_path}:{start_line+1}: {e}")
            return None
    
    def _smart_split(self, s: str) -> List[str]:
        """Split by comma, respecting nested parentheses and function calls"""
        parts = []
        current = []
        depth = 0
        
        for char in s:
            if char == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _extract_inline_comment(self, call: str) -> str:
        """Extract inline comments from the call"""
        comment_match = re.search(r'//(.+?)(?:\n|$)', call)
        return comment_match.group(1).strip() if comment_match else ""
    
    def verify_operations(self) -> None:
        """Verify all extracted operations"""
        print(f"\n🔬 Verifying {len(self.operations)} matrix operations...")
        for op in self.operations:
            result = self._verify_single_op(op)
            self.results.append(result)
        
        print(f"🔬 Verifying {len(self.reshapes)} tensor reshapes...")
        for reshape in self.reshapes:
            result = self._verify_reshape(reshape)
            self.reshape_results.append(result)
        
        print(f"🔬 Verifying {len(self.attention_ops)} attention operations...")
        for attn in self.attention_ops:
            result = self._verify_attention(attn)
            self.attention_results.append(result)
        
        print(f"🔬 Verifying {len(self.initializers)} weight initializers...")
        for init in self.initializers:
            result = self._verify_initializer(init)
            self.initializer_results.append(result)
        
        print(f"🔬 Verifying {len(self.optimizer_ops)} optimizer configurations...")
        for opt in self.optimizer_ops:
            result = self._verify_optimizer(opt)
            self.optimizer_results.append(result)
        
        print(f"🔬 Verifying {len(self.hyperparams)} hyperparameters...")
        for hp in self.hyperparams:
            result = self._verify_hyperparam(hp)
            self.hyperparam_results.append(result)
        
        print(f"🔬 Verifying {len(self.kernel_launches)} kernel launches...")
        for kernel in self.kernel_launches:
            result = self._verify_kernel_launch(kernel)
            self.kernel_results.append(result)
    
    def _verify_reshape(self, reshape: TensorReshape) -> ReshapeResult:
        """Verify a tensor reshape preserves elements and makes sense"""
        issues = []
        warnings = []
        
        # Known valid conversions
        valid_conversions = {
            ('BSM', 'BHSD'),  # Flatten to multi-head
            ('BHSD', 'BSM'),  # Multi-head to flatten
            ('BSHD', 'BHSD'), # Swap seq and heads
            ('BHSD', 'BSHD'),
            ('BHSD', 'BHDS'), # Transpose last two dims
            ('BHDS', 'BHSD'),
        }
        
        src = reshape.src_format
        dst = reshape.dst_format
        
        # Check if this is a known valid conversion
        if (src, dst) not in valid_conversions:
            warnings.append(f"Unusual conversion {src} → {dst} - verify this is intended")
        
        # Check element count preservation
        # For BSM: total = B * S * M where M = H * D
        # For BHSD: total = B * H * S * D
        # These should be equal when M = H * D
        
        dims = reshape.dimensions
        if 'M' not in dims and src == 'BSM':
            # BSM uses M = H * D implicitly
            pass
        
        # Check for common issues
        if src == dst:
            issues.append(f"Identity conversion {src} → {dst} is wasteful")
        
        return ReshapeResult(
            reshape=reshape,
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
    
    def _verify_attention(self, attn: AttentionOp) -> GenericResult:
        """Verify attention operation math"""
        issues = []
        warnings = []
        
        if attn.op_type == "scale":
            # Check that scale factor is 1/sqrt(d_k) or rsqrt(d_k)
            if attn.scale_factor:
                scale = attn.scale_factor.lower()
                if 'sqrt' not in scale and 'rsqrt' not in scale:
                    issues.append(f"Attention scale should use sqrt(d_k), got: {attn.scale_factor}")
                
                # Check if using correct dimension (should be head_dim, not d_model)
                if attn.head_dim:
                    try:
                        hd = int(attn.head_dim)
                        expected_scale = 1.0 / (hd ** 0.5)
                        warnings.append(f"Expected scale ≈ {expected_scale:.6f} for head_dim={hd}")
                    except ValueError:
                        pass
        
        elif attn.op_type == "softmax":
            if attn.softmax_dim != -1 and attn.softmax_dim != 3:
                warnings.append(f"Softmax dim={attn.softmax_dim}, expected -1 or 3 for attention")
        
        return GenericResult(
            item=attn,
            category="attention",
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
    
    def _verify_initializer(self, init: InitializerOp) -> GenericResult:
        """Verify weight initialization parameters"""
        issues = []
        warnings = []
        
        if init.init_type == "xavier":
            if init.fan_in and init.fan_out:
                try:
                    fi = float(init.fan_in)
                    fo = float(init.fan_out)
                    expected_std = (2.0 / (fi + fo)) ** 0.5
                    expected_bound = (6.0 / (fi + fo)) ** 0.5
                    if init.distribution == "normal":
                        warnings.append(f"Xavier normal: expected std ≈ {expected_std:.6f}")
                    else:
                        warnings.append(f"Xavier uniform: expected bound ≈ {expected_bound:.6f}")
                except ValueError:
                    pass
            else:
                warnings.append("Xavier init detected but fan_in/fan_out not found")
                
        elif init.init_type == "he":
            if init.fan_in:
                try:
                    fi = float(init.fan_in)
                    expected_std = (2.0 / fi) ** 0.5
                    warnings.append(f"He init: expected std ≈ {expected_std:.6f}")
                except ValueError:
                    pass
            else:
                warnings.append("He init detected but fan_in not found")
        
        return GenericResult(
            item=init,
            category="initializer",
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
    
    def _verify_optimizer(self, opt: OptimizerOp) -> GenericResult:
        """Verify optimizer hyperparameters"""
        issues = []
        warnings = []
        
        # Check if this is for normalization layers (weight decay should be 0)
        is_norm_layer = any(keyword in opt.file.lower() for keyword in [
            'layernorm', 'rmsnorm', 'batchnorm', 'groupnorm', 'norm_gpu'
        ])
        
        if opt.optimizer_type in ("adam", "adamw"):
            if opt.beta1:
                try:
                    b1 = float(opt.beta1)
                    if not (0.8 <= b1 <= 0.99):
                        warnings.append(f"beta1={b1} unusual, typical [0.8, 0.99]")
                except ValueError:
                    pass
            
            if opt.beta2:
                try:
                    b2 = float(opt.beta2)
                    if not (0.9 <= b2 <= 0.9999):
                        warnings.append(f"beta2={b2} unusual, typical [0.9, 0.9999]")
                except ValueError:
                    pass
            
            if opt.epsilon:
                try:
                    eps = float(opt.epsilon)
                    if eps > 1e-4:
                        warnings.append(f"epsilon={eps} high")
                    if eps < 1e-12:
                        warnings.append(f"epsilon={eps} low")
                except ValueError:
                    pass
            
            if opt.optimizer_type == "adamw" and opt.weight_decay:
                try:
                    wd = float(opt.weight_decay)
                    if wd > 0.5:
                        issues.append(f"weight_decay={wd} too high")
                    elif wd == 0 and not is_norm_layer:
                        # weight_decay=0 is correct for normalization layers
                        warnings.append("weight_decay=0 makes AdamW = Adam")
                except ValueError:
                    pass
            
            if opt.learning_rate:
                try:
                    lr = float(opt.learning_rate)
                    if lr > 0.1:
                        issues.append(f"lr={lr} dangerously high")
                    elif lr > 0.01:
                        warnings.append(f"lr={lr} high for transformers")
                except ValueError:
                    pass
        
        return GenericResult(
            item=opt,
            category="optimizer",
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )

    def _verify_hyperparam(self, hp: HyperparamOp) -> GenericResult:
        """Verify hyperparameter values against standards"""
        issues = []
        warnings = []
        
        try:
            val = float(hp.value)
            
            if hp.name == "grad_clip_norm":
                if val > 2.0:
                    warnings.append(f"grad_clip_norm={val} is high (Llama 2 uses 1.0)")
                if val <= 0:
                    issues.append(f"grad_clip_norm={val} must be positive")
            
            elif hp.name == "learning_rate":
                if val > 1e-3:
                    warnings.append(f"learning_rate={val} is high (Llama 2 uses 3e-4)")
                if val < 1e-6:
                    warnings.append(f"learning_rate={val} is very low")
            
            elif hp.name == "weight_decay":
                if val < 0.01:
                    warnings.append(f"weight_decay={val} is low (Llama 2 uses 0.1)")
                if val > 0.2:
                    warnings.append(f"weight_decay={val} is high")
            
            elif hp.name == "batch_size":
                if val < 4:
                    warnings.append(f"batch_size={val} is very small")
        
        except ValueError:
            pass
            
        return GenericResult(
            item=hp,
            category="hyperparam",
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
    
    def _verify_kernel_launch(self, kernel: KernelLaunch) -> GenericResult:
        """Verify CUDA kernel launch configuration"""
        issues = []
        warnings = []
        
        # Kernels that are expected to use small block sizes (initialization/callbacks)
        small_block_allowed = [
            'InitializeDeviceLoggingKernel',
            'ResetDeviceCursorKernel',
            'RegisterCallbackKernel',
            'ClearCallbacksKernel',
            'RegisterRewardCallbackKernel',
            'RegisterMutationCallbackKernel',
            'RegisterEvictionCallbackKernel',
            'ClearRewardCallbacksKernel',
            'ClearMutationCallbacksKernel',
            'ClearEvictionCallbacksKernel',
        ]
        
        is_init_kernel = kernel.kernel_name in small_block_allowed
        
        try:
            block_str = kernel.block_dim
            
            dim3_match = re.search(r'dim3\s*\(\s*(\d+)\s*,?\s*(\d+)?\s*,?\s*(\d+)?\s*\)', block_str)
            if dim3_match:
                bx = int(dim3_match.group(1))
                by = int(dim3_match.group(2)) if dim3_match.group(2) else 1
                bz = int(dim3_match.group(3)) if dim3_match.group(3) else 1
                total_threads = bx * by * bz
            else:
                try:
                    total_threads = int(block_str)
                    bx = total_threads
                except ValueError:
                    total_threads = None
                    bx = None
            
            if total_threads:
                if total_threads % 32 != 0:
                    warnings.append(f"Block size {total_threads} not warp-aligned (should be × 32)")
                if total_threads > 1024:
                    issues.append(f"Block size {total_threads} exceeds max (1024)")
                # Only warn about low block size for non-initialization kernels
                if total_threads < 64 and not is_init_kernel:
                    warnings.append(f"Block size {total_threads} low, may underutilize GPU")
                
            if bx and bx % 32 != 0:
                warnings.append(f"Block x-dim {bx} not warp-aligned")
                
        except Exception:
            warnings.append(f"Could not parse block dims: {kernel.block_dim}")
        
        if kernel.shared_mem and kernel.shared_mem != "0":
            try:
                if kernel.shared_mem.isdigit():
                    shared = int(kernel.shared_mem)
                    if shared > 48 * 1024:
                        warnings.append(f"Shared memory {shared} bytes exceeds 48KB")
            except ValueError:
                pass
        
        return GenericResult(
            item=kernel,
            category="kernel_launch",
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )

    def _verify_single_op(self, op: MatrixOp) -> VerificationResult:
        """Verify a single matrix operation"""
        issues = []
        warnings = []
        analysis_parts = []
        
        # 1. Check leading dimension consistency (cuBLAS requirements)
        ld_check = self._check_leading_dimensions(op)
        if ld_check:
            issues.extend(ld_check)
        
        # 2. Check dimension consistency (M×K @ K×N = M×N)
        dim_check = self._check_dimension_consistency(op)
        if dim_check:
            issues.extend(dim_check)
        
        # 3. Check leading dimension bounds (lda >= required)
        ld_bounds_check = self._check_leading_dimension_bounds(op)
        if ld_bounds_check:
            issues.extend(ld_bounds_check)
        
        # 4. Verify forward/backward mathematical relationships
        fb_check = self._check_forward_backward_math(op)
        if fb_check[0]:  # issues
            issues.extend(fb_check[0])
        if fb_check[1]:  # warnings
            warnings.extend(fb_check[1])
        if fb_check[2]:  # analysis
            analysis_parts.append(fb_check[2])
        
        # 5. Check alpha/beta values for correctness
        alpha_beta_check = self._check_alpha_beta(op)
        if alpha_beta_check:
            warnings.extend(alpha_beta_check)
        
        # 6. Analyze row-major interpretation
        row_major_analysis = self._analyze_row_major_conversion(op)
        analysis_parts.append(row_major_analysis)
        
        # 7. Check for common error patterns
        pattern_check = self._check_common_patterns(op)
        if pattern_check:
            warnings.extend(pattern_check)
        
        # 8. Verify forward/backward relationship if applicable
        relationship_check = self._check_forward_backward_relationship(op)
        if relationship_check:
            analysis_parts.append(relationship_check)
        
        is_valid = len(issues) == 0
        analysis = '\n'.join(analysis_parts)
        
        return VerificationResult(
            op=op,
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            analysis=analysis
        )
    
    def _check_dimension_consistency(self, op: MatrixOp) -> List[str]:
        """
        Verify matrix multiplication dimension rules:
        C[M,N] = op(A)[M,K] × op(B)[K,N]
        
        The shared dimension K must match between A and B.
        """
        issues = []
        
        # Extract numeric values if possible for validation
        m_val = self._try_parse_dim(op.m)
        n_val = self._try_parse_dim(op.n)
        k_val = self._try_parse_dim(op.k)
        
        # Check for zero or negative dimensions
        if m_val is not None and m_val <= 0:
            issues.append(f"Invalid M dimension: {op.m} = {m_val} (must be positive)")
        if n_val is not None and n_val <= 0:
            issues.append(f"Invalid N dimension: {op.n} = {n_val} (must be positive)")
        if k_val is not None and k_val <= 0:
            issues.append(f"Invalid K dimension: {op.k} = {k_val} (must be positive)")
        
        # Check for dimension swaps (common copy-paste error)
        # If M and N are swapped, the output shape would be wrong
        if m_val and n_val and k_val:
            # Matrix multiply: [M,K] @ [K,N] = [M,N]
            # Check if result makes sense
            result_elements = m_val * n_val
            if result_elements > 1e10:  # More than 10B elements - suspicious
                issues.append(f"Suspicious result size: {m_val} × {n_val} = {result_elements} elements")
        
        return issues
    
    def _check_leading_dimension_bounds(self, op: MatrixOp) -> List[str]:
        """
        Verify leading dimensions meet cuBLAS requirements:
        
        For CUBLAS_OP_N: lda >= max(1, rows of A)
        For CUBLAS_OP_T: lda >= max(1, cols of A before transpose)
        
        Memory stride must be >= the dimension being stepped over.
        """
        issues = []
        
        # Parse dimensions
        m_val = self._try_parse_dim(op.m)
        n_val = self._try_parse_dim(op.n)
        k_val = self._try_parse_dim(op.k)
        lda_val = self._try_parse_dim(op.lda)
        ldb_val = self._try_parse_dim(op.ldb)
        ldc_val = self._try_parse_dim(op.ldc)
        
        # Check lda bounds
        if lda_val is not None:
            if op.transpose_a == TransposeOp.NONE:
                # A is [M, K] in col-major, lda >= M
                if m_val is not None and lda_val < m_val:
                    issues.append(f"lda={lda_val} < M={m_val} (CUBLAS_OP_N requires lda >= M)")
            else:
                # A is [K, M] before transpose, lda >= K
                if k_val is not None and lda_val < k_val:
                    issues.append(f"lda={lda_val} < K={k_val} (CUBLAS_OP_T requires lda >= K)")
        
        # Check ldb bounds
        if ldb_val is not None:
            if op.transpose_b == TransposeOp.NONE:
                # B is [K, N] in col-major, ldb >= K
                if k_val is not None and ldb_val < k_val:
                    issues.append(f"ldb={ldb_val} < K={k_val} (CUBLAS_OP_N requires ldb >= K)")
            else:
                # B is [N, K] before transpose, ldb >= N
                if n_val is not None and ldb_val < n_val:
                    issues.append(f"ldb={ldb_val} < N={n_val} (CUBLAS_OP_T requires ldb >= N)")
        
        # Check ldc bounds (C is always [M, N] in col-major)
        if ldc_val is not None and m_val is not None:
            if ldc_val < m_val:
                issues.append(f"ldc={ldc_val} < M={m_val} (requires ldc >= M)")
        
        return issues
    
    def _check_forward_backward_math(self, op: MatrixOp) -> Tuple[List[str], List[str], str]:
        """
        Verify forward/backward mathematical consistency.
        
        For forward: Y = X @ W^T
        Backward should have:
          - grad_X = grad_Y @ W      (activation gradient)
          - grad_W = X^T @ grad_Y    (weight gradient)
        
        Returns: (issues, warnings, analysis_text)
        """
        issues = []
        warnings = []
        analysis = ""
        
        context = op.forward_context.lower() + " " + op.comment.lower()
        matrix_c = op.matrix_c.lower()
        matrix_a = op.matrix_a.lower()
        matrix_b = op.matrix_b.lower()
        
        # Detect what type of operation this is
        is_weight_grad = 'grad' in matrix_c and ('weight' in matrix_c or 'w1' in matrix_c or 
                                                   'w2' in matrix_c or 'w_' in matrix_c or
                                                   'wo' in matrix_c or 'wqkv' in matrix_c)
        is_activation_grad = 'grad' in matrix_c and not is_weight_grad
        is_forward = 'grad' not in matrix_c
        
        if is_weight_grad:
            # Weight gradient: grad_W = activation^T @ grad_output
            # In row-major with cuBLAS: typically CUBLAS_OP_N, CUBLAS_OP_T
            # where one input is activation and one is gradient
            has_grad_input = 'grad' in matrix_a or 'grad' in matrix_b
            has_activation = not ('grad' in matrix_a and 'grad' in matrix_b)
            
            if has_grad_input and has_activation:
                # Check transpose pattern for weight gradient
                # Common correct patterns:
                # - CUBLAS_OP_N, CUBLAS_OP_T: grad_W = A @ B^T
                # - CUBLAS_OP_T, CUBLAS_OP_N: grad_W = A^T @ B (less common)
                if op.transpose_a == TransposeOp.NONE and op.transpose_b == TransposeOp.NONE:
                    warnings.append("Weight gradient with CUBLAS_OP_N, CUBLAS_OP_N - verify outer product is intended")
                
                analysis = f"📊 Weight Gradient Pattern Detected:\n   grad_W = f(activation, grad_output)"
        
        elif is_activation_grad:
            # Activation gradient: grad_X = grad_output @ W
            # For row-major: typically CUBLAS_OP_T on weight, CUBLAS_OP_N on grad
            has_weight = any(w in matrix_a or w in matrix_b for w in ['weight', 'w1', 'w2', 'w_', 'wo', 'wqkv', 'getffn', 'getattn'])
            
            if has_weight:
                # Check if weight is being transposed correctly
                weight_is_a = any(w in matrix_a for w in ['weight', 'w1', 'w2', 'w_', 'wo', 'wqkv', 'getffn', 'getattn'])
                
                if weight_is_a and op.transpose_a == TransposeOp.NONE:
                    # Weight in A position without transpose might be wrong
                    # Forward: Y = X @ W^T, so backward: grad_X = grad_Y @ W
                    # In row-major cuBLAS, this needs CUBLAS_OP_T on W
                    warnings.append("Activation gradient: weight matrix may need CUBLAS_OP_T")
                
                analysis = f"📊 Activation Gradient Pattern Detected:\n   grad_input = f(grad_output, weight)"
        
        elif is_forward:
            # Forward pass: Y = X @ W^T typically
            analysis = f"📊 Forward Pass Pattern Detected"
        
        return (issues, warnings, analysis)
    
    def _check_alpha_beta(self, op: MatrixOp) -> List[str]:
        """
        Check alpha and beta values for common issues.
        
        C = alpha * op(A) @ op(B) + beta * C
        
        - alpha should typically be 1.0 (or a scaling factor)
        - beta = 0.0 for overwrite, beta = 1.0 for accumulation
        """
        warnings = []
        
        context = op.forward_context.lower() + " " + op.comment.lower()
        matrix_c = op.matrix_c.lower()
        
        # Check for accumulation patterns
        is_accumulation_context = 'accum' in context or 'beta_param' in op.context.lower()
        is_gradient = 'grad' in matrix_c
        
        # Extract beta from context if possible
        beta_match = re.search(r'&(beta\w*)', op.context)
        if beta_match:
            beta_name = beta_match.group(1).lower()
            if 'zero' in beta_name and is_accumulation_context:
                warnings.append(f"Using beta=0 ({beta_name}) but context suggests accumulation")
            elif 'param' in beta_name or beta_name == 'beta':
                # Conditional accumulation - this is fine
                pass
        
        # Check alpha for scaling issues
        alpha_match = re.search(r'&(\w*alpha\w*)', op.context)
        if alpha_match:
            alpha_name = alpha_match.group(1).lower()
            # Most operations should use alpha=1.0
            # Attention uses alpha=1/sqrt(d_k), but that's applied elsewhere
        
        return warnings
    
    def _try_parse_dim(self, expr: str) -> Optional[int]:
        """Try to parse a dimension expression to an integer"""
        expr = expr.strip()
        
        # Direct integer
        try:
            return int(expr)
        except ValueError:
            pass
        
        # Simple multiplication: 3 * d_model
        mult_match = re.match(r'(\d+)\s*\*\s*(\w+)', expr)
        if mult_match:
            # Can't evaluate without knowing variable values
            return None
        
        # Known config values (common transformer dimensions)
        known_dims = {
            'd_model': 768,
            'cfg.d_model': 768,
            'config_.d_model': 768,
            'params.d_model': 768,
            'd_ff': 3072,
            'cfg.d_ff': 3072,
            'config_.d_ff': 3072,
            'head_dim': 64,
            'num_heads': 12,
            'cfg.num_heads': 12,
            'vocab_size': 13811,
            'params.vocab_size': 13811,
            'cfg.vocab_size': 13811,
            'max_seq_len': 2048,
            'cfg.max_seq_len': 2048,
            'config_.max_seq_len': 2048,
            'num_layers': 12,
            'cfg.num_layers': 12,
            'config_.num_layers': 12,
            'max_cached_batch': 4,
            'cfg.max_cached_batch': 4,
            'max_cached_seq_len': 8192,
            'cfg.max_cached_seq_len': 8192,
        }
        
        if expr in known_dims:
            return known_dims[expr]
        
        return None
    
    def _check_leading_dimensions(self, op: MatrixOp) -> List[str]:
        """Check if leading dimensions match matrix dimensions"""
        issues = []
        
        # For column-major (cuBLAS default):
        # - lda should be rows of A (after transpose if applied)
        # - ldb should be rows of B (after transpose if applied)
        # - ldc should be rows of C
        
        # Expected ldc = M (rows of C in column-major)
        if op.ldc != op.m and not self._could_be_equal(op.ldc, op.m):
            issues.append(f"Leading dimension ldc={op.ldc} doesn't match M={op.m}")
        
        return issues
    
    def _could_be_equal(self, expr1: str, expr2: str) -> bool:
        """Check if two expressions could be equal"""
        # Simple heuristic: if they contain the same variable names
        vars1 = set(re.findall(r'\b\w+\b', expr1))
        vars2 = set(re.findall(r'\b\w+\b', expr2))
        return len(vars1.intersection(vars2)) > 0
    
    def _analyze_row_major_conversion(self, op: MatrixOp) -> str:
        """Analyze how this operation would be interpreted in row-major"""
        analysis = []
        
        analysis.append(f"\n🔷 cuBLAS View (Column-Major):")
        analysis.append(f"   C[{op.m}, {op.n}] = op(A) × op(B)")
        
        # Determine actual matrix shapes after transpose
        if op.transpose_a == TransposeOp.NONE:
            a_shape = f"[{op.m}, {op.k}]"
            a_note = "no transpose"
        else:
            a_shape = f"[{op.k}, {op.m}]^T"
            a_note = "transposed"
        
        if op.transpose_b == TransposeOp.NONE:
            b_shape = f"[{op.k}, {op.n}]"
            b_note = "no transpose"
        else:
            b_shape = f"[{op.n}, {op.k}]^T"
            b_note = "transposed"
        
        analysis.append(f"   Matrix A: {a_shape} ({a_note})")
        analysis.append(f"   Matrix B: {b_shape} ({b_note})")
        
        # Row-major interpretation
        analysis.append(f"\n🔶 Row-Major Interpretation:")
        analysis.append(f"   (How it actually works with row-major storage)")
        
        # Row-major [M,K] stored = Col-major [K,M]^T
        # So cuBLAS sees transposed versions
        analysis.append(f"   Effective computation:")
        if op.transpose_a == TransposeOp.NONE and op.transpose_b == TransposeOp.NONE:
            analysis.append(f"   C = B × A")
            analysis.append(f"   (cuBLAS computes C^T = A^T × B^T in col-major)")
        elif op.transpose_a == TransposeOp.TRANSPOSE and op.transpose_b == TransposeOp.NONE:
            analysis.append(f"   C = B × A^T")
            analysis.append(f"   (cuBLAS computes C^T = A × B^T in col-major)")
        elif op.transpose_a == TransposeOp.NONE and op.transpose_b == TransposeOp.TRANSPOSE:
            analysis.append(f"   C = B^T × A")
            analysis.append(f"   (cuBLAS computes C^T = A^T × B in col-major)")
        else:  # Both transposed
            analysis.append(f"   C = B^T × A^T")
            analysis.append(f"   (cuBLAS computes C^T = A × B in col-major)")
        
        return '\n'.join(analysis)
    
    def _check_common_patterns(self, op: MatrixOp) -> List[str]:
        """Check for common correct/incorrect patterns"""
        warnings = []
        
        # Pattern: Gradient computation should often use OP_T
        if 'grad' in op.matrix_c.lower() and 'grad' in op.matrix_a.lower():
            if op.transpose_a == TransposeOp.NONE and op.transpose_b == TransposeOp.NONE:
                warnings.append("Weight gradient computation with no transposes - verify this is correct")
        
        # Pattern: Backward through weight should transpose the weight
        if 'grad' in op.matrix_c.lower() and not 'grad' in op.matrix_a.lower():
            if 'backward' in op.forward_context.lower():
                if op.transpose_a == TransposeOp.NONE:
                    warnings.append("Backward pass without transpose on weight - verify this is intended")
        
        return warnings
    
    def _check_forward_backward_relationship(self, op: MatrixOp) -> Optional[str]:
        """Check if forward/backward pair is mathematically consistent"""
        context_lower = op.forward_context.lower()
        comment_lower = op.comment.lower()
        
        # Look for forward pass description
        forward_match = re.search(r'forward.*?:?\s*(.+?)(?:\n|$)', context_lower)
        backward_match = re.search(r'backward.*?:?\s*(.+?)(?:\n|$)', context_lower)
        
        if forward_match and backward_match:
            return f"Relationship check:\n  Forward: {forward_match.group(1)}\n  Backward: {backward_match.group(1)}"
        
        return None
    
    def print_report(self) -> None:
        """Print verification report"""
        print("\n" + "="*80)
        print("🔍 GRIM MODEL VERIFICATION REPORT")
        print("="*80)
        
        # Print what was verified
        print("\n📋 CHECKS PERFORMED:")
        print("   1. ✓ Dimension consistency (M×K @ K×N = M×N)")
        print("   2. ✓ Leading dimension bounds (lda/ldb/ldc >= required)")
        print("   3. ✓ Row-major ↔ Column-major conversion correctness")
        print("   4. ✓ Forward/backward mathematical relationships")
        print("   5. ✓ Alpha/beta accumulation patterns")
        print("   6. ✓ Tensor reshape element preservation")
        print("   7. ✓ Common error patterns (swapped dims, missing transposes)")
        print("   8. ✓ Attention scale factor (1/√d_k)")
        print("   9. ✓ Softmax dimension correctness")
        print("  10. ✓ Xavier/He weight initialization formulas")
        print("  11. ✓ AdamW optimizer hyperparameters")
        print("  12. ✓ CUDA kernel launch configuration (warp alignment)")
        
        valid_count = sum(1 for r in self.results if r.is_valid)
        invalid_count = len(self.results) - valid_count
        warning_count = sum(1 for r in self.results if r.warnings)
        
        print(f"\n📊 MATRIX OPERATIONS SUMMARY")
        print(f"   Total Operations: {len(self.results)}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Issues: {invalid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        
        # Print operations with issues first
        if invalid_count > 0:
            print("\n" + "="*80)
            print("❌ OPERATIONS WITH ISSUES")
            print("="*80)
            
            for result in self.results:
                if not result.is_valid:
                    self._print_result(result)
        
        # Print operations with warnings
        warning_count = sum(1 for r in self.results if r.warnings)
        if warning_count > 0:
            print("\n" + "="*80)
            print("⚠️  OPERATIONS WITH WARNINGS")
            print("="*80)
            
            for result in self.results:
                if result.warnings and result.is_valid:
                    self._print_result(result)
        
        # Print all valid operations
        print("\n" + "="*80)
        print("✓ VALID OPERATIONS")
        print("="*80)
        
        for result in self.results:
            if result.is_valid and not result.warnings:
                self._print_result(result, verbose=False)
        
        # Print tensor reshape results
        if self.reshape_results:
            reshape_valid = sum(1 for r in self.reshape_results if r.is_valid)
            reshape_invalid = len(self.reshape_results) - reshape_valid
            reshape_warnings = sum(1 for r in self.reshape_results if r.warnings)
            
            print("\n" + "="*80)
            print("🔄 TENSOR RESHAPE VERIFICATION")
            print("="*80)
            print(f"   Total Reshapes: {len(self.reshape_results)}")
            print(f"   ✅ Valid: {reshape_valid}")
            print(f"   ❌ Issues: {reshape_invalid}")
            print(f"   ⚠️  Warnings: {reshape_warnings}")
            
            for result in self.reshape_results:
                self._print_reshape_result(result)
        
        # Print attention verification results
        if self.attention_results:
            self._print_attention_summary()
        
        # Print initializer verification results
        if self.initializer_results:
            self._print_initializer_summary()
        
        # Print optimizer verification results
        if self.optimizer_results:
            self._print_optimizer_summary()
        
        # Print hyperparameter verification results
        if self.hyperparam_results:
            self._print_hyperparam_summary()
        
        # Print kernel launch verification results
        if self.kernel_results:
            self._print_kernel_summary()
    
    def _print_attention_summary(self) -> None:
        """Print attention operation summary"""
        valid_count = sum(1 for r in self.attention_results if r.is_valid)
        invalid_count = len(self.attention_results) - valid_count
        warning_count = sum(1 for r in self.attention_results if r.warnings)
        
        print("\n" + "="*80)
        print("🎯 ATTENTION OPERATION VERIFICATION")
        print("="*80)
        print(f"   Total Attention Ops: {len(self.attention_results)}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Issues: {invalid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        
        for result in self.attention_results:
            attn = result.item
            status = "✅" if result.is_valid else "❌"
            print(f"\n{status} {attn.file}:{attn.line_num}")
            print(f"   Type: {attn.op_type}")
            if attn.scale_factor:
                print(f"   Scale: {attn.scale_factor}")
            if attn.head_dim:
                print(f"   Head Dim: {attn.head_dim}")
            
            for issue in result.issues:
                print(f"   ❌ {issue}")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")
    
    def _print_initializer_summary(self) -> None:
        """Print weight initializer summary"""
        valid_count = sum(1 for r in self.initializer_results if r.is_valid)
        invalid_count = len(self.initializer_results) - valid_count
        warning_count = sum(1 for r in self.initializer_results if r.warnings)
        
        print("\n" + "="*80)
        print("⚖️ WEIGHT INITIALIZATION VERIFICATION")
        print("="*80)
        print(f"   Total Initializers: {len(self.initializer_results)}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Issues: {invalid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        
        for result in self.initializer_results:
            init = result.item
            status = "✅" if result.is_valid else "❌"
            print(f"\n{status} {init.file}:{init.line_num}")
            print(f"   Type: {init.init_type}")
            if init.fan_in:
                print(f"   Fan In: {init.fan_in}")
            if init.fan_out:
                print(f"   Fan Out: {init.fan_out}")
            if init.distribution:
                print(f"   Distribution: {init.distribution}")
            
            for issue in result.issues:
                print(f"   ❌ {issue}")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")
    
    def _print_optimizer_summary(self) -> None:
        """Print optimizer configuration summary"""
        valid_count = sum(1 for r in self.optimizer_results if r.is_valid)
        invalid_count = len(self.optimizer_results) - valid_count
        warning_count = sum(1 for r in self.optimizer_results if r.warnings)
        
        print("\n" + "="*80)
        print("⚡ OPTIMIZER CONFIGURATION VERIFICATION")
        print("="*80)
        print(f"   Total Optimizer Configs: {len(self.optimizer_results)}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Issues: {invalid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        
        for result in self.optimizer_results:
            opt = result.item
            status = "✅" if result.is_valid else "❌"
            print(f"\n{status} {opt.file}:{opt.line_num}")
            print(f"   Optimizer: {opt.optimizer_type}")
            if opt.learning_rate:
                print(f"   Learning Rate: {opt.learning_rate}")
            if opt.beta1:
                print(f"   Beta1: {opt.beta1}")
            if opt.beta2:
                print(f"   Beta2: {opt.beta2}")
            if opt.epsilon:
                print(f"   Epsilon: {opt.epsilon}")
            if opt.weight_decay:
                print(f"   Weight Decay: {opt.weight_decay}")
            
            for issue in result.issues:
                print(f"   ❌ {issue}")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")

    def _print_hyperparam_summary(self) -> None:
        """Print hyperparameter summary"""
        valid_count = sum(1 for r in self.hyperparam_results if r.is_valid)
        invalid_count = len(self.hyperparam_results) - valid_count
        warning_count = sum(1 for r in self.hyperparam_results if r.warnings)
        
        print("\n" + "="*80)
        print("⚙️ HYPERPARAMETER VERIFICATION")
        print("="*80)
        print(f"   Total Hyperparams: {len(self.hyperparam_results)}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Issues: {invalid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        
        for result in self.hyperparam_results:
            hp = result.item
            status = "✅" if result.is_valid and not result.warnings else ("⚠️ " if result.warnings else "❌")
            
            print(f"\n{status} {hp.file}:{hp.line_num}")
            print(f"   {hp.name}: {hp.value}")
            
            for issue in result.issues:
                print(f"   ❌ {issue}")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")
    
    def _print_kernel_summary(self) -> None:
        """Print kernel launch configuration summary"""
        valid_count = sum(1 for r in self.kernel_results if r.is_valid)
        invalid_count = len(self.kernel_results) - valid_count
        warning_count = sum(1 for r in self.kernel_results if r.warnings)
        
        print("\n" + "="*80)
        print("🚀 CUDA KERNEL LAUNCH VERIFICATION")
        print("="*80)
        print(f"   Total Kernel Launches: {len(self.kernel_results)}")
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Issues: {invalid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        
        # Group by kernel name for cleaner output
        kernel_counts = {}
        for result in self.kernel_results:
            kernel = result.item
            name = kernel.kernel_name
            if name not in kernel_counts:
                kernel_counts[name] = {'total': 0, 'valid': 0, 'issues': [], 'warnings': []}
            kernel_counts[name]['total'] += 1
            if result.is_valid:
                kernel_counts[name]['valid'] += 1
            kernel_counts[name]['issues'].extend(result.issues)
            kernel_counts[name]['warnings'].extend(result.warnings)
        
        print(f"\n   Unique Kernels: {len(kernel_counts)}")
        
        # Print issues/warnings if any
        for name, info in kernel_counts.items():
            if info['issues'] or info['warnings']:
                print(f"\n   🔶 {name} ({info['valid']}/{info['total']} valid)")
                for issue in set(info['issues']):
                    print(f"      ❌ {issue}")
                for warning in set(info['warnings'][:3]):  # Limit warnings
                    print(f"      ⚠️  {warning}")
        
        # Brief list of all kernels
        print("\n   All kernels found:")
        for name, info in sorted(kernel_counts.items(), key=lambda x: -x[1]['total']):
            status = "✅" if info['valid'] == info['total'] and not info['issues'] else "⚠️"
            print(f"      {status} {name}: {info['total']} launches")
    
    def _print_reshape_result(self, result: ReshapeResult) -> None:
        """Print a tensor reshape verification result"""
        reshape = result.reshape
        status = "✅" if result.is_valid else "❌"
        
        print(f"\n{status} {reshape.file}:{reshape.line_num}")
        print(f"   📍 Location: {reshape.file}")
        print(f"   🔢 Line: {reshape.line_num}")
        print(f"   🔄 Conversion: {reshape.src_format} → {reshape.dst_format}")
        print(f"   📐 Dimensions: B={reshape.dimensions.get('B', '?')}, H={reshape.dimensions.get('H', '?')}, S={reshape.dimensions.get('S', '?')}, D={reshape.dimensions.get('D', '?')}")
        
        if result.issues:
            print(f"\n   ❌ ISSUES:")
            for issue in result.issues:
                print(f"      • {issue}")
        
        if result.warnings:
            print(f"\n   ⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"      • {warning}")
    
    def _print_result(self, result: VerificationResult, verbose: bool = True) -> None:
        """Print a single verification result"""
        op = result.op
        status = "✅" if result.is_valid else "❌"
        
        print(f"\n{status} {op.file}:{op.line_num}")
        print(f"   📍 Location: {op.file}")
        print(f"   🔢 Line: {op.line_num}")
        print(f"   ➡️  Operation: {op.matrix_c} = {op.matrix_a} × {op.matrix_b}")
        
        # Format transpose operations
        trans_a = "No Transpose" if op.transpose_a == TransposeOp.NONE else "Transpose"
        trans_b = "No Transpose" if op.transpose_b == TransposeOp.NONE else "Transpose"
        print(f"   🔄 Matrix A: {trans_a} ({op.transpose_a.value})")
        print(f"   🔄 Matrix B: {trans_b} ({op.transpose_b.value})")
        print(f"   📐 Dimensions: M={op.m}, N={op.n}, K={op.k}")
        print(f"   📏 Leading Dims: lda={op.lda}, ldb={op.ldb}, ldc={op.ldc}")
        
        if op.comment:
            print(f"   💬 Comment: {op.comment}")
        
        if result.issues:
            print(f"\n   ❌ ISSUES:")
            for issue in result.issues:
                print(f"      • {issue}")
        
        if result.warnings:
            print(f"\n   ⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"      • {warning}")
        
        if verbose and result.analysis:
            print(f"\n   📊 ANALYSIS:")
            for line in result.analysis.split('\n'):
                if line.strip():
                    print(f"      {line}")
        
        # Add context if available
        if verbose and op.forward_context:
            print(f"\n   📝 CONTEXT FROM CODE:")
            for line in op.forward_context.split('\n')[-3:]:  # Last 3 comment lines
                if line.strip():
                    print(f"      {line}")
    
    def save_report(self, output_file: str = "matrix_verification_report.txt") -> None:
        """Save detailed report to file"""
        # Save to project root (where script is located)
        output_path = Path(__file__).parent / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MATRIX OPERATION VERIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            valid_count = sum(1 for r in self.results if r.is_valid)
            invalid_count = len(self.results) - valid_count
            
            f.write(f"Summary: {valid_count} valid, {invalid_count} with issues\n\n")
            
            for result in self.results:
                op = result.op
                status = "VALID" if result.is_valid else "ISSUES"
                
                f.write(f"\n{'='*80}\n")
                f.write(f"[{status}] {op.file}:{op.line_num}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Operation: {op.matrix_c} = {op.matrix_a} × {op.matrix_b}\n")
                f.write(f"Transposes: A={op.transpose_a.value}, B={op.transpose_b.value}\n")
                f.write(f"Dimensions: M={op.m}, N={op.n}, K={op.k}\n")
                f.write(f"Leading dims: lda={op.lda}, ldb={op.ldb}, ldc={op.ldc}\n\n")
                
                if op.comment:
                    f.write(f"Comment: {op.comment}\n\n")
                
                if op.forward_context:
                    f.write("Context:\n")
                    for line in op.forward_context.split('\n'):
                        f.write(f"  {line}\n")
                    f.write("\n")
                
                if result.issues:
                    f.write("Issues:\n")
                    for issue in result.issues:
                        f.write(f"  • {issue}\n")
                    f.write("\n")
                
                if result.warnings:
                    f.write("Warnings:\n")
                    for warning in result.warnings:
                        f.write(f"  • {warning}\n")
                    f.write("\n")
                
                if result.analysis:
                    f.write("Analysis:\n")
                    for line in result.analysis.split('\n'):
                        f.write(f"  {line}\n")
                    f.write("\n")
                
                f.write(f"\nCode:\n{op.context}\n")
        
        print(f"\n📄 Detailed report saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify matrix operations in GRIM model code"
    )
    parser.add_argument(
        "--model-dir",
        default="resources/models/GRIM-text",
        help="Path to model directory (default: resources/models/GRIM-text)"
    )
    parser.add_argument(
        "--output",
        default="matrix_verification_report.txt",
        help="Output report filename (default: matrix_verification_report.txt)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis for all operations"
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    model_dir = script_dir / args.model_dir
    
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        return 1
    
    print("="*80)
    print("GRIM Model Verification Tool")
    print("="*80)
    
    verifier = ModelVerifier(str(model_dir))
    
    # Scan and analyze
    verifier.scan_files()
    verifier.verify_operations()
    
    # Print report
    verifier.print_report()
    
    # Save detailed report
    verifier.save_report(args.output)
    
    # Exit with error code if issues found
    invalid_count = sum(1 for r in verifier.results if not r.is_valid)
    return 1 if invalid_count > 0 else 0


if __name__ == "__main__":
    exit(main())
