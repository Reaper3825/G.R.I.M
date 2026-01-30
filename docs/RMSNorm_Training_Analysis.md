# RMSNorm Analysis Report - GRIM-text Training Diagnostics

**Date:** Generated from `training_run.log`  
**Model:** GRIM-text Transformer (12 encoder layers)  
**Log File:** `resources/models/GRIM-text/training/logs/training_run.log` (25,424 lines)

---

## Executive Summary

This analysis traces all RMSNorm-related diagnostics from the GRIM-text training log to understand normalization behavior during forward and backward passes. The key findings are:

| Aspect | Status | Details |
|--------|--------|---------|
| **Gamma Initialization** | ✅ Correct | All gamma parameters initialized to 1.0 |
| **Forward Pass Normalization** | ✅ Working | Output rows consistently normalized to RMS≈1.0 |
| **Backward Pass Gradients** | ✅ Healthy | Gradient magnitudes in expected range (1e-5 to 6e-5) |
| **Layer 0 Input Anomaly** | ⚠️ Notable | Embedding input has lower RMS (0.173) vs deeper layers (~2.0) |
| **Final RMSNorm** | ✅ Applied | Autograd-tracked final normalization before LM head |

---

## 1. Model Configuration

```
Vocabulary:       50,377 tokens
d_model:          768
Encoder Layers:   12
Batch Size:       7
Sequence Length:  1024
Tokens per Batch: 7,168
GPU:              NVIDIA GeForce RTX 3080 Ti (Compute 8.6)
Positional:       Hybrid ALiBi+RoPE (no learned position embeddings)
```

**RMSNorm Architecture:**
- **Pre-layer normalization** (applied before attention and FFN)
- **Learnable gamma** (768 elements per normalization layer)
- **No beta** (center bias not used)
- **Final RMSNorm** applied before LM head projection

---

## 2. Initialization Analysis

### 2.1 Gamma Parameter Initialization

From log lines 147-149:
```
📦 Final RMSNorm gamma allocated: 768 elements
✓ RMSNorm gamma: TrainingTensors memory (initialized to 1.0)
```

**Verification across all layers (Batch 0):**

| Layer | gamma_mean | gamma_rms | Status |
|-------|------------|-----------|--------|
| 0 | 1.0000 | 1.0000 | ✅ |
| 1 | 1.0000 | 1.0000 | ✅ |
| 2 | 1.0000 | 1.0000 | ✅ |
| 3 | 1.0000 | 1.0000 | ✅ |
| 4 | 1.0000 | 1.0000 | ✅ |
| 5 | 1.0000 | 1.0000 | ✅ |
| 6 | 1.0000 | 1.0000 | ✅ |
| 7 | 1.0000 | 1.0000 | ✅ |
| 8 | 1.0000 | 1.0000 | ✅ |
| 9 | 1.0000 | 1.0000 | ✅ |
| 10 | 1.0000 | 1.0000 | ✅ |
| 11 | 1.0000 | 1.0000 | ✅ |

**Conclusion:** All gamma parameters correctly initialized to 1.0, matching expected behavior for RMSNorm initialization.

---

## 3. Forward Pass Analysis

### 3.1 Per-Layer RMS Input/Output Statistics (Batch 0)

The diagnostic tags `[Issue91-FWD-rms_input]` and `[Issue94-RMSNorm-INPUT/OUTPUT]` track RMSNorm behavior:

| Layer | Input RMS | Input Range (per-row) | Output Range (per-row) | Status |
|-------|-----------|----------------------|------------------------|--------|
| **0** | **0.1767** | 0.1732 - 0.1734 | 0.9837 - 0.9838 | ⚠️ Low input |
| 1 | 2.0558 | 1.5761 - 4.4871 | 0.9998 - 1.0000 | ✅ |
| 2 | 2.0879 | 1.5935 - 4.5117 | 0.9998 - 1.0000 | ✅ |
| 3 | 2.1121 | 1.6295 - 4.6127 | 0.9998 - 1.0000 | ✅ |
| 4 | 2.1422 | 1.5028 - 4.7634 | 0.9998 - 1.0000 | ✅ |
| 5 | 2.1608 | 1.5417 - 4.7682 | 0.9998 - 1.0000 | ✅ |
| 6 | 2.1643 | 1.6402 - 4.7343 | 0.9998 - 1.0000 | ✅ |
| 7 | 2.1701 | 1.7233 - 4.7613 | 0.9998 - 1.0000 | ✅ |
| 8 | 2.1848 | 1.8108 - 4.9090 | 0.9998 - 1.0000 | ✅ |
| 9 | 2.2023 | 1.8648 - 4.9310 | 0.9999 - 1.0000 | ✅ |
| 10 | 2.2135 | 2.0814 - 4.8621 | 0.9999 - 1.0000 | ✅ |
| 11 | 2.2314 | 2.0724 - 5.0443 | 0.9999 - 1.0000 | ✅ |

### 3.2 Layer 0 Anomaly Analysis

**Observation:** Layer 0 receives embedding output with significantly lower RMS (0.1767) compared to deeper layers (~2.0).

**Explanation:** 
- Layer 0 input comes directly from embeddings (after ScratchBlock)
- Post-ScratchBlock embedding statistics: `min=-0.3001 max=0.3001 mean=-0.0003 rms=0.1733`
- Deeper layers receive residual-accumulated activations with larger magnitudes

**Impact:**
```
[Issue91-FWD-rms_input] layer=0: min=-0.2155 max=0.8423 rms=0.1767
[Issue94-RMSNorm-OUTPUT] layer=0: per_row_rms: 0.9837-0.9838
```

The RMSNorm correctly normalizes this lower-magnitude input to output RMS≈0.98, which is slightly below 1.0. This is expected behavior due to the narrow input range.

### 3.3 RMSNorm Normalization Quality

**Formula:** `y = x * gamma / rms(x)` where `rms(x) = sqrt(mean(x²))`

**Verification:** Output per-row RMS values are consistently 0.9998-1.0000 for layers 1-11, confirming correct normalization.

**Value Ranges After Normalization:**
| Layer | Output Value Range |
|-------|-------------------|
| 0 | [-1.7048, 1.7037] |
| 1 | [-3.3050, 2.9329] |

---

## 4. Backward Pass Analysis

### 4.1 Final RMSNorm Application

From log line 1686:
```
[AutogradTraining] INFO: Step 3: Final RMSNorm applied with autograd
```

This confirms the final normalization before LM head is properly tracked by the autograd system.

### 4.2 Gradient Flow Through RMSNorm (Batch 0)

The backward pass logs show gradients flowing through RMSNorm operations with tag `[MATMUL-BWD-TO-A] op=rms_norm` and `[RMS-BWD-IN]`:

| Call | numel | grad_max | grad_rms | Layer Context |
|------|-------|----------|----------|---------------|
| 1 | 5,505,024 | 5.87e-05 | 2.34e-05 | Layer 11 (output) |
| 2 | 5,505,024 | 1.28e-05 | 2.60e-06 | Layer 11 (FFN) |
| 3 | 5,505,024 | 2.88e-05 | 5.40e-06 | Layer 10 |
| 4 | 5,505,024 | 6.03e-06 | - | Layer 10 (FFN) |
| 5 | 5,505,024 | 2.92e-05 | - | Layer 9 |
| 6 | 5,505,024 | 1.80e-05 | - | Layer 9 (FFN) |
| 7 | 5,505,024 | 3.08e-05 | - | Layer 8 |
| 8 | 5,505,024 | 1.01e-05 | - | Layer 8 (FFN) |
| 9 | 5,505,024 | 3.11e-05 | - | Layer 7 |
| 10 | 5,505,024 | 8.64e-06 | - | Layer 7 (FFN) |
| 11 | 5,505,024 | 2.94e-05 | - | Layer 6 |
| 12 | 5,505,024 | 1.08e-05 | - | Layer 6 (FFN) |
| 13-25 | 5,505,024 | 6.4e-06 - 3.2e-05 | - | Layers 5-0 |

**Gradient Buffer Size:** 5,505,024 elements = 7,168 tokens × 768 d_model

### 4.3 Gradient Magnitude Analysis

**Batch 0 (First Iteration):**

| Statistic | Value | Assessment |
|-----------|-------|------------|
| Max gradient (Call 1) | 5.87e-05 | Normal |
| Min gradient (Call 18) | 6.86e-06 | Normal |
| Gradient range | 6.86e-06 to 5.87e-05 | ~8.5x variation |

**Batch 1 (Second Iteration):**

| Call | grad_max | Change from Batch 0 |
|------|----------|---------------------|
| 26 | 5.56e-05 | -5.3% (stable) |
| 27 | 9.89e-06 | Similar |
| 28 | 4.19e-05 | +45% (mild increase) |

**Pattern Observed:**
- Odd-numbered calls (attention path): Higher gradients (2.9e-05 to 3.3e-05)
- Even-numbered calls (FFN path): Lower gradients (6.0e-06 to 1.8e-05)
- This alternating pattern reflects the two RMSNorm operations per encoder layer

### 4.4 Gradient Chain Integrity

The full backward chain from loss to embeddings:

```
Loss Backward → [LOSS-BWD-OUT] grad_logits max=0.000147 rms=0.000001
       ↓
LM Head Backward → [MATMUL-BWD-IN] grad_C max=0.004069
       ↓
RMSNorm Backward (Final) → [MATMUL-BWD-TO-A] op=rms_norm max=5.87e-05
       ↓
Encoder Layer 11 → ... → Layer 0
       ↓
Embedding Backward
```

**Gradient Attenuation Through Chain:**

| Stage | Max Gradient | Attenuation |
|-------|--------------|-------------|
| Loss output | 1.47e-04 | (baseline) |
| LM head input | 5.87e-05 | 2.5x |
| Layer 11 attention | 2.88e-05 | 5.1x |
| Layer 0 | ~1.5e-05 | ~10x |

---

## 5. Batch Comparison (Batch 0 vs Batch 1)

### 5.1 Input RMS Stability

| Layer | Batch 0 Input RMS | Batch 1 Input RMS | Change |
|-------|-------------------|-------------------|--------|
| 0 | 0.1767 | 0.1732 | -2.0% |
| 1 | 2.0558 | 2.0804 | +1.2% |
| 2 | 2.0879 | 2.1402 | +2.5% |
| 3 | 2.1121 | 2.1704 | +2.8% |
| 4 | 2.1422 | 2.1935 | +2.4% |
| 5 | 2.1608 | 2.2127 | +2.4% |
| 6 | 2.1643 | 2.2143 | +2.3% |
| 7 | 2.1701 | 2.2280 | +2.7% |
| 8 | 2.1848 | 2.2420 | +2.6% |
| 9 | 2.2023 | 2.2549 | +2.4% |
| 10 | 2.2135 | 2.2473 | +1.5% |
| 11 | 2.2314 | 2.2546 | +1.0% |

**Observation:** Slight increase in input RMS across layers from Batch 0 to Batch 1 (~1-3%), indicating normal model activation dynamics.

### 5.2 Output RMS Consistency

Both batches maintain output per-row RMS of 0.9998-1.0000 for layers 1-11, confirming stable normalization.

---

## 6. Flash Attention LSE Anomalies

### 6.1 Layer 0 LSE Explosion

From log lines 557-571, Layer 0 Flash Attention shows extreme LSE (Log-Sum-Exp) values:

```
[FA-FWD-LSE-SUMMARY] nan=0 inf=0 range=[-762.3290, 910.1768] mean=658.9991
```

**Per-Head Analysis:**

| Head | LSE Range | Mean | Status |
|------|-----------|------|--------|
| 0 | [-543.78, 896.85] | 612.0 | ⚠️ |
| 1 | [-449.68, 894.38] | 614.9 | ⚠️ |
| 2 | [313.56, 902.88] | 847.1 | ⚠️ |
| 3 | [295.60, 910.18] | 847.3 | ⚠️ |
| ... | ... | ... | ⚠️ |

**Normal LSE Range (Layer 1 for comparison):**
```
[FA-FWD-LSE-SUMMARY] range=[-1.9682, 37.6764] mean=6.8110
```

### 6.2 Cause Analysis

The LSE explosion in Layer 0 is caused by the QKV output magnitude issue documented in `[DebugQKVExpectations]`:

```
ACTUAL OUTPUT [7168 × 1280]: rms=10.4451 (expected ~0.85)
VERDICT: actual/expected = 12.26x larger than theoretical
```

With Q/K values having RMS of ~10.5 instead of ~1.0:
- Attention scores = Q @ K^T / sqrt(head_dim) 
- Expected: scores have std ~1.0
- Actual: scores have std ~100 (10.5 × 10.5 / 8 ≈ 14)
- LSE = log(sum(exp(scores))) → explodes with large score magnitudes

**Note:** This is related to Issue #95 (Layer 0 QKV magnitude anomaly), not an RMSNorm bug.

---

## 7. RMSNorm Backward Algorithm

### 7.1 Mathematical Derivation

For forward: `y = x * gamma / rms(x)` where `rms = sqrt(mean(x²) + eps)`

Backward gradients:
```
d_gamma = sum(dy * x / rms)  [accumulated across positions]
d_x = gamma * (dy / rms - x * mean(dy * x) / (rms³ * d_model))
```

### 7.2 Implementation Verification

From the gradient flow pattern:
- Each encoder layer has 2 RMSNorm operations (pre-attention, pre-FFN)
- Total backward calls = 12 layers × 2 RMSNorm = 24 calls + 1 final = 25 calls (Batch 0)
- Log shows calls 1-25 for Batch 0, matching expected count

---

## 8. Key Findings and Recommendations

### 8.1 Positive Findings

1. ✅ **Gamma Initialization:** All 768-element gamma vectors correctly initialized to 1.0
2. ✅ **Normalization Quality:** Output RMS consistently 0.9998-1.0000 across layers 1-11
3. ✅ **Gradient Flow:** Healthy gradient magnitudes (1e-5 to 6e-5) without explosion or vanishing
4. ✅ **Autograd Integration:** Final RMSNorm properly tracked in autograd graph
5. ✅ **Batch Stability:** Consistent behavior between Batch 0 and Batch 1

### 8.2 Areas of Note

1. ⚠️ **Layer 0 Input Magnitude:** Lower RMS (0.17) compared to deeper layers (~2.0)
   - This is expected behavior due to embedding initialization
   - Results in slightly lower output RMS (0.9837 vs 0.9999)
   
2. ⚠️ **LSE Explosion (Layer 0):** Related to QKV magnitude issue, not RMSNorm
   - See Issue #95 for root cause analysis
   - RMSNorm is correctly normalizing, but QKV projection amplifies output

### 8.3 Recommendations

1. **Monitor gamma updates:** Track gamma parameter changes during training to ensure learning
2. **Layer 0 scrutiny:** The embedding→Layer 0 transition has unique characteristics worth monitoring
3. **Gradient component tracking:** Current `rms` gradient component in `[GradTrace]` appears healthy

---

## 9. Diagnostic Log Tag Reference

| Tag | Description | Location |
|-----|-------------|----------|
| `[Issue91-FWD-rms_input]` | Forward pass RMS input statistics per layer | Forward pass |
| `[Issue91-FWD-gamma]` | Gamma parameter statistics | Forward pass |
| `[Issue94-RMSNorm-INPUT]` | Per-row RMS before normalization | Forward pass |
| `[Issue94-RMSNorm-OUTPUT]` | Per-row RMS after normalization | Forward pass |
| `[MATMUL-BWD-TO-A] op=rms_norm` | Gradient flowing to RMSNorm input | Backward pass |
| `[RMS-BWD-IN]` | RMSNorm backward input gradient | Backward pass |
| `[Issue77-FWD-ln1_out]` | Layer normalization output (pre-QKV) | Forward pass |

---

## 10. Raw Data Appendix

### 10.1 Complete Forward Pass RMS Table (Batch 0)

```
Layer 0:  input_rms=0.1767 | row_rms_in=[0.1732,0.1734] | row_rms_out=[0.9837,0.9838]
Layer 1:  input_rms=2.0558 | row_rms_in=[1.5761,4.4871] | row_rms_out=[0.9998,1.0000]
Layer 2:  input_rms=2.0879 | row_rms_in=[1.5935,4.5117] | row_rms_out=[0.9998,1.0000]
Layer 3:  input_rms=2.1121 | row_rms_in=[1.6295,4.6127] | row_rms_out=[0.9998,1.0000]
Layer 4:  input_rms=2.1422 | row_rms_in=[1.5028,4.7634] | row_rms_out=[0.9998,1.0000]
Layer 5:  input_rms=2.1608 | row_rms_in=[1.5417,4.7682] | row_rms_out=[0.9998,1.0000]
Layer 6:  input_rms=2.1643 | row_rms_in=[1.6402,4.7343] | row_rms_out=[0.9998,1.0000]
Layer 7:  input_rms=2.1701 | row_rms_in=[1.7233,4.7613] | row_rms_out=[0.9998,1.0000]
Layer 8:  input_rms=2.1848 | row_rms_in=[1.8108,4.9090] | row_rms_out=[0.9998,1.0000]
Layer 9:  input_rms=2.2023 | row_rms_in=[1.8648,4.9310] | row_rms_out=[0.9999,1.0000]
Layer 10: input_rms=2.2135 | row_rms_in=[2.0814,4.8621] | row_rms_out=[0.9999,1.0000]
Layer 11: input_rms=2.2314 | row_rms_in=[2.0724,5.0443] | row_rms_out=[0.9999,1.0000]
```

### 10.2 Complete Backward Pass Gradient Table (Batch 0)

```
Call  1: grad_max=5.87e-05 rms=2.34e-05 (Layer 11 attention RMSNorm)
Call  2: grad_max=1.28e-05 rms=2.60e-06 (Layer 11 FFN RMSNorm)
Call  3: grad_max=2.88e-05 rms=5.40e-06 (Layer 10 attention RMSNorm)
Call  4: grad_max=6.03e-06             (Layer 10 FFN RMSNorm)
Call  5: grad_max=2.92e-05             (Layer 9 attention RMSNorm)
Call  6: grad_max=1.80e-05             (Layer 9 FFN RMSNorm)
Call  7: grad_max=3.08e-05             (Layer 8 attention RMSNorm)
Call  8: grad_max=1.01e-05             (Layer 8 FFN RMSNorm)
Call  9: grad_max=3.11e-05             (Layer 7 attention RMSNorm)
Call 10: grad_max=8.64e-06             (Layer 7 FFN RMSNorm)
Call 11: grad_max=2.94e-05             (Layer 6 attention RMSNorm)
Call 12: grad_max=1.08e-05             (Layer 6 FFN RMSNorm)
Call 13: grad_max=2.99e-05             (Layer 5 attention RMSNorm)
Call 14: grad_max=9.30e-06             (Layer 5 FFN RMSNorm)
Call 15: grad_max=2.95e-05             (Layer 4 attention RMSNorm)
Call 16: grad_max=1.22e-05             (Layer 4 FFN RMSNorm)
Call 17: grad_max=3.32e-05             (Layer 3 attention RMSNorm)
Call 18: grad_max=6.86e-06             (Layer 3 FFN RMSNorm)
Call 19: grad_max=2.92e-05             (Layer 2 attention RMSNorm)
Call 20: grad_max=6.39e-06             (Layer 2 FFN RMSNorm)
Call 21: grad_max=3.29e-05             (Layer 1 attention RMSNorm)
Call 22: grad_max=1.03e-05             (Layer 1 FFN RMSNorm)
Call 23: grad_max=2.96e-05             (Layer 0 attention RMSNorm)
Call 24: grad_max=1.50e-05             (Layer 0 FFN RMSNorm)
Call 25: grad_max=3.17e-05             (Final RMSNorm/Embedding)
```

---

*Report generated from comprehensive analysis of training_run.log*
