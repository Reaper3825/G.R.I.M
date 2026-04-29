# GRIM-text Training Diagnostics Reference

> **Purpose**: Quick reference for all logging, analysis, and gradient debugging tools.

---

## 📂 File Locations

| Tool | Path | Purpose |
|------|------|---------|
| **Log Analyzer** | `training/analyze_training_log.py` | Parses training logs, detects issues, scores health |
| **Gradient Trace** | `training/trace_single_gradient.py` | Analyzes binary gradient dumps |
| **Gradient Validator** | `training/validate_gradient_trace.py` | Numerical gradient checking |
| **Causality Proofs** | `training/run_causality_proofs.py` | 6-level correctness tests |
| **Training Logs** | `training/logs/training_*.log` | Raw training output |
| **Analysis Logs** | `training/logs/analysis_*.log` | Saved analysis reports |
| **Layer Logs** | `training/logs/layers/` | Per-layer diagnostic dumps |

---

## 🔍 Log Analyzer (`analyze_training_log.py`)

### Usage
```powershell
# Analyze latest log
python training/analyze_training_log.py

# Analyze specific log
python training/analyze_training_log.py --log training/logs/training_17656777702397256.log

# Watch mode (live updates)
python training/analyze_training_log.py --watch --interval 5

# JSON output
python training/analyze_training_log.py --json
```

### What It Extracts

#### Loss & Learning Rate
- Initial/current/lowest loss
- LR trajectory and volatility
- Dynamic LR events (spikes, warmup, floor triggers)

#### Gradient Health
```log
[GradCheck] lm_head_weight_grads: rms=0.014, max_abs=0.063
[GradCheck] layer11_grad_Q: rms=0.00068, max_abs=0.003
[GradCheck] layer0_attn_qkv_weight_grads: rms=5.30
```

**Key metrics parsed:**
| Metric | Healthy Range | Warning |
|--------|--------------|---------|
| `lm_head_weight_grads` RMS | 1e-6 to 10 | <1e-6 vanishing, >10 exploding |
| `grad_encoder_out` RMS | 1e-6 to 200 | <1e-6 vanishing |
| Layer gradient ratio | <1000x | >1000x = explosion |

#### Attention Health (Flash Attention)
```log
[FlashBwd] row_max=X.XX row_sum=Y.YY dQ_global=Z.ZZ
[FlashBwd-KV0] P00=0.XXX score00=Y.YY dQ_sum=Z.ZZ
```

**Parsed fields:**
- `P00` - Attention weight at position 0 (1.0 = collapsed)
- `score00` - Pre-softmax score (>15 causes saturation)
- `dQ_global/dQ_sum` - Query gradient magnitude (vanishing = attention not learning)

#### Stability Indicators
- Soft restarts triggered
- GradGuard skips (catastrophic gradients)
- Clipping events
- GPU OOM events

### Health Score Calculation
```
100 - (15 × ⚠️ warnings) - (30 × 🚨 critical)
```

| Score | Status |
|-------|--------|
| 80-100 | ✅ Healthy |
| 50-79 | ⚠️ Needs attention |
| 0-49 | ❌ Critical issues |

---

## 📊 Gradient Trace (`trace_single_gradient.py`)

### Enabling Gradient Dumps

```powershell
# Set env var before training
$env:GRIM_TRACE_GRADIENTS="1"
.\build_vs_cuda\Release\train_gpu.exe

# Produces: gradient_trace_step_N.bin
```

### Analyzing Dumps
```powershell
python training/trace_single_gradient.py gradient_trace_step_0.bin
python training/trace_single_gradient.py gradient_trace_step_0.bin --output report.json
```

### Binary Format
```
Header: magic (0x47524144 = 'GRAD'), version, num_tensors
Per-tensor:
  - name_len (4 bytes)
  - name (UTF-8)
  - ndim (4 bytes)
  - shape (ndim × 4 bytes)
  - dtype_code (4 bytes: 0=float32, 1=float16)
  - data (size × itemsize bytes)
```

### Output Metrics
| Metric | Meaning |
|--------|---------|
| `norm` | L2 norm of gradient tensor |
| `mean` | Average gradient value |
| `std` | Standard deviation |
| `sparsity_pct` | % of values near zero (<1e-7) |

---

## ✅ Gradient Validator (`validate_gradient_trace.py`)

### Numerical Gradient Checking
Compares backprop gradients vs finite differences:
```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
```

### Usage
```powershell
python training/validate_gradient_trace.py gradient_trace_step_0.bin
```

### Checks Performed
1. **Numerical gradient comparison** - Finite diff vs analytical
2. **Chain rule consistency** - Gradient composition verification
3. **Known analytical validation** - Cross-entropy gradient formula
4. **Gradient-weight relationship** - Sign/magnitude sanity

---

## 🧪 Causality Proof Tests (`run_causality_proofs.py`)

### 6 Levels of Correctness

```powershell
python training/run_causality_proofs.py --all
python training/run_causality_proofs.py --level 1  # Single level
```

| Level | Test | Proof |
|-------|------|-------|
| **1** | Single-token causality | ∂loss/∂logits[y] < 0, others > 0, sum ≈ 0 |
| **2** | Causal mask correctness | Attention[t,>t] == 0 |
| **3** | Gradient path continuity | ∂loss/∂embedding[y] ≠ 0 |
| **4** | Learning changes logits | logit[y]_after > logit[y]_before |
| **5** | Tokenizer-loss alignment | Byte/atom tokens receive gradients |
| **6** | Autoregressive emergence | "abc" pattern learned from scratch |

### Expected Output
```
✓ LEVEL 1 PASSED: Single-token causality is correct
✓ LEVEL 2 PASSED: Causal mask is correct
...
Total: 6/6 passed
```

---

## 📝 Log Line Patterns

### Training Progress
```log
[Step 100] loss=2.3456 lr=0.000286
[Batch 50/483] size=2 max_len=1024
```

### Gradient Diagnostics
```log
[GradCheck] <tensor_name>: rms=X.XXe-XX, max_abs=Y.YYe-YY, range=[min,max]
[GradNorm] value=XX.XX per_token=0.XXXX mode=token_norm
[Diag] batch=N loss=X.XX preclip_grad=Y.YY preclip_norm=Z.ZZ
```

### Soft Restart Events
```log
[SoftRestart] triggered at batch N
```

### Attention Debug
```log
[FlashAttn Host] seq=1024 heads=12 head_dim=64
[FlashBwd] row_max[0]=X.XX row_sum[0]=Y.YY dQ_global=Z.ZZ
[FlashBwd-KV0] P00=0.XXX score00=Y.YY dQ_sum=Z.ZZ
```

### Stability Events
```log
[GradGuard] skip optimizer step preclip_norm=XXXXX per_token=YY.YY
[Checkpoint] Saved step=1000
```

---

## 🚨 Common Issues & Solutions

### Sequence Classification (bad_sequences.log)

When a gradient spike causes GradGuard to skip a batch, the problematic sequences are logged with automatic classification:

| Class | Description | Typical Issue |
|-------|-------------|---------------|
| `boilerplate/nav` | UI text, navigation, footers, social links | Often repetitive, low semantic value |
| `documentation` | API docs, technical references | May have unusual structure |
| `prose` | Articles, essays, narrative text | Usually healthy |
| `code` | Programming code snippets | Special tokens, brackets may confuse model |
| `mixed_junk` | Corrupted, garbled, low-quality | Primary cause of gradient explosions |

### Content-Based Loss Weighting

Low-quality sequences are automatically down-weighted during training (not removed):

| Class | Loss Weight | Effect |
|-------|-------------|--------|
| `prose`, `documentation`, `code` | 1.0× | Full gradient contribution |
| `boilerplate/nav` | 0.4× | Reduced influence on training |
| `mixed_junk` | 0.3× | Minimal influence |

**Log output:**
```log
[ContentWeight] batch=1 weights=[1.00,0.40,1.00]
```

**Example output:**
```log
--- Sequence 527 ---
Class: [mixed_junk]
Length: 941 tokens
Text preview (first 500 chars):
<s><|startoftext|> ip }} from the code. ...
```

### Vanishing Gradients
**Symptom**: `layer0_grad_* rms < 1e-6`
```
⚠️ VANISHING GRADS: Layer 0 norm 3.21e-08, Layer 11 norm 4.56e-02
```
**Fix**: Check residual connections, reduce depth, or use gradient checkpointing

### Exploding Gradients
**Symptom**: `grad_norm > 1000` or GradGuard skips
```
🚨 GRADGUARD SKIPS: 50 batches skipped (10%) - max spike 45000
```
**Fix**: Lower LR, increase gradient clip, check data for corruption

### Attention Collapse
**Symptom**: `P00 > 0.99` frequently
```
🚨 ATTENTION COLLAPSE: 60% of samples have P00>0.99 (one-hot)
```
**Fix**: Check QKV scaling (1/√head_dim), add attention dropout

### Loss Stagnation
**Symptom**: Loss flat for many batches
```
⚠️ LOSS FLAT: Std dev only 0.001 - model may not be learning
```
**Fix**: Increase LR, check data shuffling, verify tokenizer

---

## 📈 Monitoring Workflow

### During Training
```powershell
# Terminal 1: Training
.\build_vs_cuda\Release\train_gpu.exe

# Terminal 2: Live analysis
python training/analyze_training_log.py --watch
```

### Post-Training
```powershell
# 1. Run analyzer
python training/analyze_training_log.py

# 2. Check causality proofs (if concerned)
python training/run_causality_proofs.py --all

# 3. Deep gradient inspection (if issues found)
$env:GRIM_TRACE_GRADIENTS="1"
.\train_gpu.exe --max-steps 1
python training/trace_single_gradient.py gradient_trace_step_0.bin
```

---

## 📁 Log File Naming

| Pattern | Description |
|---------|-------------|
| `training_TIMESTAMP.log` | Raw training output |
| `analysis_TIMESTAMP.log` | Saved analysis report |
| `collector.log` | Metrics collection debug |
| `train_gpu_crash.log` | Crash dumps |
| `bad_sequences.log` | Rejected training sequences |

**Timestamp format**: `YYYYMMDDHHMMSS` (e.g., `17656777702397256`)

---

## 🔧 Configuration Impact

### ai_config.json Settings Affecting Logs

```json
{
  "training": {
    "config": {
      "log_interval": 10,          // Steps between log lines
      "grad_check_enabled": true,  // [GradCheck] lines
      "flash_debug": true          // [FlashBwd] lines
    }
  }
}
```

---

## 📋 Quick Diagnostic Checklist

- [ ] Health score ≥ 80?
- [ ] Loss decreasing trend?
- [ ] No GradGuard skips?
- [ ] Layer gradient ratio < 1000x?
- [ ] P00 one-hot % < 20%?
- [ ] dQ gradient stable (not vanishing)?
- [ ] No NaN/Inf warnings?
- [ ] Checkpoints saving regularly?

---

*Last updated: 2025-12-13*
