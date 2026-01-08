# Single Gradient Trace Setup

## Quick Start

### 1. Enable Gradient Tracing

Set the environment variable before running training:

```powershell
$env:GRIM_TRACE_GRADIENTS="1"
.\build_vs_cuda\Release\grim_language_model_gpu.exe --config training_config.json
```

Or on Linux/Mac:
```bash
GRIM_TRACE_GRADIENTS=1 ./build_vs_cuda/Release/grim_language_model_gpu --config training_config.json
```

### 2. Training will dump gradients on the first step

Output file: `gradient_trace_step_N.bin` (where N is the step number)

### 3. Analyze the gradient dump

```bash
python trace_single_gradient.py gradient_trace_step_0.bin
```

Or save detailed report:
```bash
python trace_single_gradient.py gradient_trace_step_0.bin --output gradient_report.json
```

## What You'll See

The trace will show:

1. **Total gradient norm** - Overall L2 norm across all parameters
2. **Gradient flow issues** - Detects vanishing/exploding gradients
3. **Top 10 gradients by norm** - Which layers have the largest updates
4. **Per-tensor statistics** - Mean, std, min, max, sparsity for each layer

### Example Output

```
================================================================================
GRADIENT TRACE SUMMARY
================================================================================
Total gradient norm: 123.456789
Number of tensors: 52

✓ Gradient flow looks healthy

--------------------------------------------------------------------------------
TOP 10 GRADIENTS BY NORM:
--------------------------------------------------------------------------------
 1. layer_5_attn_qkv_weight_grad            norm=   45.123456 shape=[768, 256]
 2. lm_head_weight_grad                     norm=   38.234567 shape=[21544, 256]
 3. layer_4_ffn_w1_grad                     norm=   23.456789 shape=[1024, 256]
 4. embedding_grad                          norm=   18.567890 shape=[21544, 256]
 5. layer_3_attn_qkv_weight_grad            norm=   16.789012 shape=[768, 256]
...

--------------------------------------------------------------------------------
PER-TENSOR STATISTICS:
--------------------------------------------------------------------------------
Name                                             Norm         Mean          Std          Min          Max  Sparsity%
--------------------------------------------------------------------------------
embedding_grad                               18.567890    -2.34e-05     3.21e-04    -1.23e-03     2.45e-03      15.23%
lm_head_weight_grad                          38.234567     1.45e-04     4.56e-04    -2.34e-03     3.21e-03       8.45%
layer_0_attn_qkv_weight_grad                 12.345678    -3.21e-05     2.78e-04    -1.56e-03     1.89e-03      18.92%
...
```

## Use Cases

### Debugging Vanishing Gradients

If early layers show very small norms (< 1e-6), gradients are vanishing:

```
⚠️  GRADIENT FLOW ISSUES:
  ⚠️  Vanishing gradients: Layer 0 norm 3.21e-08, Layer 5 norm 4.56e-02
```

### Detecting Exploding Gradients

If any gradient norm is very large (> 1000), you have explosion:

```
 1. layer_5_ffn_w2_grad                     norm= 45678.123456 shape=[256, 1024]
```

### Checking Gradient Sparsity

High sparsity (> 50%) means most gradients are near zero - might need:
- Lower dropout
- Higher learning rate
- More regularization

### Finding Dead Layers

If a layer shows `ALL INVALID` or norm < 1e-10, that layer isn't learning.

## Implementation Details

### Binary Format

The gradient dump uses a simple binary format:

```
Header:
  magic: 0x47524144 ('GRAD')
  version: 1
  num_tensors: N

For each tensor:
  name_len: uint32
  name: char[name_len]
  ndim: uint32
  shape: uint32[ndim]
  dtype: uint32 (0=float32, 1=float16)
  data: float[prod(shape)]
```

### Memory Overhead

Each gradient dump requires copying all gradients from GPU → CPU.
For a typical model:
- 6-layer, d_model=256, d_ff=1024, vocab=21544
- Total parameters: ~12M
- Gradient dump size: ~48MB per trace

**Note:** Only dumps gradients on the FIRST backward pass, then disables itself.

## Proving Correctness

### Run Analytical Validation Tests

```bash
python validate_gradient_trace.py --analytical
```

This runs three analytical tests that prove gradient math is correct:

1. **Softmax + Cross Entropy** - Validates loss gradient: `∂L/∂z = p - δ(target)`
2. **Matrix Multiplication** - Validates linear layer: `∂L/∂W = X^T @ ∂L/∂Y`
3. **GELU Activation** - Validates activation gradient: `∂GELU/∂x = Φ(x) + x·φ(x)`

Each test compares **analytical gradient** (backprop) vs **numerical gradient** (finite differences).

Expected output:
```
================================================================================
ANALYTICAL VALIDATION: Softmax + Cross Entropy
================================================================================
Logits:              [2.  1.  0.1]
Target:              0
Loss:                0.664760

Analytical gradient: [ 0.5761  -0.4239  -0.1522]
Numerical gradient:  [ 0.5761  -0.4239  -0.1522]
Relative error:      [1.23e-06 2.45e-07 8.91e-07]
Max rel error:       1.23e-06

✓ PASS: Analytical gradient matches numerical gradient
```

If all three tests pass, your gradient implementation is mathematically correct.

### Validate Gradient Dump

```bash
python validate_gradient_trace.py gradient_trace_step_0.bin
```

This runs 5 sanity checks:
1. ✓ No NaN/Inf values
2. ✓ No zero gradients (dead neurons)
3. ✓ No exploding gradients (> 10,000 norm)
4. ✓ Gradient flow through layers (no vanishing)
5. ✓ Embedding vs LM head balance

Expected output:
```
================================================================================
SANITY CHECKS: Gradient Dump Validation
================================================================================

[1/5] Checking for NaN/Inf values...
  ✓ All gradients are finite

[2/5] Checking for zero gradients...
  ✓ No zero gradients detected

[3/5] Checking for exploding gradients...
  ✓ No exploding gradients detected

[4/5] Checking gradient flow through layers...
  Layer gradient norms:
    Layer 0: 12.345678
    Layer 1: 15.234567
    Layer 2: 18.123456
  ✓ Gradient flow looks healthy (ratio: 0.68)

[5/5] Checking embedding vs LM head gradients...
  Embedding grad norm: 18.567890
  LM head grad norm:   38.234567
  Ratio:               0.4856
  ✓ Embedding and LM head gradients are balanced

================================================================================
VALIDATION SUMMARY
================================================================================
✓ ALL CHECKS PASSED - Gradients look healthy!
```

### Full Validation (Recommended)

```bash
# Run both analytical tests and gradient dump validation
python validate_gradient_trace.py gradient_trace_step_0.bin --analytical
```

If everything passes, you have **mathematical proof** that:
- Your backprop implementation matches finite differences
- Gradients are numerically stable
- Signal flows properly through the network

## Troubleshooting

### "Training state not initialized"

Make sure training has started before the trace triggers. The trace happens after the first `backward()` call.

### File not found

The gradient dump is saved in the current working directory where you run the training executable. Check:

```powershell
Get-ChildItem gradient_trace_*.bin
```

### Large file sizes

Gradient dumps can be large (50-500MB). If disk space is limited, analyze the trace immediately and delete the .bin file.

## Advanced Usage

### Trace Specific Step

Edit [train_gpu.cu:2458](train_gpu.cu#L2458) to change the trace condition:

```cpp
// Trace step 100 instead of first step
if (!trace_done && step == 100) {
```

### Continuous Tracing

To dump gradients every N steps (WARNING: large disk usage):

```cpp
// Dump every 10 steps
if (step % 10 == 0) {
    std::string dump_path = "gradient_trace_step_" + std::to_string(step) + ".bin";
    model->dumpGradients(dump_path);
}
```

### Compare Gradients Across Steps

```bash
python trace_single_gradient.py gradient_trace_step_0.bin -o step_0.json
python trace_single_gradient.py gradient_trace_step_100.bin -o step_100.json
python compare_gradient_traces.py step_0.json step_100.json
```

(You'll need to write `compare_gradient_traces.py` - it's a simple JSON diff)
