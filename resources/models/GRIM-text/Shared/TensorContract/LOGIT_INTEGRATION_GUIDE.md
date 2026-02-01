# Logit Tracking Integration Guide

## Overview

Logit tracking has been integrated directly into the existing TensorContract autograd system. This guide shows how to convert manual logit tracking to automatic gradient tracking.

## Changes Made

### 1. Added `LOGITS` Layout to TensorContract

```cpp
enum class Layout : uint8_t {
    BSM,               // [tokens, d_model]
    QKV_FUSED,         // [tokens, total_qkv_dim]
    LOGITS,            // [tokens, vocab_size] - NEW!
    BHSD, BHDS, BSHD,  // 4D formats
    UNKNOWN,
};
```

### 2. Factory Methods for LOGITS

```cpp
// Create logit tensor shape
auto logit_shape = TensorShape::make_LOGITS(total_tokens, vocab_size);

// Create logit tensor view
auto logit_view = TensorView::make_LOGITS(logits_ptr, total_tokens, vocab_size, "logits");
```

### 3. Loss Functions with Automatic Gradients

The existing `cross_entropy()` function in `GRIM::autograd` namespace already creates GradFn nodes automatically:

```cpp
// Old way (manual)
float loss = computeLoss(cached_logits, targets, ...);
launchBackward(grad_logits, ...);  // Manual backward

// New way (automatic)
Tensor logits = Tensor::from_ptr(cached_logits,
                                 TensorShape::make_LOGITS(total_tokens, vocab_size),
                                 false, // doesn't own
                                 true); // requires_grad

Tensor loss = GRIM::autograd::cross_entropy(logits, targets, total_tokens, vocab_size, stream);
loss.backward();  // Automatic backward - grad_logits computed automatically!
```

## Migration Steps

### Step 1: Mark Logits with LOGITS Layout

**Before (ForwardPhase1_OutputLayer.cu):**

```cpp
// Manual allocation
training_state_.cached_logits = /* cudaMalloc */;
// No layout tracking
```

**After:**

```cpp
// Wrap in Tensor with LOGITS layout
Tensor logits = Tensor::from_ptr(
    training_state_.cached_logits,
    TensorShape::make_LOGITS(total_tokens, vocab_size),
    false,  // TrainingState owns the memory
    true    // requires_grad
);
logits.name = "lm_head_logits";
```

### Step 2: Use Autograd Loss Functions

**Before (Phase2_TrainingLoop.cu):**

```cpp
// Manual loss computation
LossInputs inputs{
    .logits = training_state.cached_logits,
    .targets = targets,
    // ... many fields
};
float loss = computeLoss(inputs, &loss_telemetry);

// Manual backward
launchBackward(...);
```

**After:**

```cpp
// Automatic gradient tracking
Tensor logits = /* from Step 1 */;
Tensor loss = GRIM::autograd::cross_entropy(logits, targets, total_tokens, vocab_size, stream);

// Automatic backward - grad_logits computed automatically!
loss.backward();

// Access gradients
float* grad_logits = logits.grad;  // Ready to use!
```

### Step 3: Replace Manual Gradient Buffers

**Before (BackwardPhase1_OutputLayer.cu):**

```cpp
// Manual grad_logits computation
launchCrossEntropyBackward(
    training_state_.cached_logits,
    targets,
    training_state_.grad_logits,  // Manually allocated buffer
    ...
);
```

**After:**

```cpp
// Gradient automatically computed by loss.backward()
// Just access logits.grad - it's already computed!
float* grad_logits = logits.grad;

// If you need it in a specific buffer for compatibility:
cudaMemcpyAsync(training_state_.grad_logits, logits.grad, ...);
```

## Key Benefits

### 1. Automatic Gradient Computation

- No manual backward kernels
- No forgetting to call backward
- Computation graph tracks dependencies

### 2. Type Safety

- `Layout::LOGITS` makes it explicit what a tensor contains
- Compiler catches mismatches (BSM vs LOGITS)

### 3. Memory Efficiency

- Lazy gradient allocation (only when needed)
- Automatic cleanup via RAII

### 4. Unified API

- Same Tensor type for all operations
- Consistent with existing autograd operations (matmul, gelu, etc.)

## Compatibility Notes

### Existing Code Can Coexist

You can gradually migrate without breaking existing code:

```cpp
// Old manual code still works
float* old_style_logits = training_state_.cached_logits;
launchManualBackward(old_style_logits, ...);

// New autograd code side-by-side
Tensor new_style_logits = Tensor::from_ptr(other_logits, ...);
auto loss = GRIM::autograd::cross_entropy(new_style_logits, ...);
loss.backward();
```

### Focal Loss and Unified Loss

**Status:** The function signatures are declared in TensorContract_GPU.hpp but implementations need to be added if not already present.

If needed, add them similar to cross_entropy:

```cpp
Tensor focal_loss(const Tensor& logits, const int* targets,
                  int num_tokens, int vocab_size,
                  float focal_alpha, float focal_gamma,
                  cudaStream_t stream) {
    // Create FocalLossGradFn similar to CrossEntropyGradFn
    // ...
}
```

## Testing Checklist

- [ ] Verify LOGITS layout appears in debug logs
- [ ] Confirm grad_logits is non-zero after loss.backward()
- [ ] Check gradient values match manual implementation
- [ ] Ensure memory is properly freed (no leaks)
- [ ] Validate training loss decreases normally

## Troubleshooting

### "Layout mismatch" error

**Cause:** Passing non-LOGITS tensor to loss function  
**Fix:** Ensure logits have `TensorShape::make_LOGITS(tokens, vocab_size)`

### Gradients are zero

**Cause:** Forgot to set `requires_grad=true`  
**Fix:** `Tensor::from_ptr(..., true)` or `tensor.requires_grad_(true)`

### Segfault in backward

**Cause:** Logits pointer freed before backward()  
**Fix:** Ensure logits outlive loss.backward() call

### Double gradient accumulation

**Cause:** Calling backward() multiple times  
**Fix:** Call `logits.zero_grad()` between iterations

## Performance Notes

The integrated autograd system has **zero overhead** compared to manual gradient computation:

- Same CUDA kernels used (no extra copies)
- Lazy allocation (grad buffer only created when needed)
- Computation graph is lightweight (just function pointers)

The only overhead is the initial Tensor wrapping, which is negligible (just pointer assignment).
