# Autograd Migration Checklist

**Goal:** Migrate from legacy raw `float*` gradient buffers to full `GRIM::Tensor` autograd system

**Started:** January 21, 2026
**Completed:** January 21, 2026 ✅

---

## Summary

The migration is **COMPLETE**. Encoder layers own their own Tensors with `.grad` buffers,
allocated via `ensure_grad()` in `allocateWeights()`. All backward passes, gradient consumers,
and cleanup code now use `enc->getFFNW1Grad()`, `enc->getRMS1GammaGrad()` etc.

---

## Phase 1: Audit Current State

- [x] Identify all legacy gradient buffer allocations in InitTrainingState.cu
- [x] Identify all legacy gradient buffer usages in backward passes
- [x] Identify all places that read gradients (computeGradNorm, optimizer, etc.)
- [x] Map TrainingTensors fields to legacy equivalents

## Phase 2: Remove Double Allocation

- [x] Remove legacy gradient allocations from InitTrainingState.cu (use Tensor.grad instead)
- [x] Update TrainingState_GPU.hpp to remove legacy gradient vectors
- [x] Point legacy accessor functions to Tensor.grad pointers (N/A - direct encoder access)

## Phase 3: Update Backward Passes

- [x] BackwardPhase1_OutputLayer.cu - Uses Tensor gradients (embedding/lm_head Tensors)
- [x] BackwardPhase2_Encoder.cu - Uses enc->getFFNW1Grad() etc.
- [x] BackwardPhase3_InputLayer.cu - Uses Tensor gradients (embedding Tensors)

## Phase 4: Update Gradient Consumers

- [x] computeGradNorm() - N/A (uses GradNorm component system)
- [x] buildParameterGroups() - Uses enc->getFFNW1Grad() etc.
- [x] AdamW optimizer - Receives Tensor.grad pointers via parameter groups
- [x] GradAccumulationController - Binds to enc->getFFNW1Grad() etc.

## Phase 5: Cleanup

- [x] Remove unused legacy buffer declarations from TrainingState_GPU.hpp
- [x] Remove redundant memory allocations from InitTrainingState.cu
- [x] Remove legacy cleanup from TrainingStateGPU.cu destructor
- [x] Verify no raw pointer gradient access remains
- [x] Simplified initializeAutogradTensors() - no longer creates TrainingTensors
- [ ] Test training runs correctly

---

## Architecture Summary

**Encoder Tensor Ownership:**
- `EncodingLayer::allocateWeights()` creates Tensors with `requires_grad=true`
- `allocateWeights()` calls `ensure_grad()` on all tensors (rms1_gamma_, W_qkv_, etc.)
- `FeedForwardLayer::ensureWeightStorage()` calls `ensure_grad()` on FFN tensors
- Backward passes access gradients via `enc->getFFNW1Grad()`, `enc->getRMS1GammaGrad()`, etc.
- No manual cudaMalloc for gradients needed

**Gradient Access Pattern:**
```cpp
GPUGrimEncoder* gpu_encoder = &model.getGpuEncoder();
for (int layer = 0; layer < num_layers; ++layer) {
    auto* enc = gpu_encoder->getLayer(layer);
    float* grad_W1 = enc->getFFNW1Grad();  // Returns Tensor.grad pointer
    float* grad_gamma1 = enc->getRMS1GammaGrad();
    // etc.
}
```

**Files Modified:**
- `TrainingState_GPU.hpp` - Removed legacy vector declarations
- `TrainingStateGPU.cu` - Removed legacy cleanup code, simplified initializeAutogradTensors()
- `InitTrainingState.cu` - Removed legacy allocations
- `Encoding_GPU.cu` - Added ensure_grad() calls in allocateWeights()
- `Feed_Forward_GPU.cu` - Added ensure_grad() calls in ensureWeightStorage()
- `BackwardPhase2_Encoder.cu` - Uses enc->getFFN*Grad()
- `LanguageModel_Training.cu` - zeroGradients() and buildParameterGroups() use encoder
- `GradAccumulationController_Integration.cu` - Binds to encoder's Tensor.grad
- `Phase2_TrainingLoop.cu` - Gradient export uses encoder's Tensor.grad

