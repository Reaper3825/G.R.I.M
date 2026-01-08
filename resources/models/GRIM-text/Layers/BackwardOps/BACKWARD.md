# Backward Pass Refactoring - Design Document

## Overview

The backward pass is refactored into 3 distinct phases for clarity, debuggability, and maintainability.
This follows the same pattern as the training loop refactor (Phase1/2/3).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    backward() Entry Point                        │
│                  (BackwardOps_Orchestrator.cu)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Output Layer Backward                                  │
│  (BackwardPhase1_OutputLayer.cu)                                │
│                                                                  │
│  • Cross-entropy gradient computation (dL/d_logits)              │
│  • LM Head backward (grad_hidden, grad_W_lm_head)               │
│  • Output: grad_encoder_out ready for encoder backward          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Encoder Backward                                       │
│  (BackwardPhase2_Encoder.cu)                                    │
│                                                                  │
│  • Loop: layer N-1 → 0                                          │
│    - RMSNorm2 backward (FFN pre-norm)                           │
│    - FFN backward (W2, GELU, W1)                                │
│    - Residual connection gradient                                │
│    - RMSNorm1 backward (Attention pre-norm)                     │
│    - Flash Attention backward (GQA-aware)                       │
│    - QKV projection backward                                     │
│  • Output: grad ready for input layer backward                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Input Layer Backward                                   │
│  (BackwardPhase3_InputLayer.cu)                                 │
│                                                                  │
│  • ScratchBlock backward (if enabled)                           │
│  • Embedding backward (token + position)                        │
│  • Final gradient propagation complete                          │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
Layers/BackwardOps/
├── BACKWARD_REFACTOR.md          # This document
├── BackwardContext.hpp           # Shared context struct between phases
├── BackwardOps_Orchestrator.cu   # Main entry point, calls phases
├── BackwardOps_Orchestrator.hpp  # Public API
├── BackwardPhase1_OutputLayer.cu # Phase 1 implementation
├── BackwardPhase1_OutputLayer.hpp
├── BackwardPhase2_Encoder.cu     # Phase 2 implementation  
├── BackwardPhase2_Encoder.hpp
├── BackwardPhase3_InputLayer.cu  # Phase 3 implementation
├── BackwardPhase3_InputLayer.hpp
└── TODO.md                       # Implementation checklist
```

## BackwardContext Struct

The context struct carries all state between phases:

```cpp
struct BackwardContext {
    // Model configuration (immutable)
    const ModelConfig* config;
    TrainingState* training_state;
    
    // Current gradient flow (mutable)
    float* current_grad;           // Gradient flowing backward
    int total_tokens;              // batch_size * seq_len
    
    // Scaling factors
    float grad_scale;              // Token normalization scale
    bool accumulate;               // Accumulate or overwrite gradients
    
    // External components
    GPUGrimEncoder* gpu_encoder;
    ScratchBlockLayer* scratch_block;
    EmbeddingRuntime* embedding_runtime;
    
    // Telemetry
    uint64_t backward_call_id;     // For deterministic diagnostics
    cudaStream_t stream;
};
```

## Error Handling Contract

**FAIL LOUD**: All phases must detect and report errors immediately.

```cpp
// Pattern for all operations:
if (!operation_succeeded) {
    BWD_ERROR("[Phase" << phase << "] FATAL: " << operation_name << " failed");
    BWD_ERROR("  Location: " << __FILE__ << ":" << __LINE__);
    BWD_ERROR("  Context: layer=" << layer << " batch=" << batch_size);
    return BackwardStatus::FATAL_ERROR;
}
```

## TensorContract Usage

All tensor operations must use TensorContract for validation:

```cpp
// Before any tensor operation:
auto view = TensorContract::TensorView::make_BSM(
    buffer, total_tokens, d_model, "grad_encoder_out");
TensorContract::validate(view);  // Throws on invalid

// For zeroing:
TensorContract::zero(view, stream);

// For conversions:
TensorContract::convert(src_view, dst_view, stream);
```

## Telemetry Integration

Each phase reports metrics to TelemetryLattice:

- **Phase 1**: grad_logits RMS, LM head gradient norms
- **Phase 2**: Per-layer gradient norms (enables explosion detection)
- **Phase 3**: Embedding gradient statistics

## Return Values

```cpp
enum class BackwardStatus {
    SUCCESS = 0,
    FATAL_ERROR = 1,        // Unrecoverable, must stop training
    GRADIENT_EXPLOSION = 2, // Detected explosion, can skip step
    INVALID_STATE = 3,      // Training state not initialized
};
```

## Migration Path

1. ✅ Create new files in BackwardOps/
2. ✅ Implement phases with identical math to current BackwardOps.cu
3. ✅ Add TensorContract validation at every boundary
4. ✅ Add telemetry hooks (per-layer gradient RMS logging)
5. ✅ CMakeLists.txt updated to compile new files
6. ⬜ Update LanguageModel::backward() to call orchestrator
7. ⬜ Verify gradient values match old implementation
8. ⬜ Remove old monolithic backward code

## Gradient Flow Diagram

```
                    ┌──────────────┐
                    │   logits     │
                    └──────┬───────┘
                           │ dL/d_logits (cross-entropy)
                           ▼
                    ┌──────────────┐
                    │   LM Head    │ ← grad_W_lm_head, grad_b_lm_head
                    └──────┬───────┘
                           │ grad_encoder_out
                           ▼
              ┌────────────────────────────┐
              │      Encoder Layer N-1     │
              │  ┌──────────────────────┐  │
              │  │      RMSNorm2        │  │ ← grad_gamma2
              │  └──────────┬───────────┘  │
              │             ▼              │
              │  ┌──────────────────────┐  │
              │  │   FFN (W2, GELU, W1) │  │ ← grad_W1, grad_W2
              │  └──────────┬───────────┘  │
              │             │ + residual   │
              │             ▼              │
              │  ┌──────────────────────┐  │
              │  │      RMSNorm1        │  │ ← grad_gamma1
              │  └──────────┬───────────┘  │
              │             ▼              │
              │  ┌──────────────────────┐  │
              │  │  Flash Attention     │  │ ← grad_Q, grad_K, grad_V
              │  └──────────┬───────────┘  │
              │             ▼              │
              │  ┌──────────────────────┐  │
              │  │   QKV Projection     │  │ ← grad_W_qkv
              │  └──────────┬───────────┘  │
              │             │ + residual   │
              └─────────────┼──────────────┘
                           │
                           ▼
                    ... Layer 0 ...
                           │
                           ▼
              ┌────────────────────────────┐
              │       ScratchBlock         │ (if enabled)
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │        Embeddings          │ ← grad_embeddings
              └────────────────────────────┘
```

## Critical Invariants

1. **grad_Q/K/V must be zeroed before Flash Attention backward** (it accumulates)
2. **beta_param = accumulate ? 1.0f : 0.0f** for cuBLAS weight gradient accumulation
3. **GQA dimensions**: Q uses num_heads, K/V use num_kv_heads
4. **Residual connections**: grad flows through both paths (attention + skip, FFN + skip)
5. **RMSNorm gamma grads live in EncodingLayer**, not TrainingState

## References

- Original: `training/BackwardOps.cu` (2700+ lines)
- TensorContract: `Shared/TensorContract/TensorContract_GPU.hpp`
- Flash Attention: `Layers/FlashAttention/Flash_Attention_Kernal.hpp`
- RMSNorm: `Layers/LayernNorm/RMSNorm_Kernel_GPU.hpp`
- Telemetry: `Shared/Telemetry/TelemetryLattice_GPU.hpp`
