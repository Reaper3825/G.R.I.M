---
license: apache-2.0
language:
- en
pipeline_tag: grim-text-training
tags:
- grim-text
- training
- transformer
- gqa
- flash-attention
- telemetry
---

# GRIM-text — Model & Training Overview

This README documents the GRIM-text model and training stack (training-focused). It highlights the core architecture, the training loop structure, the runtime control patterns and the production optimizations used in this codebase.

This file is intentionally concise and focused on training and engineering decisions — for inference, deployment, and other tooling see other docs in the repository.

## Quick summary

- Model family: autoregressive Transformer with Grouped-Query Attention (GQA)
- Performance primitives: Flash Attention 2, fused kernels, cuBLAS GEMM
- Training: AdamW optimizer, mixed precision (FP16 compute, FP32 masters)
- Regularization: focal loss + label smoothing
- Stability and control: Telemetry control decision logic, GradAccumulationController, centralized resource controllers
- Tokenization & structure: ScratchBlock reasoning, AtomTable lookup, Aho-Corasick structural matching
- Memory & performance: pre-allocated reusable GPU buffers, GRIMTS guess cache, zero-copy FlatBuffers where applicable

## Key features (what to know quickly)

- Grouped Query Attention (GQA)
  - Reduces KV cache memory by grouping Q heads to fewer KV heads.
  - Implemented with explicit gradient scaling (heads_per_kv_group) in backward to preserve correct gradient magnitudes.

- Flash Attention 2
  - O(N) working memory attention path used for long sequences (auto-activated by seq_len threshold).
  - Fallback to tiled softmax/cuBLAS path for shorter contexts where appropriate.

- Training losses: Focal loss + Label smoothing
  - Focal loss (configurable gamma/alpha) to emphasize hard examples when enabled.
  - Label smoothing to improve calibration; implemented inside the UnifiedLoss pipeline for numerically stable gradients.

- Telemetry control & decision logic
  - TelemetryLattice collects hierarchical streaming metrics used by TelemetryControl to make runtime decisions (scale gradients, skip steps, adjust learning rate, or soft-restart).
  - Decisions are conservatively applied and logged via the centralized LogRecorder for reproducibility.

- AdamW optimizer (GPU kernel)
  - Decoupled weight decay with bias-correction; beta constants configurable.
  - GPU implementation respects host/kernel bias-correction constants (no mismatches).

- Mixed precision and master weights
  - FP16 compute with FP32 master weights for stable updates.
  - Automatic cast/uncast and fused update kernels where possible.

- ScratchBlock reasoning + AtomTable + Aho-Corasick
  - Structural token detection (URLs, emails, file paths, code atoms) performed via an Aho-Corasick automaton for O(n) pattern detection.
  - AtomTable maps structural atoms to dedicated atom token IDs (token id base offset), enabling structured reasoning in tokenization.

- Central control pattern (controllers)
  - StreamController, GradAccumulationController, Memory/BufferController, and LogRecorder centralize resource allocation and synchronization.
  - Controllers own allocations and lifecycles (no ad-hoc cudaMalloc/cudaFree outside controllers).

- GRIMTS guess cache
  - Lightweight GPU/host cache used to cache commonly observed token-sequence statistics and generation-time guesses to accelerate inference/training heuristics.

- Per-token normalized clipping & RMSNorm
  - Token-normalized gradient clipping supports stable scaling for variable-length batches.
  - RMSNorm (pre-norm style) implemented for the transformer layers for improved gradient flow.

- PBM: ALiBi ↔ RoPE Hybrid
  - Position Bias Module (PBM) combines ALiBi linear bias slopes with RoPE-style rotary position encoding in a hybrid configuration for extrapolation and stable positional signal.

- Fused kernels (GELU + Linear, LayerNorm, QKV) and cuBLAS
  - Compute-intensive paths fused into single kernels where practical; cuBLAS GEMM used for large matmuls and as a reference implementation when appropriate.

- Pre-allocate & reuse buffers
  - Training allocates buffers up-front (per-device) and reuses them across batches to avoid repeated alloc/free overhead and fragmentation.

## Training structure (three-phase orchestrator)

GRIM-text training is organized into three clear phases to make builds, tests and debugging straightforward:

1. Phase 1 — Startup
   - Load configuration and model metadata, initialize tokenizer and AtomTable.
   - Allocate and pre-warm all GPU buffers via the BufferController.
   - Initialize AdamW optimizer state, mixed-precision master weights, TelemetryLattice, and GRIMTS cache.

2. Phase 2 — Training loop
   - Data loading and packing (supports packing strategies including random and similarity-grouped).
   - Micro-batching and gradient accumulation managed by GradAccumulationController (supports accumulate=true behavior across micro-steps).
   - Forward pass: fused QKV projection, attention (Flash or cuBLAS path), FFN fused GELU, residual adds.
   - Loss: unified loss computes focal + smoothed CE in a single numerically-stable kernel; gradients produced and optionally scaled by Telemetry decision.
   - Backward pass: attention backward respects GQA scaling; gradients merged into centralized gradient buffers.
   - Optimizer step: AdamW GPU kernel updates master weights; mixed-precision casts applied.
   - Telemetry update: pre-clip gradients used to update TelemetryLattice and feed TelemetryControl decisions (skip/scale/soft-restart/adjust LR).

3. Phase 3 — Cleanup
   - Final checkpointing (FlatBuffers), release temporary buffers, and write training summary and telemetry artifacts.

## Design & implementation notes

- Central controllers
  - All GPU resources, cuBLAS handles, and streams are owned by centralized controllers exposed via the TrainingContext.
  - This prevents dangling allocations and ensures deterministic teardown and reproducible profiling.

- Gradient accumulation
  - GradAccumulationController ensures the correct use of beta (accumulate ? 1.0f : 0.0f) in GEMM operations so micro-batches add rather than overwrite gradients.

- Telemetry-driven safety
  - TelemetryControl uses hierarchical metrics (loss, pre-clip grad norm, update magnitude) to decide safe interventions.
  - Interventions are conservative (scale down gradient, skip single step, or request a soft-restart) and are fully logged by LogRecorder.

- Tokenization and ScratchBlock
  - The tokenizer uses a Unigram + byte-fallback model combined with a ScratchBlock reasoning layer.
  - ScratchBlock detects atoms (numbers, urls, paths, code literals) and emits atom placeholders (offset by ATOM_TOKEN_BASE) which are handled as dedicated tokens by the AtomTable.

- Attention & GQA
  - Forward: compute Q,K,V via fused projection then run Flash Attention when eligible.
  - Backward: KV gradients are accumulated across grouped heads with an explicit gqa_grad_scale to normalize contribution.

- Numerical stability
  - FP16 arithmetic is used for kernels with FP32 master weights and selective FP32 accumulators in GEMMs to reduce drift.
  - Unified loss and softmax kernels implement log-sum-exp and stabilized backward formulas.

## Configuration & important files

- Core config: `ai_config.json` — learning rate, warmup, focal/label-smoothing hyperparameters, packing strategy, GQA params
- Training orchestrator: `training/Phase1_Startup.cu`, `training/Phase2_TrainingLoop.cu`, `training/Phase3_Cleanup.cu`
- Tokenizer & atoms: `Shared/UnigramByte/` and `Shared/AtomTable/`
- Telemetry & control: `Shared/Telemetry/TelemetryLattice_GPU.cu`, `TelemetryControl_GPU.cu`, `LogRecorder` implementation
- Flash attention: `grim_flash_attention/` and `grim_training_kernels.cu`

## Quick developer workflow

1. Configure `ai_config.json` (model dims, GQA heads, LR schedule, focal/label smoothing).
2. Build training target:

```powershell
cd resources/models/GRIM-text/training
cmake --build build --config Release --target train_gpu
```

3. Run training (example):

```powershell
.\TrainingLoop\build\Release\train_gpu.exe --config ..\model_config.json
```

4. Inspect telemetry & logs: training writes telemetry and logs to the training `logs/` directory and updates `training_status.fb` (search the repo for this filename to find runtime artifacts).

## Testing & verification

- Unit/kernel tests exist under `training/test_*` (layernorm, fused kernels, backward checks). Use the test targets in the `training` CMake to run them.
- Checkpoint & serialization use FlatBuffers; run the `test_gpu_training_integration` target for a small end-to-end sanity check.

## Notes for contributors

- Follow the central-controller pattern: allocate GPU resources only via controllers.
- Avoid silent fallbacks; prefer fail-loud with clear error messages for invalid states.
- When adding instrumentation, write telemetry into the TelemetryLattice and add a conservative TelemetryControl policy rather than ad-hoc skips.

## Contact & references

- Primary maintainer: Austin (Reaper3825) — see repository issues for questions and PRs.

---
Last updated: 2025-12-29
---
license: apache-2.0
language:
- en
pipeline_tag: grim-text-training
tags:
- grim-text
- training
- transformer
- gqa
- flash-attention
- telemetry
---

# GRIM-text — Model & Training Overview

This README documents the GRIM-text model and training stack (training-focused). It highlights the core architecture, the training loop structure, the runtime control patterns, and the production optimizations used in this codebase.

This file is intentionally concise and focused on training and engineering decisions — for inference, deployment, and other tooling see other docs in the repository.

## Quick summary

- Model family: autoregressive Transformer with Grouped-Query Attention (GQA)
- Performance primitives: Flash Attention 2, fused kernels, cuBLAS GEMM
- Training: AdamW optimizer, mixed precision (FP16 compute, FP32 masters)
- Regularization: focal loss + label smoothing
- Stability and control: Telemetry control decision logic, GradAccumulationController, centralized resource controllers
- Tokenization & structure: ScratchBlock reasoning, AtomTable lookup, Aho-Corasick structural matching
- Memory & performance: pre-allocated reusable GPU buffers, GRIMTS guess cache, zero-copy FlatBuffers where applicable

## Key features (what to know quickly)

- Grouped Query Attention (GQA)
  - Reduces KV cache memory by grouping Q heads to fewer KV heads.
  - Implemented with explicit gradient scaling (heads_per_kv_group) in backward to preserve correct gradient magnitudes.

- Flash Attention 2
  - O(N) working memory attention path used for long sequences (auto-activated by seq_len threshold).
  - Fallback to tiled softmax/cuBLAS path for shorter contexts where appropriate.

- Training losses: Focal loss + Label smoothing
  - Focal loss (configurable gamma/alpha) to emphasize hard examples when enabled.
  - Label smoothing to improve calibration; implemented inside the UnifiedLoss pipeline for numerically stable gradients.

- Telemetry control & decision logic
  - TelemetryLattice collects hierarchical streaming metrics used by TelemetryControl to make runtime decisions (scale gradients, skip steps, adjust learning rate, or soft-restart).
  - Decisions are conservatively applied and logged via the centralized LogRecorder for reproducibility.

- AdamW optimizer (GPU kernel)
  - Decoupled weight decay with bias-correction; beta constants configurable.
  - GPU implementation respects host/kernel bias-correction constants (no mismatches).

- Mixed precision and master weights
  - FP16 compute with FP32 master weights for stable updates.
  - Automatic cast/uncast and fused update kernels where possible.

- ScratchBlock reasoning + AtomTable + Aho-Corasick
  - Structural token detection (URLs, emails, file paths, code atoms) performed via an Aho-Corasick automaton for O(n) pattern detection.
  - AtomTable maps structural atoms to dedicated atom token IDs (token id base offset), enabling structured reasoning in tokenization.

- Central control pattern (controllers)
  - StreamController, GradAccumulationController, Memory/BufferController, and LogRecorder centralize resource allocation and synchronization.
  - Controllers own allocations and lifecycles (no ad-hoc cudaMalloc/cudaFree outside controllers).

- GRIMTS guess cache
  - Lightweight GPU/host cache used to cache commonly observed token-sequence statistics and generation-time guesses to accelerate inference/training heuristics.

- Per-token normalized clipping & RMSNorm
  - Token-normalized gradient clipping supports stable scaling for variable-length batches.
  - RMSNorm (pre-norm style) implemented for the transformer layers for improved gradient flow.

- PBM: ALiBi ↔ RoPE Hybrid
  - Position Bias Module (PBM) combines ALiBi linear bias slopes with RoPE-style rotary position encoding in a hybrid configuration for extrapolation and stable positional signal.

- Fused kernels (GELU + Linear, LayerNorm, QKV) and cuBLAS
  - Compute-intensive paths fused into single kernels where practical; cuBLAS GEMM used for large matmuls and as a reference implementation when appropriate.

- Pre-allocate & reuse buffers
  - Training allocates buffers up-front (per-device) and reuses them across batches to avoid repeated alloc/free overhead and fragmentation.

## Training structure (three-phase orchestrator)

GRIM-text training is organized into three clear phases to make builds, tests and debugging straightforward:

1. Phase 1 — Startup
   - Load configuration and model metadata, initialize tokenizer and AtomTable.
   - Allocate and pre-warm all GPU buffers via the BufferController.
   - Initialize AdamW optimizer state, mixed-precision master weights, TelemetryLattice, and GRIMTS cache.

2. Phase 2 — Training loop
   - Data loading and packing (supports packing strategies including random and similarity-grouped).
   - Micro-batching and gradient accumulation managed by GradAccumulationController (supports accumulate=true behavior across micro-steps).
   - Forward pass: fused QKV projection, attention (Flash or cuBLAS path), FFN fused GELU, residual adds.
   - Loss: unified loss computes focal + smoothed CE in a single numerically-stable kernel; gradients produced and optionally scaled by Telemetry decision.
   - Backward pass: attention backward respects GQA scaling; gradients merged into centralized gradient buffers.
   - Optimizer step: AdamW GPU kernel updates master weights; mixed-precision casts applied.
   - Telemetry update: pre-clip gradients used to update TelemetryLattice and feed TelemetryControl decisions (skip/scale/soft-restart/adjust LR).

3. Phase 3 — Cleanup
   - Final checkpointing (FlatBuffers), release temporary buffers, and write training summary and telemetry artifacts.

## Design & implementation notes

- Central controllers
  - All GPU resources, cuBLAS handles, and streams are owned by centralized controllers exposed via the TrainingContext.
  - This prevents dangling allocations and ensures deterministic teardown and reproducible profiling.

- Gradient accumulation
  - GradAccumulationController ensures the correct use of beta (accumulate ? 1.0f : 0.0f) in GEMM operations so micro-batches add rather than overwrite gradients.

- Telemetry-driven safety
  - TelemetryControl uses hierarchical metrics (loss, pre-clip grad norm, update magnitude) to decide safe interventions.
  - Interventions are conservative (scale down gradient, skip single step, or request a soft-restart) and are fully logged by LogRecorder.

- Tokenization and ScratchBlock
  - The tokenizer uses a Unigram + byte-fallback model combined with a ScratchBlock reasoning layer.
  - ScratchBlock detects atoms (numbers, urls, paths, code literals) and emits atom placeholders (offset by ATOM_TOKEN_BASE) which are handled as dedicated tokens by the AtomTable.

- Attention & GQA
  - Forward: compute Q,K,V via fused projection then run Flash Attention when eligible.
  - Backward: KV gradients are accumulated across grouped heads with an explicit gqa_grad_scale to normalize contribution.

- Numerical stability
  - FP16 arithmetic is used for kernels with FP32 master weights and selective FP32 accumulators in GEMMs to reduce drift.
  - Unified loss and softmax kernels implement log-sum-exp and stabilized backward formulas.

## Configuration & important files

- Core config: `ai_config.json` — learning rate, warmup, focal/label-smoothing hyperparameters, packing strategy, GQA params
- Training orchestrator: `training/Phase1_Startup.cu`, `training/Phase2_TrainingLoop.cu`, `training/Phase3_Cleanup.cu`
- Tokenizer & atoms: `Shared/UnigramByte/` and `Shared/AtomTable/`
- Telemetry & control: `Shared/Telemetry/TelemetryLattice_GPU.cu`, `TelemetryControl_GPU.cu`, `LogRecorder` implementation
- Flash attention: `grim_flash_attention/` and `grim_training_kernels.cu`

## Quick developer workflow

1. Configure `ai_config.json` (model dims, GQA heads, LR schedule, focal/label smoothing).
2. Build training target:

```powershell
cd resources/models/GRIM-text/training
cmake --build build --config Release --target train_gpu
```

3. Run training (example):

```powershell
.\TrainingLoop\build\Release\train_gpu.exe --config ..\model_config.json
```

4. Inspect telemetry & logs: training writes telemetry and logs to the training `logs/` directory and updates `training_status.fb` (search the repo for this filename to find runtime artifacts).

## Testing & verification

- Unit/kernel tests exist under `training/test_*` (layernorm, fused kernels, backward checks). Use the test targets in the `training` CMake to run them.
- Checkpoint & serialization use FlatBuffers; run the `test_gpu_training_integration` target for a small end-to-end sanity check.

## Notes for contributors

- Follow the central-controller pattern: allocate GPU resources only via controllers.
- Avoid silent fallbacks; prefer fail-loud with clear error messages for invalid states.
- When adding instrumentation, write telemetry into the TelemetryLattice and add a conservative TelemetryControl policy rather than ad-hoc skips.

## Contact & references

- Primary maintainer: Austin (Reaper3825) — see repository issues for questions and PRs.

---
Last updated: 2025-12-29


---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- text-generation-inference
- transformer
- cuda
- gpu-accelerated
- embeddings
- flatbuffers
- zero-copy
- alibi
- layer-normalization
---

# GRIM Language Model

**GPU-accelerated transformer-based language model with production-grade FlatBuffers serialization**

GRIM (General Reasoning Intelligence Model) is a high-performance text generation system featuring full GPU acceleration, advanced positional encoding (ALiBi), and efficient zero-copy memory management.

## 🚀 Quick Start

### Inference (HTTP Server)

# NOTE: THIS IS THE ONLY PLACE OTHER THAN THE ACTUAL CODE FILES THEMSELVES with the place the collection logs are read and wrote to and from SEARCH FOR:======training_status.fb=======

```bash
# Build and run the Ollama-compatible server
cd training/build
cmake --build . --config Release --target grim_text_server
./Release/grim_text_server.exe
```

```bash
# Query the model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grim-text",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Embedding Management (FlatBuffers)

```cpp
#include "grim_embedding_serialization_fb.hpp"

// Save with Zstd compression
FlatBufferEmbeddingSerializer::save(
    "embeddings.grem",
    embedder,
    GRIMEmbedding::CompressionType_Zstd,
    3  // compression level
);

// Load with zero-copy memory mapping
FlatBufferEmbeddingSerializer::load("embeddings.grem", embedder);
```

### GPU-Accelerated Inference

```cpp
#include "grim_language_model.hpp"

GRIM::LanguageModel model("config.json");
model.initGPU();  // Enable CUDA acceleration

auto output = model.generate("Your prompt here", 100);
```

## ⚡ Performance Highlights

### Flash Attention 2
- **O(N) memory complexity** vs O(N²) for standard attention
- **48x48 block tiling** optimized for RTX 3080 Ti (164KB shared memory)
- **Automatic activation** at seq_len ≥ 512 tokens
- **Memory-efficient** long sequence processing (up to 2048+ tokens)
- **Compatible architectures**: sm_80, sm_86, sm_89 (Ampere/Ada)
- **Supports head dimensions**: 32 and 64

### GPU Acceleration
- **2.12x faster** inference (12,597 vs 6,478 tokens/sec @ 64 tokens)
- **16,562 tokens/sec** @ 512 tokens with Flash Attention
- **CUDA Tensor Cores** support (FP16 precision)
- **Multi-head attention** fully parallelized on GPU
- **Fused kernels** for GELU, LayerNorm, and FFN operations

### FlatBuffers Serialization
- **8.4x faster** loading (45ms vs 380ms uncompressed)
- **5.5x smaller** files (28MB vs 153MB with Zstd)
- **Zero memory overhead** (memory-mapped access)
- **Instant metadata** queries without loading weights

### Training Pipeline
- **Complete GPU backward pass** for all transformer components
- **AdamW optimizer** with GPU acceleration (decoupled weight decay)
- **Gradient clipping** and mixed precision support
- **100% CUDA-accelerated** training loop
- **Fused kernels** for reduced memory bandwidth:
  - Fused QKV projection (3 matrices in single kernel)
  - Fused FFN Layer 1 + GELU activation
  - Fused layer normalization (mean + variance + normalize)
- **AMSGrad variant** supported for adaptive learning rates

## 📚 Documentation

- **[INDEX.md](INDEX.md)** - Complete deliverables overview
- **[FLATBUFFER_SUMMARY.md](FLATBUFFER_SUMMARY.md)** - Executive summary
- **[FLATBUFFER_GUIDE.md](FLATBUFFER_GUIDE.md)** - Comprehensive usage guide
- **[FLATBUFFER_QUICKREF.cpp](FLATBUFFER_QUICKREF.cpp)** - Copy-paste ready code
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams
- **[FLATBUFFER_COMPARISON.md](FLATBUFFER_COMPARISON.md)** - Legacy vs FlatBuffers

## 🏗️ Architecture

### Transformer Components

**Multi-Head Attention**
- Configurable heads (default: 8)
- ALiBi positional bias for length extrapolation
- Causal masking for autoregressive generation
- Full GPU backward pass implementation

**Feed-Forward Network**
- GELU activation (GPU-fused kernel)
- Configurable hidden dimension (default: 4x model dim)
- Layer-wise dropout support

**Layer Normalization**
- Pre-normalization architecture
- Epsilon: 1e-6 for numerical stability
- GPU-accelerated forward and backward

**Embeddings**
- Token embeddings (configurable vocabulary)
- Positional encodings (learned or ALiBi)
- FP16 conversion for Tensor Cores
- Zero-copy loading via FlatBuffers

### GPU Implementation

**Forward Pass Kernels:**
- `flashAttentionForwardWithWorkspace` - Memory-efficient Flash Attention 2 (O(N) memory)
- `launchLayerNorm` - Fused layer normalization (mean + variance + normalize)
- `launchFusedQKVProjection` - Combined Q, K, V projection in single kernel
- `launchAttentionScores` - Scaled dot-product attention (standard path)
- `launchSoftmax` - Numerically stable softmax
- `launchAttentionOutput` - Attention @ Value
- `launchFFNLayer1Gelu` - Fused FFN first layer + GELU activation
- `launchFFNLayer2` - FFN second layer
- `launchResidualAdd` - Residual connections

**Backward Pass Kernels:**
- `launchLayerNormBackward` - LayerNorm gradients
- `launchAttentionOutputBackward` - Attention output gradients
- `launchSoftmaxBackward` - Softmax gradients
- `launchAttentionScoresBackward` - Q, K gradients
- `launchGeluBackward` - GELU activation gradients
- `launchFFNLayer2BackwardBias` - Bias gradients
- `launchAdamOptimizer` - AdamW weight updates (decoupled weight decay)
- `launchCrossEntropyLoss` - Loss computation
- `launchCrossEntropyGradient` - Loss gradients
- `launchEmbeddingBackward` - Embedding gradients
- `launchGradientClipping` - Gradient clipping

**Memory Management:**
- `GPUBuffer<T>` template class for device memory
- Automatic allocation/deallocation
- Stream-aware operations
- Memory-mapped FlatBuffers for zero-copy

### Build System

**CMake Configuration:**
- CUDA 12.5+ support
- Compute capabilities: 75, 80, 86, 89 (Turing, Ampere, Ada)
- Object libraries for modular compilation:
  - `grim_cuda_kernels` - CUDA kernels
  - `grim_language_model_gpu_impl` - GPU wrapper classes
  - `grim_embedding_gpu_impl` - GPU embedding templates
  - `grim_training_kernels` - Training-specific kernels
  - `grim_flash_attention` - Flash Attention 2 kernels (sm_80+)

**Dependencies:**
- CUDA Toolkit 12.5+
- cuBLAS (matrix operations)
- FlatBuffers (serialization)
- cpp-httplib (HTTP server)
- nlohmann/json (JSON parsing)
- Zstd (compression, optional)

## 🧪 Testing

### Test Suites

**`test_gpu_kernels`** (4/4 passing)
- Layer normalization validation
- Residual addition validation
- FFN Layer 1 with GELU validation
- FFN Layer 2 validation
- Max error: 2.38e-07 (float32 precision)

**`test_accuracy`** (5/5 passing)
- Forward pass validation
- Batch consistency validation
- Performance benchmark (CPU vs GPU)
- Numerical stability validation
- **Flash Attention validation** (256, 512, 1024 token sequences)
  - Standard attention: 15,876 tokens/sec @ 256
  - Flash Attention: 16,562 tokens/sec @ 512
  - Flash Attention: 14,146 tokens/sec @ 1024

**`test_gpu_backward`** (3/3 passing)
- LayerNorm backward validation
- GELU backward validation
- Adam optimizer validation
- Max error vs CPU: 1.27e-07

**`test_accuracy`** (4/4 passing)
- CPU vs GPU output comparison
- Performance benchmarking
- Numerical stability checks
- Embedding layer validation

**`test_gpu_training_integration`** (5/5 passing)
- FFN forward/backward integration
- Gradient computation validation
- Weight update verification
- End-to-end training step

### Benchmarks

**GPU: NVIDIA GeForce RTX 3080 Ti (12GB, Compute 8.6)**

| Sequence Length | CPU (tokens/sec) | GPU (tokens/sec) | Speedup |
|-----------------|------------------|------------------|---------|
| 8               | 839              | 2,836            | 3.38x   |
| 16              | 1,741            | 5,263            | 3.02x   |
| 32              | 3,612            | 7,849            | 2.17x   |
| 64              | 6,444            | 11,320           | 1.76x   |

**Model Configuration:** 4 layers, 256 dims, 8 heads, 1024 FFN dims
- **Adam optimizer** with GPU acceleration
- **Gradient clipping** and mixed precision support
- **100% CUDA-accelerated** training loop

## Model Details

### Model Description

**GRIM Language Model** is a transformer-based text generation system designed for high-performance inference and training on NVIDIA GPUs. The model features a complete CUDA-accelerated pipeline with advanced optimizations including ALiBi positional encoding, fused kernels, and FlatBuffers serialization.

**Key Features:**
- Full GPU acceleration for forward and backward passes
- **Flash Attention 2** for memory-efficient long sequence processing
- ALiBi positional bias for length extrapolation beyond training context
- Production-grade FlatBuffers serialization with Zstd compression
- Ollama-compatible HTTP API for easy integration
- Multi-head attention with causal masking
- Pre-normalization transformer architecture
- Mixed precision support (FP32/FP16)

- **Developed by:** Austin (Reaper3825)
- **Model type:** Transformer-based language model with Flash Attention 2
- **Language(s):** English
- **License:** Apache 2.0
- **Primary use:** Text generation, completion, and conversational AI
- **Architecture:** Encoder-only transformer with ALiBi and Flash Attention

### Model Sources

- **Repository:** [G.R.I.M](https://github.com/Reaper3825/G.R.I.M)
- **Training Code:** `training/` directory
- **Inference Server:** `grim_text_server.cpp`
- **GPU Kernels:** `grim_transformer_gpu.cu`, `grim_training_kernels.cu`

## 💡 Features

### Advanced Positional Encoding
- **ALiBi (Attention with Linear Biases)** for improved length extrapolation
- Allows inference on sequences longer than training length
- No learned positional embeddings required
- Configurable bias slopes per attention head

### GPU Optimization
- **Flash Attention 2:** Memory-efficient attention for long sequences
  - O(N) memory complexity vs standard O(N²)
  - 48x48 block tiling optimized for RTX 3080 Ti (164KB shared memory)
  - Automatic activation at seq_len ≥ 512 tokens
  - 16,562 tokens/sec @ 512 tokens (31% faster than standard attention)
  - Compatible architectures: sm_80, sm_86, sm_89 (Ampere/Ada GPUs)
- **Fused Kernels:** Combined operations reduce memory bandwidth
  - **Fused QKV Projection:** Single kernel for Q, K, V computation (3x reduction in memory operations)
  - **Fused FFN+GELU:** Combined linear layer and activation (40% bandwidth reduction)
  - **Fused LayerNorm:** Mean, variance, and normalization in one pass
- **cuBLAS Integration:** Optimized matrix multiplications
- **Tensor Core Support:** FP16 precision for 2x throughput
- **Memory Mapping:** Zero-copy weight loading via FlatBuffers
- **Stream Pipeline:** Asynchronous CUDA operations

### HTTP Server
- **Ollama-compatible API** (`/v1/chat/completions`)
- **Streaming responses** for real-time generation
- **Temperature and top-p sampling**
- **GPU toggle:** Switch between CPU/GPU at runtime
- **JSON configuration:** Easy model parameter tuning

### Serialization
- **FlatBuffers format:** Zero-copy deserialization
- **Zstd compression:** 5.5x smaller model files
- **Metadata queries:** Inspect model without loading weights
- **Version compatibility:** Schema evolution support
- **Memory efficiency:** Direct GPU upload from mapped memory

## Uses

### Direct Use

**Text Generation:**
```cpp
GRIM::LanguageModel model("config.json");
model.initGPU();  // Enable GPU acceleration

std::string prompt = "The future of AI is";
auto generated = model.generate(prompt, 100);  // Generate 100 tokens
```

**HTTP API:**
```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grim-text",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Embedding Extraction:**
```cpp
auto embeddings = model.getEmbeddings({"hello", "world"});
// Returns vector<Vector> with token embeddings
```

### Downstream Use

**Fine-tuning:**
- Pre-trained weights loadable via FlatBuffers
- Full backward pass for gradient-based optimization
- Adam optimizer with GPU acceleration
- Gradient clipping and learning rate scheduling

**Integration:**
- Ollama-compatible API for drop-in replacement
- JSON configuration for easy parameter tuning
- Modular architecture for custom components
- C++ API for embedding in larger applications

**Recommended Applications:**
- Conversational AI assistants
- Code completion and generation
- Text summarization and rewriting
- Question answering systems
- Creative writing assistance

### Out-of-Scope Use

**Not Recommended:**
- Mission-critical medical or legal decisions without human oversight
- Generating harmful, biased, or misleading content
- Real-time systems requiring deterministic responses
- Applications requiring perfect factual accuracy
- Languages other than English (not trained on multilingual data)

## Bias, Risks, and Limitations

**Technical Limitations:**
- Model size constrained by GPU memory (12GB tested)
- FP16 precision may cause numerical instability in edge cases
- ALiBi extrapolation degrades beyond 2-3x training length
- Single-language (English) only
- Requires CUDA-capable GPU for acceleration

**Model Biases:**
- Training data biases reflected in outputs
- May generate stereotypical or biased content
- Limited understanding of recent events (depends on training data cutoff)
- Potential for factual inaccuracies or hallucinations

**Safety Considerations:**
- No built-in content filtering or safety mechanisms
- Can generate harmful content if prompted
- Requires external moderation for production use
- May produce biased outputs for sensitive topics

### Recommendations

**For Users:**
- Review all generated content for accuracy and appropriateness
- Implement content filtering for production deployments
- Use temperature < 0.7 for more deterministic outputs
- Enable gradient clipping (max_norm=1.0) during fine-tuning
- Monitor GPU memory usage for large batch sizes


## How to Get Started with the Model

### Prerequisites

```bash
# Install dependencies (Windows with vcpkg)
vcpkg install cuda flatbuffers nlohmann-json cpp-httplib zstd

# Or build from source
git clone https://github.com/Reaper3825/G.R.I.M
cd G.R.I.M/resources/models/GRIM-text
```

### Building

```bash
cd training
mkdir build && cd build

# Configure with CUDA support
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ENABLED=ON \
         -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

# Build all targets
cmake --build . --config Release -j8

# Or build specific targets
cmake --build . --config Release --target grim_text_server
cmake --build . --config Release --target test_gpu_kernels
```

### Running Tests

```bash
cd Release

# Test GPU kernels
./test_gpu_kernels.exe

# Test backward pass
./test_gpu_backward.exe

# Test accuracy and performance
./test_accuracy.exe

# Test training integration
./test_gpu_training_integration.exe
```

### Starting the Server

```bash
# Start HTTP server (default port 8080)
./grim_text_server.exe

# Custom configuration
./grim_text_server.exe --config ../model_config.json --port 8080
```

### Example Usage

```cpp
#include "grim_language_model.hpp"
#include <iostream>

int main() {
    // Load model
    GRIM::LanguageModel model("config.json");
    
    // Enable GPU acceleration
    model.initGPU();
    std::cout << "GPU initialized" << std::endl;
    
    // Generate text
    std::string prompt = "Once upon a time";
    auto output = model.generate(prompt, 50, 0.8, 0.95);
    
    std::cout << "Generated: " << output << std::endl;
    
    return 0;
}
```

## Training Details

### Training Data

**Dataset:**
- Custom training corpus collected from web sources
- Data collected via `collect_data` utility
- Preprocessing pipeline for tokenization and cleaning
- Training data stored in `.grmt` format

**Data Collection:**
```bash
# Collect and preprocess training data
./collect_data.exe --url "https://example.com" --output training_data.grmt
```

**Vocabulary:**
- Configurable vocabulary size (default: 10,000 tokens)
- Byte-pair encoding (BPE) tokenization
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`

### Training Procedure

#### Preprocessing

**Tokenization:**
- BPE tokenizer with learned merge operations
- Subword units for out-of-vocabulary handling
- Configurable vocabulary size

**Data Augmentation:**
- Random sequence length sampling
- Dropout for regularization
- Optional data augmentation techniques

#### Training Hyperparameters

**Model Architecture:**
- **Layers:** 4-12 (configurable)
- **Model Dimension:** 256-768
- **FFN Dimension:** 1024-3072 (4x model dim)
- **Attention Heads:** 8-16
- **Max Sequence Length:** 128-512 tokens
- **Dropout:** 0.1

**Optimizer:** AdamW (Adam with decoupled weight decay)
- **Learning Rate:** 1e-4 to 5e-4 (with warmup)
- **Beta1:** 0.9
- **Beta2:** 0.999
- **Epsilon:** 1e-8
- **Weight Decay:** 0.01 (decoupled from gradient)
- **Gradient Clipping:** 1.0 (max norm)
- **AMSGrad:** Optional variant for better convergence

**Training:**
- **Batch Size:** 1+ (depends on GPU memory)
- **Training Steps:** 100,000+
- **Warmup Steps:** 4,000
- **Learning Rate Schedule:** Linear warmup + cosine decay
- **Mixed Precision:** FP16 with FP32 master weights
- **Gradient Accumulation:** Supported for larger effective batch sizes

**Hardware:**
- NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
- CUDA 12.5, cuBLAS
- **Training regime:** Mixed precision (FP16/FP32)

#### Speeds, Sizes, Times

**Training Performance:**
- **Throughput:** ~11,000 tokens/sec (seq_len=64, GPU)
- **Training Time:** ~24 hours for 100K steps (estimated)
- **Checkpoint Size:** 28MB (compressed), 153MB (uncompressed)
- **Memory Usage:** 8-10GB GPU memory (batch_size=64)

**Inference Performance:**
- **Latency:** 2.8ms per forward pass (seq_len=8)
- **Throughput:** 11,320 tokens/sec (seq_len=64)
- **GPU Utilization:** 70-85% during inference

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

**Validation Set:**
- Held-out portion of training corpus
- Multiple domains for robustness testing
- Perplexity evaluation on diverse text

**Test Suites:**
1. **GPU Kernel Validation** - Numerical accuracy tests
2. **Backward Pass Validation** - Gradient correctness
3. **Performance Benchmarking** - Speed and efficiency
4. **Numerical Stability** - Edge case handling

#### Factors

**Evaluation disaggregated by:**
- Sequence length (8, 16, 32, 64 tokens)
- Batch size (1, 8, 32, 64)
- Precision (FP32 vs FP16)
- Hardware (CPU vs GPU)
- Model size (layers, dimensions)

#### Metrics

**Accuracy Metrics:**
- **Perplexity:** Model uncertainty (lower is better)
- **Numerical Error:** Max difference vs reference implementation
- **Gradient Correctness:** Backward pass validation against CPU

**Performance Metrics:**
- **Throughput:** Tokens per second
- **Latency:** Milliseconds per forward pass
- **Memory Usage:** Peak GPU memory consumption
- **Speedup:** GPU vs CPU performance ratio

**Stability Metrics:**
- **Numerical Stability:** No NaN/Inf in outputs
- **Gradient Norms:** Within expected bounds
- **Convergence:** Training loss reduction over time

### Results

**Test Results (All Passing):**

| Test Suite                      | Tests | Pass | Max Error    |
|---------------------------------|-------|------|--------------|
| test_gpu_kernels                | 4     | 4    | 2.38e-07     |
| test_gpu_backward               | 3     | 3    | 1.27e-07     |
| test_accuracy                   | 4     | 4    | N/A          |
| test_gpu_training_integration   | 5     | 5    | N/A          |

**Performance Results:**

| Configuration        | CPU Speed    | GPU Speed    | Speedup |
|---------------------|--------------|--------------|---------|
| 4L-256D-8H, seq=8   | 839 tok/s    | 2,836 tok/s  | 3.38x   |
| 4L-256D-8H, seq=16  | 1,741 tok/s  | 5,263 tok/s  | 3.02x   |
| 4L-256D-8H, seq=32  | 3,612 tok/s  | 7,849 tok/s  | 2.17x   |
| 4L-256D-8H, seq=64  | 6,444 tok/s  | 11,320 tok/s | 1.76x   |

**Numerical Accuracy:**
- GPU vs CPU max difference: **2.38e-07** (well within FP32 tolerance)
- Backward pass gradient error: **1.27e-07**
- All tests show numerical stability across 1000+ random sequences

#### Summary

The GRIM Language Model demonstrates:
- ✅ **Excellent numerical accuracy** (<1e-6 error)
- ✅ **Strong GPU acceleration** (1.76x - 3.38x speedup)
- ✅ **Numerical stability** (no NaN/Inf in extensive testing)
- ✅ **Correct gradient computation** (validated backward pass)
- ✅ **Production-ready performance** (11,000+ tokens/sec)

All 16 test cases pass, validating correctness of both forward and backward passes across CPU and GPU implementations.



## Model Examination

### GPU Implementation Analysis

**Kernel Efficiency:**
- **Fused QKV kernel** eliminates 66% of memory operations (3 separate kernels → 1)
- **Fused GELU kernel** reduces memory bandwidth by 40% (no intermediate storage)
- **Fused LayerNorm** computes mean, variance, normalize in single pass
- Combined QKV projection eliminates redundant memory operations
- Softmax kernel uses numerically stable log-sum-exp
- Attention backward implemented with minimal atomic operations

**Memory Patterns:**
- Coalesced memory access for optimal bandwidth
- Shared memory usage in softmax and layer norm kernels
- Zero-copy FlatBuffers loading reduces startup time
- Stream-aware operations enable async execution

**Optimization Techniques:**
- `--use_fast_math` for transcendental functions
- Tensor Core utilization with FP16 (up to 2x speedup)
- cuBLAS for optimized matrix multiplications
- Register allocation tuned for compute capability 8.6

### Architecture Decisions

**ALiBi vs Learned Positional Encodings:**
- ALiBi chosen for length extrapolation capability
- Enables inference on sequences 2-3x longer than training
- Reduces parameter count (no learned position embeddings)
- Per-head bias slopes provide inductive bias

**Pre-Normalization:**
- Layer norm applied before attention and FFN (not after)
- Improves gradient flow during training
- Better numerical stability with deep models
- Standard in modern transformer architectures

**FlatBuffers over Pickle/JSON:**
- Zero-copy deserialization (no parsing overhead)
- 8.4x faster loading with memory mapping
- Zstd compression reduces file size by 5.5x
- Schema evolution for backward compatibility

## Environmental Impact

**Hardware:** NVIDIA GeForce RTX 3080 Ti
- **TDP:** 350W
- **Training Power:** ~350W during active training
- **Idle Power:** ~50W during model loading

**Estimated Training Impact:**
- **Hours used:** ~24 hours (100K steps)
- **Total Energy:** ~8.4 kWh
- **Carbon Intensity:** Varies by region (0.3-0.9 kg CO2/kWh)
- **Estimated CO2:** ~2.5-7.5 kg CO2eq (depends on electricity source)

**Inference Impact:**
- **Power per query:** ~0.001 kWh (3 seconds @ 350W)
- **Daily usage (1000 queries):** ~1 kWh
- **GPU utilization:** 70-85% (efficient resource usage)

**Efficiency Improvements:**
- FP16 precision reduces energy consumption vs FP32
- Fused kernels minimize memory traffic (lower power)
- FlatBuffers loading reduces startup energy cost
- GPU acceleration provides better performance-per-watt than CPU

**Note:** Actual carbon emissions depend on the electricity grid's carbon intensity. Training on renewable energy sources can reduce environmental impact to near-zero.

## Technical Specifications

### Model Architecture and Objective

**Architecture:** Transformer encoder with the following components:

**Embedding Layer:**
- Token embeddings: Learned lookup table (vocab_size × d_model)
- Positional encoding: ALiBi (no learned parameters)
- Layer normalization before feeding to encoder

**Encoder Layers (Repeated N times):**
```
Input → LayerNorm → Multi-Head Attention → Residual
      → LayerNorm → Feed-Forward Network → Residual → Output
```

**Multi-Head Attention:**
- Scaled dot-product attention: `softmax(Q·K^T / √d_k) · V`
- ALiBi bias: Linear penalties based on distance
- Causal masking for autoregressive generation
- Q, K, V projections: `d_model → num_heads × d_head`
- Output projection: `num_heads × d_head → d_model`

**Feed-Forward Network:**
```
FFN(x) = W2 · GELU(W1·x + b1) + b2
```
- Hidden dimension: `d_ff = 4 × d_model`
- GELU activation (smooth approximation of ReLU)
- Dropout applied after each sub-layer

**Output Layer:**
- Linear projection to vocabulary: `d_model → vocab_size`
- Softmax for probability distribution (or log-softmax for training)

**Training Objective:**
- Cross-entropy loss for next-token prediction
- Language modeling: `P(w_t | w_{<t})`
- Autoregressive generation with teacher forcing

**Default Configuration:**
```json
{
  "vocab_size": 10000,
  "d_model": 256,
  "d_ff": 1024,
  "num_layers": 4,
  "num_heads": 8,
  "max_seq_len": 128,
  "dropout": 0.1,
  "alibi": true
}
```

### Compute Infrastructure

#### Hardware

**Development/Training:**
- **GPU:** NVIDIA GeForce RTX 3080 Ti
  - **Memory:** 12 GB GDDR6X
  - **CUDA Cores:** 10,240
  - **Tensor Cores:** 320 (3rd gen)
  - **Compute Capability:** 8.6
  - **Memory Bandwidth:** 912 GB/s
  
- **CPU:** AMD Ryzen 9 / Intel Core i9 (for fallback)
- **RAM:** 32 GB+ DDR4
- **Storage:** NVMe SSD (for fast model loading)

**Minimum Requirements:**
- CUDA-capable GPU with 6GB+ VRAM (Compute Capability 7.5+)
- CUDA Toolkit 12.0+
- 16 GB system RAM
- C++17 compatible compiler

**Supported GPUs:**
- **Turing:** RTX 2070, 2080, 2080 Ti (Compute 7.5)
- **Ampere:** RTX 3060, 3070, 3080, 3090, A100 (Compute 8.0, 8.6)
- **Ada Lovelace:** RTX 4070, 4080, 4090 (Compute 8.9)

#### Software

**Core Dependencies:**
- **CUDA:** 12.5.82 (NVIDIA GPU Computing Toolkit)
- **cuBLAS:** Included with CUDA (matrix operations)
- **C++ Compiler:** 
  - MSVC 14.44+ (Visual Studio 2022) on Windows
  - GCC 11+ or Clang 14+ on Linux
- **CMake:** 3.22+ (build system)

**Libraries:**
- **FlatBuffers:** 23.5.26+ (serialization)
- **nlohmann/json:** 3.11.2+ (JSON parsing)
- **cpp-httplib:** 0.14.0+ (HTTP server)
- **Zstd:** 1.5.5+ (compression, optional)

**Build Tools:**
- **vcpkg:** Package manager (Windows)
- **ninja or make:** Build executor
- **git:** Version control

**Runtime:**
- **Windows:** 10/11 with latest GPU drivers
- **Linux:** Ubuntu 22.04+ or equivalent
- **NVIDIA Driver:** 525+ (for CUDA 12.5)

**Development Environment:**
- **IDE:** Visual Studio 2022, VS Code, CLion
- **Debugging:** CUDA-GDB, Nsight Compute, Nsight Systems
- **Profiling:** nvprof, Nsight Compute for kernel analysis

## Citation

**BibTeX:**

```bibtex
@software{grim_language_model_2025,
  author = {Austin (Reaper3825)},
  title = {GRIM Language Model: GPU-Accelerated Transformer with ALiBi},
  year = {2025},
  url = {https://github.com/Reaper3825/G.R.I.M},
  note = {Transformer-based language model with full CUDA acceleration and FlatBuffers serialization}
}
```

**APA:**

Austin (Reaper3825). (2025). *GRIM Language Model: GPU-Accelerated Transformer with ALiBi* [Computer software]. https://github.com/Reaper3825/G.R.I.M

## Glossary

**ALiBi (Attention with Linear Biases):** Positional encoding method that adds linear biases to attention scores based on token distance, enabling length extrapolation beyond training context.

**Autoregressive:** Generation strategy where each token is predicted based on all previous tokens, enabling coherent text generation.

**cuBLAS:** NVIDIA's optimized BLAS (Basic Linear Algebra Subprograms) library for GPU matrix operations.

**FlatBuffers:** Zero-copy serialization format allowing direct memory mapping without parsing overhead.

**GELU (Gaussian Error Linear Unit):** Smooth activation function: `x · Φ(x)` where Φ is the Gaussian CDF.

**Tensor Cores:** Specialized hardware units on NVIDIA GPUs for accelerated mixed-precision matrix operations.

**AdamW (Adam with Decoupled Weight Decay):** Optimizer variant that separates weight decay from gradient-based updates, often leading to better generalization than standard Adam.

**Zero-copy:** Data access pattern where buffers are memory-mapped directly without copying, reducing load time and memory usage.

**Fused Kernel:** CUDA kernel that combines multiple operations into a single GPU launch, reducing memory bandwidth usage and improving performance.

**Causal Masking:** Attention mask preventing tokens from attending to future positions, necessary for autoregressive generation.

**Pre-normalization:** Architecture where layer normalization is applied before (not after) attention and FFN sublayers.

**Mixed Precision:** Training technique using FP16 for computation and FP32 for weight updates, balancing speed and numerical stability.

## More Information

### File Structure
```
GRIM-text/
├── grim_language_model.hpp/cpp      # Main model class
├── grim_transformer_gpu.hpp/cu      # GPU-accelerated transformer
├── grim_embedding_gpu.hpp/cu        # GPU embedding layer
├── grim_training_kernels.cu         # Training-specific CUDA kernels
├── grim_text_embedding.cpp          # CPU embedding implementation
├── grim_encoder_layer.hpp           # CPU encoder layer
├── training/
│   ├── grim_text_server.cpp        # HTTP inference server
│   ├── test_gpu_kernels.cu         # Forward pass tests
│   ├── test_gpu_backward.cu        # Backward pass tests
│   ├── test_accuracy.cu            # Accuracy benchmarks
│   ├── test_gpu_training_integration.cu  # Training tests
│   └── CMakeLists.txt              # Build configuration
└── README.md                        # This file
```

### Related Projects
- **G.R.I.M Main System:** Full AI assistant with voice, vision, and perception
- **GRIM-text:** Language model component (this repository)
- **OSINT Tools:** Data collection and preprocessing utilities

### Future Work
- **Flash Attention integration** for longer contexts (currently limited to 512 tokens)
- Multi-language support (non-English corpora)
- Larger model variants (1B+ parameters)
- Quantization (INT8, INT4) for deployment
- Distributed training across multiple GPUs
- Model distillation for edge deployment
- **Flash Attention 2/3** for 4-8x memory reduction and 2-4x speedup

### Support
- **Issues:** https://github.com/Reaper3825/G.R.I.M/issues
- **Discussions:** GitHub Discussions
- **Documentation:** See `docs/` directory

## Model Card Authors

**Austin (Reaper3825)**
- GitHub: [@Reaper3825](https://github.com/Reaper3825)
- Primary developer and maintainer

## Model Card Contact

For questions, issues, or collaboration:
- **GitHub:** https://github.com/Reaper3825/G.R.I.M
- **Issues:** https://github.com/Reaper3825/G.R.I.M/issues

---

**Last Updated:** November 6, 2025  
**Model Version:** 1.0  
**API Version:** v1 (Ollama-compatible)