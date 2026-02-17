# Layer Ownership Cleanup Plan

## GRIM-text Architectural Standard: Layer Self-Management

GRIM-text uses a **broadcast tape-based autograd system** with global coordination points:
- **Autograd tape**: Tensors register via `tensor.ensure_grad()`
- **Optimizer**: `buildParameterGroups()` collects learnable parameters via layer accessors
- **Gradient controller**: `zeroGrad()` resets before accumulation window

---

## The Correct Pattern: Layer Self-Management

**Each layer is a self-contained module that:**

1. **Creates its own Tensor objects** in constructor or `allocateWeights()`:
   ```cpp
   class MyLayer {
       Tensor weight_;  // Member variable IS a Tensor object
       
       void allocateWeights() {
           weight_ = Tensor::zeros([out_dim, in_dim], true, stream, "my_weight");
           weight_.ensure_grad();  // Register with autograd tape
       }
   };
   ```

2. **Owns the GPU memory** (`Tensor::owns_data = true`, proper destructor cleanup)

3. **Registers with globals** through public accessors:
   ```cpp
   // In buildParameterGroups():
   registerTensor(layer->getWeight(), ParamGroupType::MY_LAYER);
   ```

4. **Handles its own initialization** (Xavier, zeros, custom logic):
   ```cpp
   void initializeWeights() {
       launchXavierInit(weight_.data, weight_.shape, seed, stream);
   }
   ```

**Why this scales:**
- Adding a new layer = write the layer class only, zero changes to external files
- No god object knowing every layer's internal structure
- True encapsulation — layer internals are private
- Decoupled modules communicating through clean interfaces

---

## What Layers DON'T Own: Forward Outputs

**Critical distinction:** Layers own their **weights** (learnable parameters), but **NOT their forward outputs** (activations/intermediate results).

**In the tape-based autograd system:**
- **Layers don't store outputs** — `forward()` returns a Tensor but doesn't keep a reference
- **The computation graph (autograd tape) holds references** to all intermediate Tensors via `GradFn` nodes
- **Context variables keep them alive until backward** — caller stores returned Tensors until `backward()` completes

**Example:**
```cpp
// Layer creates output but doesn't store it
Tensor MyLayer::forward(const Tensor& input) {
    Tensor output = autograd::matmul(input, weight_);  // Creates new Tensor
    return output;  // Layer doesn't keep reference
}

// Caller (AutogradTraining) stores intermediate
ctx.layer_output = my_layer->forward(ctx.input);  // Kept alive in context

// Autograd tape keeps reference through GradFn chain
// When backward() runs, tape traverses: loss → logits → layer_output → input

// After backward completes:
ctx.clearIntermediates();  // Tensors destruct, GPU memory freed
```

**Why this matters:**
- Layers are **stateless** across forward passes (except for weights)
- Forward outputs are **ephemeral** — created during forward, destroyed after backward
- Only **weights and gradients** persist across batches (managed by optimizer)
- This allows layers to be called multiple times per batch (e.g., for each token in generation)

---

## Gradient Ownership: Tensors Own Their Gradients

**Rule:** The **Tensor object owns its gradient buffer** (the `grad_data` pointer). This applies to BOTH weights and intermediates.

### Weight Gradients (Persistent)

```cpp
class MyLayer {
    Tensor weight_;  // Layer owns the weight Tensor
    
    void allocateWeights() {
        weight_ = Tensor::zeros([out, in], true, stream, "weight");
        weight_.ensure_grad();  // Allocates weight_.grad_data (owned by weight_ Tensor)
    }
};

// During backward:
// MatMulGradFn::apply() writes to weight_.grad_data
// accumulates across gradient accumulation window

// At optimizer step:
// buildParameterGroups() registered &weight_, so optimizer accesses weight_.grad_data
// AdamW updates weight_.data using weight_.grad_data

// At start of next window:
// zeroGrad() calls cudaMemsetAsync(weight_.grad_data, 0, ...)
```

**Lifetime:** Allocated by `ensure_grad()` → persists across batches → freed when Tensor destructs

### Intermediate Gradients (Ephemeral)

```cpp
// In AutogradTraining.cu forward:
ctx.lm_logits_tensor = lm_head->forward(encoder_output);  // Creates new Tensor
ctx.lm_logits_tensor.ensure_grad();  // Allocates grad_data for logits

// During backward:
// CrossEntropyGradFn::apply() writes to ctx.lm_logits_tensor.grad_data
// MatMulGradFn::apply() reads ctx.lm_logits_tensor.grad_data to propagate backward

// After backward:
ctx.clearIntermediates();  // ctx.lm_logits_tensor destructs → frees grad_data
```

**Lifetime:** Allocated during forward → used during backward → freed when context clears

### Special Case: Tied Weights (Issue #60)

When `tie_embeddings=true`, embedding and LM head share the SAME weight buffer:

```cpp
// Both Tensors point to same GPU memory:
embedding_weights_.data == lm_head_weights_.data  // TRUE (aliased)
embedding_weights_.grad_data == lm_head_weights_.grad_data  // TRUE (aliased)

// PCGrad handles conflicting gradients:
// 1. LM head backward writes to shared grad buffer
// 2. Embedding backward writes to TEMP buffer (pcgrad_temp_buffer)
// 3. PCGrad kernel projects and combines: grad_shared = grad_lm + (grad_emb - proj)
```

### Summary Table

| Gradient Type | Owner | Allocated By | Lifetime | Cleared By |
|--------------|-------|--------------|----------|------------|
| Weight gradients | Weight Tensor (owned by layer) | `weight_.ensure_grad()` | Until layer destroyed | `zeroGrad()` each window |
| Intermediate gradients | Intermediate Tensor (owned by context) | `intermediate.ensure_grad()` | Forward → backward → clear | `clearIntermediates()` |
| Tied weight grads | Shared between two Tensors | First `ensure_grad()` call | Until owning Tensor destroyed | `zeroGrad()` via owner |

**Key insight:** Gradients live where their Tensors live. If the layer owns the weight Tensor, the layer owns the weight gradients. If the context owns the intermediate Tensor, the context owns the intermediate gradients.

---

## Normalization Layers: Should They Be Layers?

**Current state:** RMSNorm is NOT a layer - it's just a struct and kernel functions.

```cpp
// RMSNorm_GPU.hpp - Just a struct, no class
struct RMSNormWeights {
    float* gamma = nullptr;       // Raw pointer!
    float* gamma_grad = nullptr;  // Raw pointer!
    int size = 0;
};

// Gammas stored in TrainingTensors god object
struct EncoderLayerParams {
    Tensor rms1_gamma;           // Pre-attention norm
    Tensor rms2_gamma;           // Pre-FFN norm  
    Tensor rms_post_attn_gamma;  // Post-attention norm
    Tensor rms_post_ffn_gamma;   // Post-FFN norm
    // 4 separate RMSNorm gammas per layer!
};
```

**The architectural question:** RMSNorm has a learnable parameter (gamma). According to Pattern B, **anything with learnable parameters should be a layer that self-manages**.

**What it SHOULD be with Pattern B:**

```cpp
class RMSNormLayer {
private:
    Tensor gamma_;  // Owns its scale parameter
    float epsilon_;
    int d_model_;
    cudaStream_t stream_;
    
    void allocateWeights() {
        gamma_ = Tensor::ones([d_model_], true, stream_, "rms_gamma");
        gamma_.ensure_grad();
    }

public:
    explicit RMSNormLayer(int d_model, float eps, cudaStream_t stream)
        : d_model_(d_model), epsilon_(eps), stream_(stream) 
    {
        allocateWeights();
    }
    
    Tensor forward(const Tensor& input) {
        return autograd::rms_norm(input, gamma_, epsilon_);
    }
    
    Tensor& getGamma() { return gamma_; }  // For buildParameterGroups()
};

class EncodingLayer {
private:
    RMSNormLayer rms1_;           // Each norm owns its gamma
    RMSNormLayer rms2_;
    RMSNormLayer rms_post_attn_;
    RMSNormLayer rms_post_ffn_;
    // Natural composition!
};
```

**Why this is better:**
- **Encapsulation** - gamma lifecycle tied to the norm instance that uses it
- **Consistency** - follows same pattern as all other layers with learnable params
- **Composability** - EncodingLayer composes RMSNorm instances naturally
- **No god object** - TrainingTensors doesn't need to know about 4 gammas × 12 layers = 48 separate norm parameters

**Scope:** 61 total RMSNorm instances in the model:
- 4 per encoder layer × 12 layers = 48
- 1 final RMSNorm before LM head = 1
- Total: 49 gamma parameters currently stored in TrainingTensors

**LayerScale - Another Learnable Parameter in TrainingTensors:**

```cpp
// TrainingTensors.hpp - Pattern A god object
struct EncoderLayerParams {
    Tensor layer_scale1;  // [1] - scales attention output before residual
    Tensor layer_scale2;  // [1] - scales FFN output before residual
    // 2 scalars per layer × 12 layers = 24 parameters in god object!
};

// autograd::layer_scale() - Forward: y = x * scale[0], Backward: grad_scale = sum(grad_y * x)
```

LayerScale (from CaiT paper, Issue #109) is a **learnable scalar** that multiplies residual connections to reduce correlation buildup. Same ownership problem as RMSNorm - Pattern B says these should be owned by EncodingLayer:

```cpp
class EncodingLayer {
private:
    RMSNormLayer rms1_, rms2_, rms_post_attn_, rms_post_ffn_;
    Tensor layer_scale1_;  // [1] - learnable scalar for attention residual
    Tensor layer_scale2_;  // [1] - learnable scalar for FFN residual
    
    void allocateWeights() {
        const float init_val = config_.layer_scale_init;  // e.g., 1.0
        layer_scale1_ = Tensor::ones({1}, stream_, "layer_scale1");
        layer_scale1_.requires_grad_();
        cudaMemcpyAsync(layer_scale1_.data, &init_val, sizeof(float), ...);
        // Same for layer_scale2_
    }
    
    Tensor& getLayerScale1() { return layer_scale1_; }
    Tensor& getLayerScale2() { return layer_scale2_; }
};
```

This refactor is part of the larger Pattern A → Pattern B migration (out of scope for current session).

**Fixed Scaling Constants (no ownership, just compile-time or config values):**

```cpp
// Attention scaling - fixed constant, no learnable parameter
const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));  // In Flash_Attention_Kernal.cu
// Used in attention score computation: score = (Q @ K^T) * scale

// Embedding scaling - config-time constant (Issue #140)
const float embedding_scale = 1.0f;  // In AutogradTraining.cu
// Was sqrt(d_model)=27.7 for AIAYN compatibility, removed because GRIM uses ALiBi/RoPE
// Used in embedding lookup: output[i] = weight[token_ids[i]] * embedding_scale

// Gradient scaling - runtime computation from batch stats
ClipSelection selection = TNC::computeClipSelection(base_clip, batch_stats);
// effective_clip = per_token_limit × total_tokens
```

**The distinction for scaling:**
- **Learnable scalar** (LayerScale) → **Layer owns Tensor** with Pattern B
- **Fixed constant** (attention scale, embedding scale) → **Just a float** in config or formula
- **Runtime computation** (gradient scaling) → **Utility function** calculates from data

No ownership complexity for fixed constants - they're just values used in kernels. Only learnable scalars need layer ownership.

---

## Optimizer State: Centralized Parallel Storage

**Current architecture (CORRECT):**

```cpp
// TrainingState_GPU.hpp - Centralized optimizer state management
struct TrainingState {
    std::vector<Tensor> optimizer_m_states;  // First moment per param group
    std::vector<Tensor> optimizer_v_states;  // Second moment per param group
    
    void allocateOptimizerStates(const std::vector<size_t>& sizes);
};

// TensorContract_GPU.hpp - Non-owning pointers to states
struct ParameterGroup {
    std::string name;
    Tensor* tensor;     // Non-owning pointer to weight
    Tensor* m_tensor;   // Non-owning pointer to momentum state
    Tensor* v_tensor;   // Non-owning pointer to variance state
    ParamGroupType type;
    int layer_index = -1;
};

// AdamW_Kernal_GPU.cu - Operates on ParameterGroup
void launchAdamWKernel(ParameterGroup& group, float lr, float weight_decay, int step, cudaStream_t stream) {
    // Reads from group.tensor, group.m_tensor, group.v_tensor
    // Updates all three in-place
}
```

**Why centralized is correct for optimizer state:**
- Optimizer states are **parallel storage** to parameters, not part of layer semantics
- 1:1 correspondence: each parameter Tensor has corresponding m/v Tensors
- TrainingState allocates them together via `allocateOptimizerStates(sizes)` where `sizes[i]` matches `parameter_groups_[i].size()`
- ParameterGroups hold **non-owning pointers** to both weights (owned by layers) and optimizer states (owned by TrainingState)
- Clean separation: layers own weights, training infrastructure owns optimizer state

**This is NOT the Pattern A anti-pattern because:**
- TrainingState doesn't know about layer internals
- Layers don't call `useExternalWeights()` for optimizer states
- ParameterGroups act as **lightweight coordination structs** pointing to separately-owned resources
- Similar to how PyTorch's `param_groups` in optimizers don't own parameters

**The distinction:**
- ❌ **Pattern A anti-pattern:** God object allocates layer weights, layers receive via `useExternalWeights(14 params)`
- ✅ **Optimizer state pattern:** TrainingState allocates optimizer states parallel to parameter_groups_, groups hold pointers

---

## Activation Functions: Stateless Autograd Operations

**Current architecture (CORRECT):**

```cpp
// GELU.hpp - No learnable parameters, pure function
struct GELUForwardArgs {
    TensorView input;   // Read-only
    TensorView output;  // Write destination
    cudaStream_t stream;
};

void launchGeluForward(const GELUForwardArgs& args);
void launchGeluBackward(const GELUBackwardArgs& args);

// TensorContract_GPU.hpp - Autograd wrapper
Tensor gelu(const Tensor& x, cudaStream_t stream, bool enable_grad);
// Creates GeluGradFn that captures input for backward pass
```

**Why stateless is correct:**
- GELU has no learnable parameters (unlike LayerScale or RMSNorm gamma)
- Formula: `y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- Just a mathematical function applied element-wise
- Same as `dropout()`, `add()`, `softmax()` - pure computation

**Other activation patterns:**
- **ReLU, Tanh, Sigmoid** - Same as GELU (stateless functions)
- **PReLU** - Would need learnable alpha parameter → should be a layer if added
- **LayerNorm** - Has learnable gamma/beta → should be a layer (like RMSNorm refactor)

**The distinction:**
- **Stateless activation** (GELU, ReLU) → Utility functions in autograd namespace
- **Parametric activation** (PReLU, LayerNorm) → Layer class with Pattern B

---

## Forward Pass Ownership: Context Owns Intermediates

**The tape-based autograd pattern:**

```cpp
// Layers implement forward() that RETURNS output Tensor
class EncodingLayer {
public:
    Tensor forward(const Tensor& input, int seq_len, cudaStream_t stream,
                   ForwardIntermediates& intermediates);
    // Returns output, does NOT store it
};

// Context storage: AutogradIntermediates (owned by TrainingState)
struct AutogradIntermediates {
    Tensor embedding_tensor;
    std::vector<Tensor> encoder_layer_outputs;  // One per layer
    Tensor encoder_output_tensor;
    Tensor logits_tensor;
    Tensor loss_tensor;
    
    // Per-layer intermediate storage (keeps autograd graph alive)
    AllLayerIntermediates layer_intermediates;
};

// Per-layer storage (Issue #56 - prevents grad_fn destruction)
struct ForwardIntermediates {
    Tensor ln1_out;          // RMSNorm output
    Tensor qkv_out;          // QKV projection
    Tensor attn_out;         // Attention result
    Tensor residual1;        // After attention residual
    Tensor ffn_out;          // FFN output
    // ... 17 intermediate tensors total per layer
};
```

**Why intermediates MUST be stored:**

Without storage, local Tensor variables are destroyed when `forward()` returns:
```cpp
Tensor EncodingLayer::forward(...) {
    Tensor ln1_out = rms_norm(input);    // Creates grad_fn
    Tensor attn_out = attention(ln1_out);  // Creates grad_fn → depends on ln1_out
    return residual_add(attn_out, input);  // Returns output
    // ← ln1_out destructor runs HERE
    // ← ln1_out.grad_fn is DELETED
    // ← attn_out.grad_fn still points to deleted ln1_out.grad_fn → USE-AFTER-FREE in backward!
}
```

With storage, intermediates survive until `backward()`:
```cpp
Tensor output = layer.forward(input, seq_len, stream, intermediates);
// intermediates.ln1_out, intermediates.attn_out, etc. stay alive
ctx.autograd_intermediates.encoder_layer_outputs.push_back(std::move(output));
// ... later ...
loss.backward();  // Safe: all grad_fn objects still exist
ctx.autograd_intermediates.clear();  // NOW safe to destroy
```

**Ownership summary:**
- **Layers implement**: `forward()` method - returns output Tensor
- **Context owns**: All intermediate Tensors from forward pass
- **Storage location**: `TrainingState::autograd_intermediates`
- **Lifetime**: Created during forward, cleared after backward
- **12 encoder layers × 17 intermediates = 204 Tensor objects** kept alive during training

---

## Backward Pass Ownership: Autograd Graph Self-Manages

**Backward is NOT a method on layers:**

```cpp
// NO EncodingLayer::backward() method!
// Backward is handled by the autograd graph

// Training flow:
loss_tensor.backward();  // Entry point
// → loss.grad_fn->apply() is called
// → propagates through computation graph automatically
// → each GradFn::apply() computes gradients for its inputs
// → gradients written directly to Tensor::grad_data buffers
```

**How backward traverses the graph:**

```cpp
// Simplified backward implementation in Tensor
void Tensor::backward(const Tensor* grad_output) {
    if (!grad_fn) return;  // Leaf tensor (e.g., weights)
    
    grad_fn->apply();  // Compute gradients for this operation
    // apply() internally calls backward() on input tensors
}

// Example: MatMulGradFn
struct MatMulGradFn : GradFn {
    Tensor input_a;  // Saved from forward
    Tensor input_b;  // Saved from forward
    
    void apply() override {
        // Compute grad_A = grad_output @ B^T
        Tensor grad_a = matmul(grad_output, input_b.transpose());
        
        // Write to input tensor's grad buffer
        input_a.backward(&grad_a);  // Propagate recursively
        
        // Compute grad_B = A^T @ grad_output
        Tensor grad_b = matmul(input_a.transpose(), grad_output);
        input_b.backward(&grad_b);  // Propagate recursively
    }
};
```

**Gradient buffer ownership:**

```cpp
// Tensors own their gradient buffers
struct Tensor {
    float* data = nullptr;       // Forward data (owned if owns_data=true)
    float* grad_data = nullptr;  // Gradient buffer (owned if has_grad=true)
    GradFn* grad_fn = nullptr;   // Backward function (owned)
    
    void ensure_grad() {
        if (!grad_data) {
            cudaMalloc(&grad_data, size_bytes());  // Tensor allocates its own grad
        }
    }
};
```

**Where gradients accumulate:**
- **Weight gradients**: Accumulate in layer-owned Tensor::grad_data (persistent)
- **Intermediate gradients**: Accumulate in context-owned Tensor::grad_data (ephemeral)
- Both write to their respective Tensor's grad_data - no separate backward-specific storage

**Ownership summary:**
- **Backward functions**: Owned by GradFn objects (created during forward)
- **GradFn lifetime**: Owned by output Tensor, kept alive by ForwardIntermediates storage
- **Gradient buffers**: Owned by Tensor objects (allocated via ensure_grad())
- **Backward propagation**: Self-managing recursive traversal, no explicit backward() methods
- **Lifecycle**: Graph built during forward, traversed during backward, destroyed after backward

**Stateless operations (no learnable parameters) don't need layer classes:**

```cpp
// Token-normalized gradient clipping - pure function, no parameters
namespace TNC {
    struct ClipSelection {
        float per_token_limit = 0.0f;
        float effective_clip_norm = 0.0f;  // = per_token_limit × total_tokens
        ClipMode mode = ClipMode::TokenNormalized;
    };
    
    // Pure function - computes scaling, no state
    ClipSelection computeClipSelection(float base_clip, 
                                      const BatchTokenStats& stats);
}
```

**The distinction:**
- **Learnable parameters** (gamma, weights, biases) → **Layer class** with Pattern B
- **Pure computation** (token-normalized clipping, scaling) → **Utility functions** in namespace

Token-normalized clipping is stateless logic that normalizes gradient clipping by `total_tokens`. It doesn't have learnable parameters, so it stays as utility functions in `TNC::` namespace. No ownership issues - it's just imported functions.

Similarly, if operations like dropout, activation functions, or scaling have no learnable parameters, they can be utility functions (or autograd ops). Only when parameters are introduced do they need layer ownership.

---

## Reference Implementation: ScratchBlockLayer

`ScratchBlockReasoning_GPU.{hpp,cu}` is the canonical example:

```cpp
class ScratchBlockLayer {
private:
    // 5 learnable Tensors (owned by layer)
    Tensor atom_type_embeddings_;      // [256, 768]
    Tensor atom_projection_;           // [768, 768]  
    Tensor text_feature_projection_;   // [768, 768]
    Tensor value_extraction_weight_;   // [768, 1]
    Tensor value_extraction_bias_;     // [1]
    
    // 4 raw cudaMalloc scratch buffers (NOT learnable, reused across forwards)
    int* d_atom_positions_;
    int* d_num_atoms_;
    float* d_atom_embeddings_;
    float* d_extraction_output_;
    
    void allocateWeights() {
        atom_type_embeddings_ = Tensor::zeros([256, d_model_], true, stream_, "atom_emb");
        atom_type_embeddings_.ensure_grad();
        // ... repeat for other 4 Tensors
        
        cudaMalloc(&d_atom_positions_, max_atoms_ * sizeof(int));
        // ... repeat for other 3 scratch buffers
    }
    
    void initializeWeights() {
        launchXavierInit(atom_type_embeddings_.data, ...);
        // ... repeat for other Tensors
    }

public:
    // Accessors for buildParameterGroups()
    Tensor& atomProjection() { return atom_projection_; }
    Tensor& textProjection() { return text_feature_projection_; }
    // ...
};
```

**Integration points:**
- `buildParameterGroups()` calls `layer->atomProjection()` to get Tensor reference
- Autograd tape automatically tracks gradients (registered via `ensure_grad()`)
- Serialization reads/writes via `layer->atomProjection().data` pointers
- No external code needs to know the layer has 5 weights vs 10 weights

---

## Technical Debt: Centralized Factory Anti-Pattern

The following layers violate the architectural standard by using `TrainingTensors` as a centralized factory. They SHOULD self-manage but currently don't:

### What's Wrong: The TrainingTensors God Object

`TrainingTensors` knows the internal structure of every layer:

```cpp
struct EncoderLayerParams {
    Tensor rms1_gamma;
    Tensor rms2_gamma;
    Tensor attn_qkv_weight;     // Layer internals exposed!
    Tensor attn_qkv_bias;
    Tensor attn_out_weight;
    Tensor attn_out_bias;
    Tensor ffn_w1;              // FFN sublayer structure leaked!
    Tensor ffn_b1;
    Tensor ffn_w2;
    Tensor ffn_b2;
    // ... 14 fields per layer!
};

class TrainingTensors {
    std::vector<EncoderLayerParams> encoder_layers;  // 12 layers × 14 params = 168 Tensors
    // TrainingTensors must know EVERYTHING about layer internals
};
```

Then layers receive via injection:

```cpp
// In TrainingOps.cu line ~298
for (int layer = 0; layer < cfg.num_layers; ++layer) {
    auto& params = training_state_.tensors_->encoder_layers[layer];
    gpu_layer->useExternalWeights(
        params.rms1_gamma,
        params.rms2_gamma,
        params.attn_qkv_weight,
        params.attn_qkv_bias,
        params.attn_out_weight,
        params.attn_out_bias,
        params.ffn_w1,
        params.ffn_b1,
        params.ffn_w2,
        params.ffn_b2,
        params.rms_post_attn_gamma,
        params.rms_post_ffn_gamma,
        ls1_ptr,
        ls2_ptr
    );  // 14 parameters passed explicitly!
}
```

**Problems:**
- Adding a weight to EncodingLayer requires modifying TrainingTensors, TrainingOps, serialization, and all call sites
- TrainingTensors is a 2000+ line god object that knows everyone's business
- Zero encapsulation — internals of every layer are public knowledge
- Tight coupling across the entire codebase

### Layers Using Anti-Pattern (Need Refactoring)

| Layer | Tensors | Current Wiring | Should Be |
|---|---|---|---|
| **EncodingLayer** | 14 per layer × 12 = 168 | `useExternalWeights(14 params)` | `allocateWeights()` like ScratchBlock |
| **FeedForwardLayer** | 4 per layer | Constructor injection from Encoding | `allocateWeights()` |
| **LMHeadLayer** | 2 (or 1 if tied) | Constructor injection, rebuilt per forward | `allocateWeights()` |
| **RMSNorm** | 1 per instance | Struct only, gammas in EncoderLayerParams | Tensor member |

**Files needing refactor:**
- `Encoding_GPU.{hpp,cu}` — Delete `useExternalWeights()`, add `allocateWeights()`
- `Feed_Forward_GPU.{hpp,cu}` — Move from constructor injection to self-allocation
- `lm_head_GPU.{hpp,cu}` — Self-allocate instead of external factory
- `TrainingOps.cu` line ~298 — Delete 14-parameter wiring loop
- `TrainingTensors` — Can be deleted entirely or reduced to simple container
- `LanguageModel_Training.cu` — buildParameterGroups accesses via layer APIs, not TrainingTensors struct

**This is a multi-week refactor** affecting initialization order, serialization, and ~30 call sites. Out of scope for current session.

---

## Current Status: Migration Complete

**ALL sessions completed (1-6). TrainingTensors god object DELETED.**

| Task | Status | Impact |
|---|---|---|
| 1. Delete orphaned workspace in InitInferenceState | ✅ DONE | Reclaimed GPU memory |
| 2. Delete FlashAttentionLayer class | ✅ DONE | Removed ~500 dead lines |
| 3. Keep QuantizationLayer | ⏭️ SKIP | Future feature, not dead |
| 4. Keep softmax_lse | ⏭️ SKIP | Active FlashAttention utility |
| 5. Update CMakeLists | ✅ DONE | Removed deleted targets |
| 6. FeedForwardLayer self-allocation | ✅ DONE | 4 Tensors per layer pattern B |
| 7. EncodingLayer + Forward_GPU migration | ✅ DONE | 168 Tensors migrated, `useExternalWeights` deleted |
| 8. LMHeadLayer self-allocation | ✅ DONE | 3 Tensors, tied weight handling encapsulated |
| 9. Diagnostic extraction | ✅ DONE | ~1660 lines to TrainingDiagnostics.cu |
| 10. EmbeddingLayer self-allocation | ✅ DONE | 2 Tensors, inference mode support |
| 11. TrainingTensors deletion | ✅ DONE | God object eliminated entirely |

**All 178/178 learnable parameters now owned by Pattern B layers.**

---

## Writing New Layers: The Standard

**ALL new layers MUST follow the ScratchBlock pattern:**

```cpp
class NewLayer {
private:
    // 1. Tensor member variables (owned by layer)
    Tensor weight_;
    Tensor bias_;
    bool weights_allocated_ = false;
    
    cudaStream_t stream_;
    
    // 2. Private allocation method
    void allocateWeights() {
        if (weights_allocated_) return;
        
        weight_ = Tensor::zeros([out_dim_, in_dim_], true, stream_, "new_layer_weight");
        weight_.ensure_grad();  // Register with autograd tape
        
        bias_ = Tensor::zeros([out_dim_], true, stream_, "new_layer_bias");
        bias_.ensure_grad();
        
        weights_allocated_ = true;
    }
    
    // 3. Private initialization method
    void initializeWeights() {
        if (!weights_allocated_) {
            throw std::runtime_error("Must allocate before initialize");
        }
        launchXavierInit(weight_.data, weight_.shape, seed_, stream_);
        cudaMemsetAsync(bias_.data, 0, bias_.numel() * sizeof(float), stream_);
    }

public:
    // 4. Constructor calls allocation
    explicit NewLayer(const LayerConfig& cfg) 
        : stream_(cfg.stream) 
    {
        allocateWeights();
        initializeWeights();
    }
    
    // 5. Public accessors for buildParameterGroups()
    Tensor& getWeight() { return weight_; }
    Tensor& getBias() { return bias_; }
    
    // 6. Forward pass uses autograd operations
    Tensor forward(const Tensor& input) {
        return autograd::linear(input, weight_, bias_);  // Creates GradFn on tape
    }
};
```

**Registration in `buildParameterGroups()`:**

```cpp
// In LanguageModel_Training.cu
if (new_layer_) {
    registerTensor(new_layer_->getWeight(), ParamGroupType::NEW_LAYER);
    registerTensor(new_layer_->getBias(), ParamGroupType::NEW_LAYER);
}
```

**That's it.** No changes to TrainingTensors, no changes to TrainingOps, no changes to serialization schemas (until you want checkpoints). The layer is completely self-contained.

---

## Orchestration Hierarchy: Who Calls What

**The training pipeline is a clean hierarchy with clear separation of concerns:**

```
train_gpu.cu::main()
  ↓
executePhase2(TrainingContext&) ← Training loop coordinator
  ↓
runEpoch(TrainingContext&, TrainingLoopState&) ← Epoch orchestrator
  ↓
processBatch(TrainingContext&, TrainingLoopState&, BatchAssignment&, indices) ← Batch orchestrator
  ↓
autogradTrainingStep(TrainingContext&, BatchPayload&, bool first_batch) ← Forward+Loss+Backward atomic unit
  ├─ GPU copies: payload → GPU buffers
  ├─ executeAutogradForward(AutogradContext&) → Builds computation graph
  │   ├─ embedding_layer->forward()
  │   ├─ scratch_block->forward()
  │   ├─ encoder_layer->forward() × 12
  │   └─ lm_head->forward()
  ├─ computeAutogradLoss(AutogradContext&, BatchPayload&) → Creates loss Tensor
  └─ executeAutogradBackward(AutogradContext&, loss_tensor) → Triggers autograd
      ├─ loss_tensor.backward() → Entry point
      ├─ CrossEntropyGradFn::apply() → Recursive traversal
      ├─ MatMulGradFn::apply() → Computes grad_A, grad_B, propagates
      └─ EmbeddingGradFn::apply() → Accumulates to weight.grad_data
```

**Separation of Concerns:**

| Level | File | Responsibility | Knows About |
|-------|------|----------------|-------------|
| **Phase Loop** | Phase2_TrainingLoop.cu | Epoch iteration, early stopping, checkpointing | Epochs, batches, stopping criteria |
| **Epoch Loop** | Phase2_TrainingLoop.cu::runEpoch() | Batch iteration, shuffling, guess cache reset | Batches, data order |
| **Batch Loop** | Phase2_TrainingLoop.cu::processBatch() | Gradient accumulation window, optimizer steps, diagnostics | Gradient clipping, AdamW, validation |
| **Atomic Step** | AutogradTraining.cu::autogradTrainingStep() | Single forward+loss+backward unit | GPU copies, non-finite loss detection |
| **Forward Pass** | AutogradTraining.cu::executeAutogradForward() | Build computation graph | Layer topology, intermediate storage |
| **Loss Computation** | AutogradTraining.cu::computeAutogradLoss() | Create loss Tensor with GradFn | Loss formula, label smoothing, focal weight |
| **Backward Pass** | AutogradTraining.cu::executeAutogradBackward() | Trigger recursive traversal | Gradient scaling, post-backward cleanup |
| **Autograd Graph** | TensorContract_GPU.cu | Self-managing backward propagation | GradFn chain, Tensor.grad_data, chain rule |

**Key insight:** Each level knows ONLY about its immediate children. `executeAutogradForward()` calls `layer->forward()` but doesn't know about optimizer steps. `processBatch()` knows about gradient clipping but doesn't know about layer internals. Clean separation of concerns.

**Where layers fit in:**

Layers are **leaf nodes** in this hierarchy — they implement `forward()` and return Tensors, but have **zero knowledge** of:
- Training loops, epochs, batches
- Gradient accumulation, clipping, optimizer steps
- Loss computation, backward propagation
- Other layers (except through composition, like FFN inside EncodingLayer)

Layers are **pure computational modules** that participate in the autograd tape. The orchestration hierarchy coordinates them without coupling.

---

## Position Indexing & Internal Batch Semantics

**Three-level indexing hierarchy** for tensor addressing:

```
Batch (B) → Sequence (S) → Token position (T)
     ↓            ↓               ↓
  batch_idx   within-batch    within-sequence
              sequence       autoregressive pos
```

### BatchPayload: Single Source of Truth

**All batch metadata computed ONCE by `buildBatchPayload()`, consumed everywhere:**

```cpp
struct BatchPayload {
    // IDENTITY (from scheduler)
    std::vector<uint32_t> seq_ids;       // which sequences from corpus [batch_size]
    
    // GEOMETRY (computed once, never recomputed)
    int batch_size = 0;                  // B = number of sequences
    int max_seq_len = 0;                 // S = longest sequence (pad target)
    int total_tokens = 0;                // B × S (includes padding)
    int actual_tokens = 0;               // sum of real lengths (no padding)
    int valid_tokens = 0;                // unmasked targets (for loss mean reduction)
    std::vector<int> seq_lengths;        // [B] actual length per sequence
    
    // PADDED DATA (flat [B×S] layout — GPU-ready)
    std::vector<int> input_ids;          // [total_tokens] padded with 0
    std::vector<int> target_ids;         // [total_tokens] padded with -1 (masked)
    std::vector<float> numeric_values;   // [total_tokens] padded with 0.0
    std::vector<uint8_t> numeric_mask;   // [total_tokens] padded with 0
    
    void validate();  // Rule 20: crash on ANY inconsistency
};
```

**Key insight:** Once built, `BatchPayload` is **immutable** — read-only reference passed everywhere. No component recomputes `total_tokens` or `valid_tokens`.

### Position Indexing Modes

**1. Flat Token Index (GPU kernels)**

```cpp
// Used in: embedding lookups, loss computation, gradient kernels
const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
if (flat_idx >= total_tokens) return;  // total_tokens = B × S

// Example: Embedding lookup
const int token_id = input_ids[flat_idx];
output[flat_idx] = embedding_weights[token_id];
```

**2. Batch + Sequence 2D Index (attention, matmul)**

```cpp
// Used in: FlashAttention, residual connections
const int b = flat_idx / max_seq_len;  // Which sequence in batch
const int s = flat_idx % max_seq_len;  // Position within sequence

// Check if position is real or padding
if (s >= seq_lengths[b]) {
    // Padding position — skip or mask
    return;
}
```

**3. Autoregressive Position (ALiBi/RoPE, position embeddings)**

```cpp
// Position within sequence [0, seq_len-1], repeated for each batch element
__global__ void generatePositionIdsKernel(int* position_ids, int total_tokens, int seq_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;
    
    position_ids[idx] = idx % seq_len;  // [0,1,2,...,S-1, 0,1,2,...,S-1, ...]
}

// Used in: ALiBi bias, RoPE rotation
const int pos = position_ids[flat_idx];
const float alibi_bias = alibi_slope * pos;
```

### Batch Construction Pipeline

**Five-phase builder (BatchPayload.cu):**

```
Phase 1: Identity → seq_ids from scheduler (BatchAssignment)
Phase 2: Extract + Geometry → load sequences, compute max_seq_len, actual_tokens
Phase 3: Cache fit check → crash if batch exceeds GPU buffer sizes
Phase 4: Pad + Flatten → create [B×S] flat arrays, count valid_tokens
Phase 5: Validate → crash if ANY invariant violated (Rule 20)
```

**Example: 3 sequences of lengths [5, 3, 4] with max_seq_len=5:**

```
Sequence 0: [101, 234, 456, 789, 234]     ← length 5 (no padding)
Sequence 1: [101, 789, 234, 0,   0]       ← length 3 (2 pad tokens)
Sequence 2: [101, 456, 234, 789, 0]       ← length 4 (1 pad token)

Flat layout: [101,234,456,789,234, 101,789,234,0,0, 101,456,234,789,0]
             └────── seq 0 ──────┘ └─── seq 1 ───┘ └──── seq 2 ────┘
             positions: 0...4      positions: 0...4  positions: 0...4

total_tokens = 3 × 5 = 15
actual_tokens = 5 + 3 + 4 = 12
padding_tokens = 15 - 12 = 3
packing_efficiency = 12/15 = 80%
```

### Target Masking Strategy

**Two masking rules (applied during Phase 4 padding):**

1. **Autoregressive boundary:** Final position of EACH sequence masked with `-1` (no next-token target)
2. **Non-content tokens:** UNK, PAD, BOS masked (never valid prediction targets). EOS IS valid (model must learn to predict end).

```cpp
// Copy targets EXCEPT final position
if (seq_len > 1) {
    memcpy(&target_ids[row_offset], targets, (seq_len - 1) * sizeof(int));
}
target_ids[row_offset + seq_len - 1] = -1;  // Mask final position

// Defense mask non-content tokens leaked through DataLoader
for (int t = 0; t < seq_len - 1; ++t) {
    if (target_ids[t] >= 0 && token_layout.isNonContent(target_ids[t])) {
        target_ids[t] = -1;  // Safety net masking
    }
}
```

**Why final position is masked:** In autoregressive training, each token predicts the NEXT token. The final token has no "next" token to predict within the sequence (next would be in a different sequence or padding). Masking prevents model from training on nonsense targets.

### Ownership: Who Owns What During Training?

| Component | Owner | Lifetime | Indexing |
|-----------|-------|----------|----------|
| **BatchAssignment** | Scheduler (Phase2) | Epoch duration | seq_ids only |
| **BatchPayload** | processBatch() stack | Single batch | All indices |
| **GPU buffers** | TrainingState | Persistent | Reused across batches |
| **Position IDs** | AutogradContext temp | Forward+backward | Generated per batch |
| **Sequence metadata** | DataLoader views | Training duration | Indexed via seq_ids |

**Critical:** `BatchPayload` lives on **CPU stack** as `const&` reference. GPU gets memcpy'd data (input_ids, targets, etc.). Position indices are **generated on-demand** via CUDA kernel (not stored in BatchPayload).

### Typical Access Patterns

**Embedding lookup (flat index):**

```cpp
// AutogradTraining.cu line ~270
const int total_tokens = ctx.batch_size * ctx.seq_len;
ctx.embedding_tensor = autograd::embedding(
    weight, 
    payload.input_ids.data(),  // [total_tokens] flat array
    total_tokens,
    d_model,
    ctx.stream
);
```

**Loss computation (flat index with mask):**

```cpp
// ComputeLossBatch.cu
for (int t = 0; t < total_tokens; ++t) {
    const int target = targets[t];
    if (target < 0) continue;  // Skip masked positions
    
    const float* logits_t = &logits[t * vocab_size];
    loss += -logf(logits_t[target]);
    ++valid_count;
}
loss_mean = loss / valid_count;  // Mean reduction over valid_tokens
```

**Diagnostic (2D index):**

```cpp
// Phase2_TrainingLoop.cu line ~1732
for (int i = 0; i < valid_positions.size(); ++i) {
    const int t = valid_positions[i];           // Flat index
    const int b = t / max_seq_len;              // Which sequence
    const int s = t % max_seq_len;              // Position in sequence
    const int seq_len = payload.seq_lengths[b]; // Actual length
    
    if (s >= seq_len) continue;  // Skip padding positions
    // ... analyze token at [b,s]
}
```

### Invariants (Rule 20: Crash if Violated)

```cpp
// BatchPayload::validate() enforces:
assert(total_tokens == batch_size * max_seq_len);
assert(valid_tokens > 0);  // Must have trainable targets
assert(seq_lengths.size() == batch_size);
assert(input_ids.size() == total_tokens);
assert(sum(valid_target_counts) == valid_tokens);  // Cross-check
assert(sum(seq_lengths) == actual_tokens);         // Cross-check
```

If ANY invariant fails, training crashes with detailed error message. No silent fallbacks.

---

## Logging System & Equation-Based Diagnostics

**Three-tier logging infrastructure:**

```
Rule 21 Equation Logs (stderr direct) → humans debug math
      ↓
ModuleLog System (async queue) → structured module logging
      ↓
WebSocket Server (port 9002) → real-time UI streaming
```

### Rule 21: Equation-Based Diagnostic Logging (Mandatory for ML Code)

**From copilot-instructions.md Rule 21:**

> When adding diagnostic logging for AI/ML math operations, ALWAYS use the `[*_EQUATION]` format. This format is non-negotiable for training/inference debugging because **you can't argue with hard mathematical facts**.

**Required Format Structure:**

```cpp
fprintf(stderr, "[OPERATION_EQUATION] operation_name: mathematical_formula\n");
fprintf(stderr, "  INPUT (description): shape=[dims] min=X max=Y rms=Z\n");
fprintf(stderr, "  ACTUAL result: shape=[dims] min=X max=Y rms=Z\n");
// Optional: PARAMETERS (only if operation has configurable params)
// Optional: EXPECTED (only if result is predictable from inputs)
// Optional: [ANOMALY] (only if something is wrong)
```

**Statistics:** `rms` = Root Mean Square = sqrt(mean(x²)) = typical size/scale of values
  - Use to detect explosion (rms >> 1) or vanishing (rms << 0.001)

**Production Examples:**

**1. Hidden State Weight Gradient (Issue #37, #108, #128)**

```cpp
// Phase2_TrainingLoop.cu line ~1058
std::ostringstream oss;
oss << "[HIDDEN_STATE_EQUATION] GRAD_W[" << tok << "]: "
    << "grad_W[" << tok << ",i] = Σ_t (hidden[t,i] × grad_logits[t," << tok << "])\n";
oss << "  HIDDEN STATES (encoder output): mean=" << hidden_mean
    << " ||h||_mean=" << hidden_norm_mean << " std=" << hidden_std << "\n";
oss << "  AT_TRACKED_TARGETS (n=" << count_tracked << "): ||h||=" << hidden_tracked_norm << "\n";
oss << "  AT_OTHER_TARGETS (n=" << count_other << "): ||h||=" << hidden_other_norm << "\n";
oss << "  GRAD_LOGITS[" << tok << "]: at_tracked=" << grad_at_tracked 
    << " (p_t - 1, always negative), at_other=" << grad_at_other 
    << " (p_v/N, always positive for pure CE)\n";
oss << "  GRAD_W[" << tok << "] PER-DIMENSION DECOMPOSITION:\n";
oss << "    from_tracked_targets: ||g||=" << grad_norm_tracked << "\n";
oss << "    from_other_targets: ||g||=" << grad_norm_other << "\n";
oss << "    cos(g_tracked, g_other)=" << grad_cosine 
    << " (negative=opposing, positive=reinforcing)\n";
if (std::abs(hidden_mean) > 0.001f) {
    oss << "  [ANOMALY] NON_ZERO_HIDDEN_MEAN: hidden_mean=" << hidden_mean 
        << " should be ~0 for RMSNorm output. This biases weight gradients!\n";
}
```

**Why this works:** Decomposition into `from_tracked` vs `from_other` reveals gradient sign conflicts. When cosine is negative (opposing), total gradient can flip sign → weight paradox.

**2. Feedback Loop Decomposition (Issue #114)**

```cpp
// Phase2_TrainingLoop.cu line ~1483
oss << "[FEEDBACK_LOOP_EQUATION] TOKEN_" << tok << "_MODE_COLLAPSE: "
    << "logit[" << tok << "] = ||h|| × ||W[" << tok << "]|| × cos(h, W[" << tok << "])\n";
oss << "  COMPONENT BREAKDOWN:\n";
oss << "    ||h||_mean=" << hidden_norm_mean << " growth=" << hidden_growth_pct << "%\n";
oss << "    ||W[" << tok << "]||=" << weight_norm << " growth=" << weight_growth_pct << "%\n";
oss << "    cos(h, W)=" << cosine_mean << " growth=" << cosine_growth_pct << "%\n";
oss << "  PREDICTED logit = " << hidden_norm_mean << " × " << weight_norm 
    << " × " << cosine_mean << " = " << predicted_logit << "\n";
oss << "  ACTUAL logit_mean=" << actual_logit << " error=" << error_pct << "%\n";
oss << "  GROWTH RATE: logit_growth=" << logit_growth_pct << "% per batch\n";
if (weight_paradox) {
    oss << "  [ANOMALY] WEIGHT_PARADOX: grad·W=" << grad_dot_w 
        << " > 0 (should shrink) BUT ||W|| GREW " << weight_growth_pct << "%\n";
}
```

**Why this works:** Each component's growth rate reveals which term drives explosion. If `||h||` grows +111% while `||W||` only +32%, encoder amplification is the root cause.

**3. Logit Scale Expectation (Issue #130, #133)**

```cpp
// Phase2_TrainingLoop.cu line ~4088
oss << "[LOGIT_SCALE_EQUATION] logit[v] = h · W[v]^T, logit_range = max - min\n";
oss << "  LOGIT STATS: std=" << logit_std << " range=" << logit_range << "\n";
oss << "  HIDDEN (LM input): ||h||_mean=" << h_norm_mean << " ||h||_max=" << h_norm_max 
    << " h_rms_mean=" << h_rms_mean << "\n";
oss << "  WEIGHTS (LM head): ||W||_mean=" << w_norm_mean << " ||W||_rms=" << w_norm_rms 
    << " ||W||_max=" << w_norm_max << " (tok=" << w_norm_max_tok << ")\n";
oss << "  EXPECTED logit_std = sqrt(d_model) × h_rms × ||W||_rms = "
    << sqrt(d_model) << " × " << h_rms_mean << " × " << w_norm_rms 
    << " = " << expected_logit_std << "\n";
oss << "  ACTUAL logit_std = " << logit_std 
    << " ratio(actual/expected)=" << logit_std_ratio << "\n";
if (logit_std_ratio > 3.0f) {
    oss << "  [ANOMALY] LOGIT_STD_MISMATCH: actual/expected=" << logit_std_ratio 
        << " >> 3.0. Possible hidden-weight alignment or missing 1/sqrt(d) scaling.\n";
}
```

**Why this works:** Expected formula from statistical math. If `ratio > 3×`, model deviates from random initialization assumptions → hidden-weight correlation.

### Standard Tags in Production

| Tag | Purpose | File Location |
|-----|---------|---------------|
| `[HIDDEN_STATE_EQUATION]` | Weight gradient sign decomposition (Issue #37, #128) | Phase2_TrainingLoop.cu:1058 |
| `[FEEDBACK_LOOP_EQUATION]` | Mode collapse amplification tracking (Issue #114) | Phase2_TrainingLoop.cu:1483 |
| `[LOGIT_SCALE_EQUATION]` | Logit magnitude expectation vs actual (Issue #133) | Phase2_TrainingLoop.cu:4088 |
| `[EMB_GRAD_EQUATION]` | Tied-weight gradient spike analysis (Issue #141) | Phase2_TrainingLoop.cu:688 |
| `[WEIGHT_GRADIENT_EQUATION]` | AdamW weight update decomposition | Phase2_TrainingLoop.cu:845 |
| `[EMBED_COSINE_EQUATION]` | Hidden state correlation before encoder | AutogradTraining.cu:447 |
| `[SPECIAL_TOKEN_EQUATION]` | Special token (BOS, EOS, UNK) health check | Phase2_TrainingLoop.cu:4615 |

**Writing New Equations:**

```cpp
// BAD: No equation, just values
fprintf(stderr, "Loss spike: loss=165.23 at batch 42\n");

// GOOD: Equation + decomposition + expected vs actual
fprintf(stderr, "[LOSS_EQUATION] CE_LOSS: loss = -ln(p_target)\n");
fprintf(stderr, "  LOGITS: min=%.4f max=%.4f std=%.4f\n", logit_min, logit_max, logit_std);
fprintf(stderr, "  TARGET: token_id=%d p_target=%.10f (expected ~1/V=%.6f)\n", 
        target_id, p_target, 1.0f / vocab_size);
fprintf(stderr, "  EXPECTED loss ~= -ln(1/V) = %.2f\n", -logf(1.0f / vocab_size));
fprintf(stderr, "  ACTUAL loss = %.4f\n", actual_loss);
if (actual_loss > 20.0f) {
    fprintf(stderr, "  [ANOMALY] LOSS_EXPLOSION: p_target=%.2e (near-zero probability)\n", p_target);
}
- // DONT use fprintf though use logging systems
```

**Mandatory Elements Checklist:**

- [ ] Mathematical formula with variable names in first line and the algebraic expression
- [ ] Actual value computation equation algabraically derived from inputs (not just raw numbers)
- [ ] Expected value computation from input statistics (if predictable)


**Why Expected Value is Critical:**

Without expected value, you can't tell if `logit_std=0.17` is good or bad. WITH expected value:
```
EXPECTED logit_std = sqrt(768) × 0.006 × 0.87 ≈ 0.14
ACTUAL logit_std = 0.17
ratio = 1.2× → OK (within 2× is noise)
```

vs

```
EXPECTED logit_std = sqrt(768) × 0.006 × 0.87 ≈ 0.14
ACTUAL logit_std = 24.3
ratio = 173× → ANOMALY! Hidden-weight explosion
```

**Pre-commit checklist for ML code:**

- [ ] All tensor operations have `[*_EQUATION]` logging available (can be behind `#ifdef DEBUG_EQUATIONS`)
- [ ] Expected values computed from input statistics
- [ ] Anomaly thresholds defined (e.g., score > 100, LSE > 50, gradient > 1e6)
- [ ] Log includes tensor shapes for dimensional analysis

### ModuleLog System (Structured Logging)

**Async multi-module logging with level-based filtering:**

```cpp
// LogRecorder.hpp - Core system
namespace GRIM::Logging {
    enum class ModuleLogLevel { Info = 0, Warning = 1, Error = 2 };
    enum class ModuleId {
        ForwardPass, BackwardPass, Optimizer, Scheduler,
        Loss, Attention, Autograd, Training, ...
    };
    
    void EmitModuleLog(const std::string& module_name,
                       ModuleLogLevel level,
                       std::string_view message,
                       uint64_t global_step = 0,
                       bool force_sync = false);
}
```

**Usage in Autograd:**

```cpp
// AutogradTraining.cu lines 40-46
#define AG_INFO(msg) do { \
    std::ostringstream _oss; _oss << msg; \
    GRIM::Logging::EmitModuleInfo("Autograd", _oss.str()); \
} while(0)
#define AG_ERROR(msg) do { std::cerr << "[AutogradTraining] ERROR: " << msg << std::endl; } while(0)
#define AG_WARN(msg) do { std::cerr << "[AutogradTraining] WARN: " << msg << std::endl; } while(0)

// Example usage:
AG_INFO("Autograd Forward: batch=" << ctx.batch_size << " seq=" << ctx.seq_len);
AG_INFO("Step 1: Embedding complete, shape=[" << total_tokens << ", " << d_model << "]");
AG_INFO("Step 2: Running " << num_layers << " encoder layers with autograd...");
```

**Module-Level Configuration (ai_config.json):**

```json
{
  "logging": {
    "default_level": "Info",
    "module_overrides": [
      {"module": "ForwardPass", "level": "Warning"},
      {"module": "Loss", "level": "Info"},
      {"module": "Attention", "level": "Error"}
    ]
  }
}
```


**Use Cases:**

- **Remote monitoring:** Watch training from different machine
- **Real-time diagnostics:** See equation logs without SSH/RDP
- **Multi-client:** Multiple viewers (browser dashboard + terminal + file)
- **Low latency:** Sub-second log delivery vs polling files

### File-Based Logging (Traditional)


**GRIM-text stderr → file redirect:**

```powershell
# Training run with log capture
./resources/models/grim-text/training/trainingloop/build/release/train_gpu.exe *>&1 | Tee-Object -FilePath resources/models/GRIM-text/training/logs/training_run.log
```

**Equation logs bypass ModuleLog** — always to stderr, always synchronous, always visible. This ensures critical math diagnostics are NEVER lost in async queue drops.

### Logging Decision Tree

**"Which logging system should I use?"**

```
Is this ML/math diagnostic? (gradients, loss, attention scores, weight norms)
    YES → Use [*_EQUATION] format to stderr
          - Mathematical formula in first line
          - Expected vs actual comparison
    NO ↓

Is this a phase transition? (initialization done, epoch start, checkpoint saved)
    YES → Use LOG_PHASE() for structured phase tracking
    NO ↓

Is this module-scoped debug info? (forward pass step, layer output)
    YES → Use AG_INFO/AG_WARN/AG_ERROR (or module-specific macros)
    NO ↓

Is this a fatal error?
    YES → Use LOG_ERROR() + throw std::runtime_error() (Rule 20: fail loud)
    NO ↓
Skip logging (noise reduction)
```

### Performance Considerations

**Equation logging cost:**

- `fprintf(stderr)` blocks for disk I/O (~0.1-1ms per call)
- Keep equation logs sparse: batch intervals (every 10-100 batches)
- Use conditional compilation for heavy diagnostics:
  ```cpp
  #ifdef DEBUG_EQUATIONS
      fprintf(stderr, "[DETAILED_EQUATION] ...\n");
  #endif
  ```

**ModuleLog async queue:**

- Non-blocking for caller (0.001ms overhead)
- Background thread writes to file/websocket
- Queue overflow → oldest logs dropped with warning
- Flush on shutdown via `FlushModuleLogQueue()`

**WebSocket streaming:**

- Max 100 messages/sec per client (rate limiting)
- Auto-disconnect slow clients to prevent backpressure
- Broadcast mode: all clients get all logs (no filtering yet)

### Ownership & Lifecycle

| Component | Owner | Lifetime | Thread Safety |
|-----------|-------|----------|---------------|
| **Equation logs (stderr)** | Immediate output | N/A | Kernel mutex (OS) |
| **ModuleLog queue** | Global singleton | Process lifetime | Async queue (lock-free) |
| **ModuleLogSink** | Caller stack | RAII (destructor unregisters) | Thread-safe delegate |
| **WebSocket server** | main.cpp static | wsServer.start() → stop() | Internal thread pool |
| **Logger file (grim.log)** | Logger singleton | initLogger() → shutdownLogger() | std::mutex |

**Critical:** Equation logs are **synchronous** and **unfiltered** because math diagnostics must NEVER be lost. If training diverges, the equation log WILL show why — even if async ModuleLog queue drops messages due to backpressure.

---

## File Inventory

### ✅ Correct Architecture (Self-Managed Layers)
- `ScratchBlockReasoning_GPU.{hpp,cu}` — 5 learnable Tensors, private `allocateWeights()`, registers with autograd tape
- `Embedding_GPU.{hpp,cu}` — 2 learnable Tensors (token + position), `EmbeddingLayerConfig` with `requires_grad` for inference
- `Encoding_GPU.{hpp,cu}` — 14 learnable Tensors per layer, self-allocated RMSNorm gammas + LayerScale + attention/FFN weights
- `Feed_Forward_GPU.{hpp,cu}` — 4 learnable Tensors (w1, b1, w2, b2), config struct constructor
- `lm_head_GPU.{hpp,cu}` — 3 learnable Tensors (weights, bias, final_rms_gamma), tied-weight handling via `Tensor::from_ptr()`

### ✅ Deleted (God Object Eliminated)
- ~~`TrainingTensors.{hpp,cu}`~~ — **DELETED** in Session 6. Zero weight parameters remained.


### 🔵 Kept (Not Dead)
- `Quantization_GPU.{hpp,cu}` — Future Int8/FP16 quantization feature
- `softmax_lse.{hpp,cu}` — Active FlashAttention utility
- `QKV_Projector.{hpp,cu}` — Single kernel utility (no weights)

**Total Parameters:** 178 Tensor registrations in `buildParameterGroups()`
- All 178 via layer accessors (Pattern B — correct)

---

## Summary

**Architectural Standard:** Layer self-management (ScratchBlock pattern)  
**Migration Status:** ✅ COMPLETE — All 178/178 parameters migrated to Pattern B  
**TrainingTensors:** ✅ DELETED — God object eliminated entirely  
**Sessions Completed:** 1 (dead code), 2 (FFN), 3 (Encoding), 4 (LMHead), 5 (diagnostics), 6 (Embedding + deletion)  
**Future Improvements:** RMSNormLayer / LayerScaleLayer sub-classes (low priority, cosmetic encapsulation)

---

## Change Log: Pattern A → Pattern B Migration

### Session 1: Dead Code Cleanup

| Change | Files | Impact |
|--------|-------|--------|
| Delete orphaned workspace in InitInferenceState | `InitinferenceState.cu` | Reclaimed GPU memory |
| Delete FlashAttentionLayer class | `Flash_Attention_Kernal.cu/hpp` | Removed ~500 dead lines |
| Update CMakeLists | `TrainingLoop/CMakeLists.txt` | Removed deleted targets |

### Session 2: FeedForwardLayer Self-Allocation

| Change | Files | Impact |
|--------|-------|--------|
| FeedForwardLayer self-allocates `w1_`, `b1_`, `w2_`, `b2_` | `Feed_Forward_GPU.{hpp,cu}` | 4 Tensors per layer now layer-owned |
| Constructor takes config struct, allocates + Xavier-inits | `Feed_Forward_GPU.{hpp,cu}` | No more external injection |

### Session 3: EncodingLayer + Forward_GPU + TrainingOps Encoder Migration

| Change | Files | Impact |
|--------|-------|--------|
| EncodingLayer self-allocates RMSNorm gammas, LayerScale, attention + FFN weights | `Encoding_GPU.{hpp,cu}` | 14 Tensors per layer now layer-owned |
| Delete `useExternalWeights(14 params)` | `Encoding_GPU.{hpp,cu}` | Removed god-object injection |
| Forward_GPU unconditional Pattern B | `Forward_GPU.cu` | No more conditional pattern switching |
| TrainingOps encoder creation via self-allocating constructor | `TrainingOps.cu` | Deleted 14-param wiring loop |
| TrainingTensors `EncoderLayerParams` removed | `TrainingTensors.{hpp,cu}` | 168 Tensors removed from god object |

### Session 4: LMHeadLayer Self-Allocation

| Change | Files | Impact |
|--------|-------|--------|
| LMHeadLayer self-allocates `weights_`, `bias_`, `final_rms_gamma_` | `lm_head_GPU.{hpp,cu}` | Persistent layer, not rebuilt per forward |
| Constructor handles tied weights via `Tensor::from_ptr()` + `share_grad()` | `lm_head_GPU.{hpp,cu}` | Weight tying encapsulated inside layer |
| `final_rms_gamma_` owned by LMHeadLayer (RMSNorm is Step 0 of forward) | `lm_head_GPU.{hpp,cu}` | Norm gamma co-located with its consumer |
| `getLmHeadLayer()` accessor on LanguageModel | `grim_language_model_cuda.hpp` | Clean access pattern for all consumers |
| AutogradContext wired with `LMHeadLayer*` | `AutogradTraining.{hpp,cu}` | Forward/backward use layer accessors |
| `lm_head_weights`, `lm_head_bias`, `final_rms_gamma` removed from TrainingTensors | `TrainingTensors.{hpp,cu}` | 3 more Tensors removed from god object |
| `getLmHeadWeights()` deleted from TrainingState | `TrainingState_GPU.{hpp}`, `TrainingStateGPU.cu` | No more indirection layer |
| `initializeAutogradTensors` signature simplified (removed `tie_embeddings`/`use_bias`) | `TrainingState_GPU.{hpp}`, `TrainingStateGPU.cu` | LMHeadLayer handles config internally |
| All consumer references updated (~50 sites across 10 files) | `Phase2_TrainingLoop.cu`, `Phase1_Startup.cu`, `InitTrainingState.cu`, `InitinferenceState.cu`, `LanguageModel_Training.cu`, `grim_model_serialization.cu`, `diagnostic_train_minimal.cu` | Zero remaining `tensors_->lm_head_*` references |

### Session 5: Diagnostic Extraction from Orchestrator

| Change | Files | Impact |
|--------|-------|--------|
| Extract ~1660 lines of diagnostic functions from Phase2_TrainingLoop.cu | `training/Diagnostics/TrainingDiagnostics.{hpp,cu}` (NEW) | Diagnostics in own compilation unit |
| Moved functions: `sampleWeightStats`, `computeToken277Diagnostic`, `computeFeedbackLoopDiagnostic`, `computePC1CausalityTest`, `computeEmbGradEquation`, `computeUpdateRms` | `Phase2_TrainingLoop.cu` → `TrainingDiagnostics.cu` | Orchestrator no longer includes layer headers for diagnostics |
| Moved types: `WeightSample`, `FeedbackLoopState` | `Phase2_TrainingLoop.cu` → `TrainingDiagnostics.hpp` | Diagnostic state encapsulated |
| Phase2_TrainingLoop.cu calls via `GRIM::Diagnostics::` namespace | `Phase2_TrainingLoop.cu` | Clean separation of concerns |
| Added to CMakeLists.txt source list | `TrainingLoop/CMakeLists.txt` | New .cu compiled |

### Session 6: EmbeddingLayer Self-Allocation + TrainingTensors Deletion

| Change | Files | Impact |
|--------|-------|--------|
| Created `EmbeddingLayer` class (Pattern B, self-allocating) | `Layers/Embedding/Embedding_GPU.{hpp,cu}` (NEW) | Token + position weights layer-owned |
| `EmbeddingLayerConfig` with `requires_grad` field for inference mode | `Layers/Embedding/Embedding_GPU.hpp` | Inference creates without grad buffers |
| Added `getEmbeddingLayer()` accessor on LanguageModel | `grim_language_model_cuda.hpp` | Clean access pattern for all consumers |
| AutogradContext wired with `EmbeddingLayer*` | `AutogradTraining.{hpp,cu}` | All 18+ embedding references via layer accessors |
| TrainingOps section 6 split: 6a EmbeddingLayer + 6b LMHeadLayer | `TrainingOps.cu` | EmbeddingLayer created BEFORE LMHeadLayer (weight tying dependency) |
| Migrated `computeEmbGradEquation`, `computeToken277Diagnostic` signatures | `TrainingDiagnostics.{hpp,cu}` | Take `const EmbeddingLayer*` instead of `const TrainingState&` |
| Migrated `traceGradientComponents` | `diagnostic_train_minimal.cu` | Uses `model.getEmbeddingLayer()->tokenWeights()` |
| Migrated all debug validation in InitTrainingState | `InitTrainingState.cu` | Uses `embedding_layer_->` with `hasPositionEmbeddings()` guards |
| Migrated inference path to Pattern B | `InitinferenceState.cu` | EmbeddingLayer created with `requires_grad=false` |
| Migrated `logGradientAttribution` to take `const EmbeddingLayer*` | `TrainingState_GPU.{hpp}`, `TrainingStateGPU.cu` | No more embedding accessors on TrainingState |
| Deleted 4 embedding accessor declarations/implementations | `TrainingState_GPU.{hpp}`, `TrainingStateGPU.cu` | Clean API surface |
| Removed `embedding_weights`, `position_embedding_weights`, `cached_embeddings` from TrainingTensors | `TrainingTensors.{hpp,cu}` | Zero weight params remaining |
| Replaced `initializeAutogradTensors(12 params)` with `initializeAutogradSeed(seed)` | `TrainingState_GPU.{hpp}`, `TrainingStateGPU.cu`, `Phase1_Startup.cu` | Simple seed storage on TrainingState |
| **DELETED TrainingTensors god object entirely** | `TrainingTensors.{hpp,cu}` DELETED | Zero remaining weight params — object was zombie |
| Removed `#include "TrainingTensors.hpp"` from 6+ files | Multiple files | No stale includes |
| Added `Embedding_GPU.cu` to CMakeLists, removed `TrainingTensors.cu` | `TrainingLoop/CMakeLists.txt` | Build updated |
| Updated all diagnostic call sites in Phase2_TrainingLoop.cu | `Phase2_TrainingLoop.cu` | Uses `ctx.model->getEmbeddingLayer()` |

### ✅ Migration Complete — All Technical Debt Resolved

| Component | Tensors | Status |
|-----------|---------|--------|
| **EmbeddingLayer** | 1 (token) + 1 (position, if learned) | ✅ Self-managed (Session 6) |
| **EncodingLayer** | 14 per layer × 12 = 168 | ✅ Self-managed (Session 3) |
| **LMHeadLayer** | 3 (weights, bias, final_rms_gamma) | ✅ Self-managed (Session 4) |
| **ScratchBlockLayer** | 5 | ✅ Self-managed (was already correct) |
| **TrainingTensors** | — | ✅ **DELETED** (Session 6) |
| **`useExternalWeights()`** | — | ✅ **DELETED** (Session 3) |
| **`buildParameterGroups()`** | 178 registrations | ✅ All via layer accessors |

**Parameter ownership progress:**
- ✅ 168 encoder params → EncodingLayer self-manages
- ✅ 3 LM head params → LMHeadLayer self-manages
- ✅ 5 ScratchBlock params → ScratchBlockLayer self-manages
- ✅ 2 embedding params → EmbeddingLayer self-manages
- **Total: 178/178 migrated (100%) — COMPLETE**

### Future Improvements (Not Blocking)

| Component | Description | Priority |
|-----------|-------------|----------|
| **RMSNormLayer sub-class** | EncodingLayer stores 4 gammas directly; could compose 4 `RMSNormLayer` sub-objects instead | Low — works correctly, just less composable |
| **LayerScale sub-class** | 2 scalar Tensors per encoder layer; could be a tiny `LayerScaleLayer` class | Low — functional as-is |
