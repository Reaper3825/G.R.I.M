//======================================================//
//  EXAMPLE: Before and After Logit Integration
//======================================================//

#include "TensorContract_GPU.hpp"

// ============================================================
// BEFORE: Manual Logit Tracking (OLD WAY)
// ============================================================

void train_batch_old_way(TrainingState& ts, const int* targets, cudaStream_t stream) {
    int total_tokens = ts.total_tokens;
    int vocab_size = ts.vocab_size;
    
    // Step 1: Forward pass - manual logit computation
    // (LM head GEMM happens here, writes to ts.cached_logits)
    lm_head_forward(ts.encoder_output, ts.cached_logits, stream);
    
    // Step 2: Loss computation - manual
    LossInputs loss_inputs{
        .logits = ts.cached_logits,
        .targets = targets,
        .total_tokens = total_tokens,
        .vocab_size = vocab_size,
        // ... many more fields
    };
    
    LossTelemetry telemetry;
    float loss_value = computeLoss(loss_inputs, &telemetry, stream);
    
    // Step 3: Backward - MANUAL gradient computation
    launchCrossEntropyBackward(
        ts.cached_logits,   // Input logits
        targets,            // Target labels
        ts.grad_logits,     // Output gradient buffer (must pre-allocate!)
        total_tokens,
        vocab_size,
        stream
    );
    
    // Step 4: LM head backward - uses grad_logits
    lm_head_backward(ts.grad_logits, ts.grad_encoder_output, stream);
    
    // Problems:
    // - Manual gradient computation (easy to forget)
    // - Separate loss/backward steps (can diverge)
    // - Must manually allocate grad_logits buffer
    // - No type safety (cached_logits is just float*)
}

// ============================================================
// AFTER: Automatic Logit Tracking (NEW WAY)
// ============================================================

void train_batch_new_way(TrainingState& ts, const int* targets, cudaStream_t stream) {
    using namespace GRIM;
    using namespace GRIM::autograd;
    
    int total_tokens = ts.total_tokens;
    int vocab_size = ts.vocab_size;
    
    // Step 1: Wrap logits in Tensor with LOGITS layout
    Tensor logits = Tensor::from_ptr(
        ts.cached_logits,
        TensorContract::TensorShape::make_LOGITS(total_tokens, vocab_size),
        false,  // TrainingState owns memory, don't take ownership
        true    // requires_grad=true for automatic tracking
    );
    logits.name = "lm_head_logits";
    
    // Step 2: Compute loss - AUTOMATIC gradient graph creation
    Tensor loss = cross_entropy(logits, targets, total_tokens, vocab_size, stream);
    
    // Step 3: Backward - ONE LINE! Computes grad_logits automatically
    loss.backward();
    
    // Step 4: Access gradients - they're already computed!
    float* grad_logits = logits.grad;  // Ready to use
    
    // Optional: Copy to existing buffer for compatibility
    if (ts.grad_logits && ts.grad_logits != logits.grad) {
        cudaMemcpyAsync(ts.grad_logits, logits.grad,
                       total_tokens * vocab_size * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }
    
    // Step 5: LM head backward - same as before
    lm_head_backward(logits.grad, ts.grad_encoder_output, stream);
    
    // Benefits:
    // ✅ Automatic gradient computation (can't forget backward)
    // ✅ Unified loss/backward (mathematically consistent)
    // ✅ Lazy grad allocation (only when needed)
    // ✅ Type-safe (Layout::LOGITS explicit)
    // ✅ Zero overhead (same kernels, just wrapped)
}

// ============================================================
// COMPARISON: Lines of Code
// ============================================================

// OLD WAY:
// - 15+ lines of boilerplate
// - Manual gradient kernel launch
// - Easy to forget backward
// - No compile-time layout checking

// NEW WAY:
// - 5 essential lines (wrap, loss, backward, access, use)
// - Automatic gradient computation
// - Impossible to forget backward (compile error if loss unused)
// - Compile-time layout validation

// ============================================================
// GRADUAL MIGRATION PATH
// ============================================================

void gradual_migration_example(TrainingState& ts, const int* targets, cudaStream_t stream) {
    using namespace GRIM;
    
    // OLD CODE: Keep existing LM head forward (writes to ts.cached_logits)
    lm_head_forward(ts.encoder_output, ts.cached_logits, stream);
    
    // NEW CODE: Wrap result in Tensor for automatic tracking
    Tensor logits = Tensor::from_ptr(
        ts.cached_logits,
        TensorContract::TensorShape::make_LOGITS(ts.total_tokens, ts.vocab_size),
        false, true
    );
    
    // NEW CODE: Automatic loss + backward
    auto loss = GRIM::autograd::cross_entropy(logits, targets, 
                                              ts.total_tokens, ts.vocab_size, stream);
    loss.backward();
    
    // OLD CODE: Continue using existing backward logic (just points to logits.grad)
    ts.grad_logits = logits.grad;  // Alias - no copy needed
    lm_head_backward(ts.grad_logits, ts.grad_encoder_output, stream);
    
    // This way you can migrate piece by piece without breaking existing code!
}

// ============================================================
// VALIDATION: Verify Gradients Match
// ============================================================

bool validate_gradients_match() {
    using namespace GRIM;
    
    // Allocate test data
    const int tokens = 100;
    const int vocab_size = 1000;
    float *logits, *manual_grad, *auto_grad;
    int *targets;
    
    cudaMalloc(&logits, tokens * vocab_size * sizeof(float));
    cudaMalloc(&manual_grad, tokens * vocab_size * sizeof(float));
    cudaMalloc(&targets, tokens * sizeof(int));
    // ... initialize with test data
    
    // OLD WAY: Manual gradient computation
    launchCrossEntropyBackward(logits, targets, manual_grad, tokens, vocab_size, nullptr);
    
    // NEW WAY: Automatic gradient computation
    Tensor logits_tensor = Tensor::from_ptr(
        logits,
        TensorContract::TensorShape::make_LOGITS(tokens, vocab_size),
        false, true
    );
    auto loss = GRIM::autograd::cross_entropy(logits_tensor, targets, tokens, vocab_size, nullptr);
    loss.backward();
    auto_grad = logits_tensor.grad;
    
    // COMPARE: Should be identical
    bool match = compareGPUBuffers(manual_grad, auto_grad, tokens * vocab_size, 1e-5f);
    
    // Cleanup
    cudaFree(logits);
    cudaFree(manual_grad);
    cudaFree(targets);
    // Note: auto_grad is freed when logits_tensor goes out of scope
    
    return match;
}
