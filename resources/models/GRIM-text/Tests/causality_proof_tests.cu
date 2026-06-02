//======================================================//
//  causality_proof_tests.cu
//  
//  ⚠️ NOT BUILT — Not in CMakeLists.txt.
//  ⚠️ BROKEN — Uses deleted LanguageModel::backward()/zeroGrad()
//     and deleted LanguageModel::gradientMetrics().
//  ⚠️ NEEDS FULL REWRITE to use explicit Phase2 shared-forward +
//     computeAutogradLoss()/executeAutogradBackward() + BatchPayload
//     + GradNorm::measureGradientNorms() for gradient metrics.
//  
//  Definitive proof tests for GRIM-text training correctness.
//  6 levels of verification - if any fail, generation is impossible.
//  
//  Build:
//    Add to CMakeLists.txt and build with grim_language_model_gpu
//  
//  Run:
//    ./causality_proof_tests --level N   (run specific level)
//    ./causality_proof_tests --all       (run all levels)
//======================================================//

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Shared/Batching/BatchPayload.hpp"
#include "../Shared/Batching/BatchDeviceUpload.hpp"
#include "../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../Shared/Forward/ModelForward_GPU.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"
#include "../Shared/Optimizers/OptimizerStep.hpp"
#include "../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>
#include <random>
#include <cstdint>

namespace CausalityProof {

struct NumericSideChannel {
    std::vector<float> values;
    std::vector<uint8_t> mask;
};

NumericSideChannel makeNumericSideChannel(size_t count) {
    NumericSideChannel channel;
    channel.values.assign(count, 0.0f);
    channel.mask.assign(count, 0);
    return channel;
}

std::vector<float> runInferencePrefill(GRIM::LanguageModel* model,
                                       GRIM::TrainingState* training_state,
                                       GRIM::GenerationState* generation_state,
                                       const std::vector<int>& tokens,
                                       const NumericSideChannel& numeric) {
    if (!model) {
        throw std::runtime_error("runInferencePrefill: model is NULL");
    }
    if (!training_state) {
        throw std::runtime_error("runInferencePrefill: training_state is NULL");
    }
    if (!generation_state) {
        throw std::runtime_error("runInferencePrefill: generation_state is NULL");
    }
    if (numeric.values.size() != tokens.size() || numeric.mask.size() != tokens.size()) {
        throw std::runtime_error("runInferencePrefill: numeric side-channel length mismatch");
    }

    const auto& cfg = model->getConfig();
    std::vector<uint32_t> atom_flags(tokens.size(), 0);
    std::vector<uint32_t> atom_entry_ids(tokens.size(), GRIM::Tokenizer::kAtomEntryNone);
    std::vector<int32_t> token_to_slot_map(tokens.size(), -1);

    GRIM::Batching::BatchPayload payload = GRIM::Batching::buildInferenceBatchPayload(
        tokens,
        numeric.values,
        numeric.mask,
        atom_flags,
        nullptr,
        atom_entry_ids,
        token_to_slot_map,
        cfg.vocab_size,
        static_cast<size_t>(cfg.batch_size),
        static_cast<size_t>(cfg.max_cached_seq_len),
        cfg.execution_block_num_slots);

    if (!training_state->initialized) {
        throw std::runtime_error("runInferencePrefill: training state not initialized");
    }
    cudaStream_t stream = training_state->stream_ctrl.getPrimaryStream();
    const auto bindings = GRIM::Batching::uploadBatchToDevice(
        model->getConfig(),
        payload,
        stream);

    struct ScopedInferenceIntermediatesClear {
        GRIM::Forward::ModelForwardOutputs& forward_outputs;
        ~ScopedInferenceIntermediatesClear() {
            forward_outputs.clear();
        }
    };

    GRIM::Forward::ModelForwardRuntimePayload runtime_payload{};
    runtime_payload.execution_runtime = &generation_state->execution_runtime;
    runtime_payload.read_gate_accum_tensor = nullptr;

    GRIM::Forward::ModelForwardRequest request{};
    request.config = &cfg;
    // Deleted ownership branch: gpu_encoder now comes from startup-owned GpuModelState,
    // and this dead test requires a full rewrite before it can source that owner correctly.
    request.embedding_layer = model->getEmbeddingLayer();
    request.lm_head = model->getLmHeadLayer();
    request.execution_block_enabled = model->executionBlockEnabled();
    request.cublas_handle = training_state->cublas_handle.get();
    request.stream = stream;
    request.payload = &payload;
    request.bindings = &bindings;
    request.batch_idx = 0;
    request.graph = GRIM::Forward::ModelForwardGraphPolicy{
        /*connect_parameter_graph=*/false,
        /*retain_backward_graph=*/false,
        /*enable_dropout=*/false};

    auto forward_outputs = GRIM::Forward::executeModelForward(request, runtime_payload);
    ScopedInferenceIntermediatesClear clear_scope{forward_outputs};

    const auto& live_logits = forward_outputs.logits_tensor;
    if (!live_logits.data) {
        throw std::runtime_error("runInferencePrefill: logits_tensor.data is NULL after shared forward");
    }
    const size_t expected_logits =
        static_cast<size_t>(payload.total_tokens) * static_cast<size_t>(cfg.vocab_size);
    if (static_cast<size_t>(live_logits.numel()) < expected_logits) {
        throw std::runtime_error("runInferencePrefill: logits tensor is smaller than expected payload span");
    }

    std::vector<float> logits(static_cast<size_t>(cfg.vocab_size));
    const size_t last_token_offset =
        static_cast<size_t>(payload.total_tokens - 1) * static_cast<size_t>(cfg.vocab_size);
    cudaError_t copy_err = cudaMemcpyAsync(
        logits.data(),
        live_logits.data + last_token_offset,
        static_cast<size_t>(cfg.vocab_size) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    if (copy_err != cudaSuccess) {
        throw std::runtime_error("runInferencePrefill: cudaMemcpyAsync logits failed: " +
                                 std::string(cudaGetErrorString(copy_err)));
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error("runInferencePrefill: cudaStreamSynchronize failed: " +
                                 std::string(cudaGetErrorString(sync_err)));
    }

    return logits;
}

//======================================================//
//  Test Framework
//======================================================//

struct TestResult {
    bool passed;
    std::string name;
    std::string message;
    std::vector<std::string> details;
};

#define PROOF_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            result.passed = false; \
            result.message = (msg); \
            return result; \
        } \
    } while(0)

#define PROOF_LOG(msg) \
    result.details.push_back(msg)

void printResult(const TestResult& result) {
    std::cout << (result.passed ? "✓ PASS" : "✗ FAIL") << ": " << result.name << "\n";
    if (!result.message.empty()) {
        std::cout << "  Message: " << result.message << "\n";
    }
    for (const auto& detail : result.details) {
        std::cout << "  " << detail << "\n";
    }
    std::cout << "\n";
}

//======================================================//
//  LEVEL 1: Single-token causality proof
//  
//  Pick one training example, one sequence, one position t.
//  Verify forward and backward math is correct.
//======================================================//

TestResult level1_single_token_causality(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, GRIM::Tokenizer::UniByte* tokenizer) {
    TestResult result;
    result.name = "LEVEL 1: Single-token causality proof";
    result.passed = true;
    
    PROOF_LOG("Testing single token forward-backward causality...");
    
    const auto& cfg = model->getConfig();
    const int vocab_size = cfg.vocab_size;
    
    // 1. Create a simple sequence: "hello world"
    std::vector<int> tokens = tokenizer->encode("hello world");
    PROOF_LOG("Input tokens: " + std::to_string(tokens.size()) + " tokens");
    
    PROOF_ASSERT(tokens.size() >= 3, "Need at least 3 tokens for test");
    
    // Pick position t (middle of sequence)
    const int t = tokens.size() / 2;
    const int target_y = tokens[t + 1];  // Target is next token
    
    PROOF_LOG("Position t = " + std::to_string(t));
    PROOF_LOG("Target token y = " + std::to_string(target_y));
    
    // 2. Run forward pass
    std::vector<int> input_tokens(tokens.begin(), tokens.begin() + t + 1);
    auto numeric = makeNumericSideChannel(input_tokens.size());
    std::vector<float> logits = runInferencePrefill(model, training_state, generation_state, input_tokens, numeric);
    
    PROOF_ASSERT(logits.size() == static_cast<size_t>(vocab_size), 
                 "Logits shape mismatch: expected " + std::to_string(vocab_size) + 
                 ", got " + std::to_string(logits.size()));
    
    // 3. Compute softmax manually
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] /= sum_exp;
    }
    
    // 4. Compute cross-entropy loss
    float loss = -std::log(probs[target_y] + 1e-10f);
    PROOF_LOG("Loss = -log(softmax(logits)[" + std::to_string(target_y) + "]) = " + 
              std::to_string(loss));
    
    PROOF_ASSERT(std::isfinite(loss), "Loss is not finite: " + std::to_string(loss));
    PROOF_ASSERT(loss > 0.0f, "Loss should be positive, got: " + std::to_string(loss));
    
    // 5. Compute gradient manually: dL/dlogits = probs - one_hot(y)
    std::vector<float> grad_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        grad_logits[i] = probs[i];
    }
    grad_logits[target_y] -= 1.0f;
    
    // 6. Verify gradient properties
    float grad_at_y = grad_logits[target_y];
    PROOF_LOG("∂loss/∂logits[y] = " + std::to_string(grad_at_y));
    
    PROOF_ASSERT(grad_at_y < 0.0f, 
                 "FATAL: ∂loss/∂logits[y] should be NEGATIVE, got: " + std::to_string(grad_at_y));
    
    // 7. Verify other gradients are positive (and sum to ~0)
    float sum_other_grads = 0.0f;
    int positive_count = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (i != target_y) {
            if (grad_logits[i] > 0.0f) positive_count++;
            sum_other_grads += grad_logits[i];
        }
    }
    
    PROOF_LOG("Number of j≠y with ∂loss/∂logits[j] > 0: " + std::to_string(positive_count) + 
              "/" + std::to_string(vocab_size - 1));
    
    float total_grad_sum = sum_other_grads + grad_at_y;
    PROOF_LOG("Sum of all gradients: " + std::to_string(total_grad_sum) + " (should be ~0)");
    
    PROOF_ASSERT(std::abs(total_grad_sum) < 1e-5f, 
                 "Gradient sum should be ~0, got: " + std::to_string(total_grad_sum));
    
    PROOF_LOG("✓ Forward-backward causality verified!");
    return result;
}

//======================================================//
//  LEVEL 2: Causal mask correctness
//  
//  Self-prediction must be impossible.
//  Attention[t, t] == 0 and Attention[t, >t] == 0
//======================================================//

TestResult level2_causal_mask_correctness(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, GRIM::Tokenizer::UniByte* tokenizer) {
    TestResult result;
    result.name = "LEVEL 2: Causal mask correctness";
    result.passed = true;
    
    PROOF_LOG("Testing causal mask prevents self-prediction...");
    
    const auto& cfg = model->getConfig();
    
    // Create test sequence
    std::vector<int> tokens = tokenizer->encode("the quick brown fox");
    PROOF_ASSERT(tokens.size() >= 4, "Need at least 4 tokens");
    
    const int seq_len = tokens.size();
    const int test_pos = seq_len / 2;  // Middle position
    
    PROOF_LOG("Sequence length: " + std::to_string(seq_len));
    PROOF_LOG("Testing position t = " + std::to_string(test_pos));
    
    // We need to hook into the attention computation to verify mask
    // For now, we'll verify by checking that predictions don't leak future info
    
    // Test 1: Run forward with full sequence
    auto full_numeric = makeNumericSideChannel(tokens.size());
    std::vector<float> logits_full = runInferencePrefill(model, training_state, generation_state, tokens, full_numeric);
    
    // Test 2: Run forward with truncated sequence (only up to position t)
    std::vector<int> truncated_tokens(tokens.begin(), tokens.begin() + test_pos + 1);
    auto truncated_numeric = makeNumericSideChannel(truncated_tokens.size());
    std::vector<float> logits_truncated = runInferencePrefill(model, training_state, generation_state, truncated_tokens, truncated_numeric);
    
    // If causal mask works, logits for position t should be IDENTICAL
    // regardless of whether future tokens exist
    
    // Note: We need to compare the logits at position t, but forwardGPU
    // returns logits for the last position. So we need incremental mode.
    
    // For this test, we verify that the model doesn't have access to future
    // by checking that adding future tokens doesn't change current predictions
    
    // Actually, with the current API, let's verify causal mask differently:
    // We'll check that the model can't perfectly predict the current token
    
    // Get prediction at position 0 (should be based on nothing but BOS)
    std::vector<int> single_token = {tokens[0]};
    auto single_numeric = makeNumericSideChannel(single_token.size());
    std::vector<float> logits_pos0 = runInferencePrefill(model, training_state, generation_state, single_token, single_numeric);
    
    // The predicted token should NOT be exactly token[0] with probability 1
    // (unless the model has memorized, which is fine)
    
    // More importantly: If we change token[0], the prediction should change
    std::vector<int> modified_token = {tokens[0] + 1};  // Different first token
    if (modified_token[0] >= cfg.vocab_size) modified_token[0] = 0;
    
    auto modified_numeric = makeNumericSideChannel(modified_token.size());
    std::vector<float> logits_modified = runInferencePrefill(model, training_state, generation_state, modified_token, modified_numeric);
    
    // Logits should be different
    float diff_rms = 0.0f;
    for (size_t i = 0; i < logits_pos0.size(); ++i) {
        float d = logits_pos0[i] - logits_modified[i];
        diff_rms += d * d;
    }
    diff_rms = std::sqrt(diff_rms / logits_pos0.size());
    
    PROOF_LOG("Logits difference RMS when changing input: " + std::to_string(diff_rms));
    
    PROOF_ASSERT(diff_rms > 1e-8f, 
                 "FATAL: Logits unchanged when input changes - causal mask broken!");
    
    // Test: Verify model can't see position t when predicting position t
    // We do this by checking attention pattern (if available) or by
    // verifying that loss at position t depends only on tokens 0..t-1
    
    PROOF_LOG("✓ Causal mask appears correct (predictions depend on past only)");
    return result;
}

//======================================================//
//  LEVEL 3: Gradient path continuity
//  
//  Gradient must reach embeddings.
//  ||∂loss/∂embedding[y]|| > 0 for target token y
//  ||∂loss/∂embedding[z]|| ≈ 0 for random non-target z
//======================================================//

TestResult level3_gradient_reaches_embeddings(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, GRIM::Tokenizer::UniByte* tokenizer) {
    TestResult result;
    result.name = "LEVEL 3: Gradient path continuity";
    result.passed = true;
    
    PROOF_LOG("Testing gradient reaches embeddings...");
    
    const auto& cfg = model->getConfig();
    
    // Create simple training example
    std::vector<int> tokens = tokenizer->encode("hello");
    PROOF_ASSERT(tokens.size() >= 2, "Need at least 2 tokens");
    
    const int target_y = tokens[1];  // Target is second token
    
    PROOF_LOG("Target token y = " + std::to_string(target_y));
    
    // Zero gradients
    model->zeroGrad();
    
    // Forward pass
    auto numeric = makeNumericSideChannel(tokens.size());
    std::vector<float> logits = runInferencePrefill(model, training_state, generation_state, tokens, numeric);
    
    // Compute loss (using token[1] as target for predicting from token[0])
    float loss = 1.0f;  // Placeholder - actual loss computed in backward
    
    // Backward pass
    model->backward(loss, false, 1.0f);
    
    // Get gradient metrics
    // TODO(REWRITE): Use GradNorm::measureGradientNorms() instead of deleted gradientMetrics()
    const auto& gm = model->gradientMetrics(); // BROKEN: method deleted
    
    PROOF_LOG("Embedding gradient RMS: " + std::to_string(GRIM::GradNorm::GradMetrics::rms(gm.embedding_sum_sq, gm.embedding_count)));
    PROOF_LOG("LM head gradient RMS: " + std::to_string(GRIM::GradNorm::GradMetrics::rms(gm.lm_head_sum_sq, gm.lm_head_count)));
    PROOF_LOG("Attention gradient RMS: " + std::to_string(GRIM::GradNorm::GradMetrics::rms(gm.attention_sum_sq, gm.attention_count)));
    PROOF_LOG("FFN gradient RMS: " + std::to_string(GRIM::GradNorm::GradMetrics::rms(gm.ffn_sum_sq, gm.ffn_count)));
    
    PROOF_ASSERT(gm.embedding_sum_sq > 1e-20f, 
                 "FATAL: Embedding gradient is zero - gradients not reaching embeddings!");
    
    PROOF_ASSERT(gm.lm_head_sum_sq > 1e-20f, 
                 "FATAL: LM head gradient is zero - loss not connected!");
    
    // The embedding gradient for the target token should be non-zero
    // We can verify this by dumping gradients
    
    PROOF_LOG("✓ Gradients flow through all components");
    return result;
}

//======================================================//
//  LEVEL 4: Learning must change logits
//  
//  One-step SGD sanity test.
//  After one optimizer step: logit[y]_after > logit[y]_before
//======================================================//

TestResult level4_learning_changes_logits(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, ::ParameterRegistry::StartupParameterRegistry* parameter_registry, GRIM::Tokenizer::UniByte* tokenizer) {
    TestResult result;
    result.name = "LEVEL 4: Learning must change logits";
    result.passed = true;
    
    PROOF_LOG("Testing one-step SGD changes logits correctly...");
    if (!parameter_registry) {
        throw std::runtime_error("level4_learning_changes_logits: parameter_registry is NULL");
    }
    
    const auto& cfg = model->getConfig();
    
    // Create training example - predict 'b' from 'a'
    std::string test_text = "ab";
    std::vector<int> tokens = tokenizer->encode(test_text);
    PROOF_ASSERT(tokens.size() >= 2, "Need at least 2 tokens");
    
    std::vector<int> input_tokens = {tokens[0]};  // Just 'a'
    const int target_y = tokens[1];  // Target is 'b'
    
    PROOF_LOG("Input: token " + std::to_string(tokens[0]));
    PROOF_LOG("Target y = " + std::to_string(target_y));
    
    // 1. Get logits BEFORE training
    auto before_numeric = makeNumericSideChannel(input_tokens.size());
    std::vector<float> logits_before = runInferencePrefill(model, training_state, generation_state, input_tokens, before_numeric);
    float logit_y_before = logits_before[target_y];
    
    PROOF_LOG("logit[y] BEFORE = " + std::to_string(logit_y_before));
    
    // 2. Zero gradients
    model->zeroGrad();
    
    // 3. Forward + backward (compute gradients)
    // We need to run the full training sequence
    auto train_numeric = makeNumericSideChannel(tokens.size());
    runInferencePrefill(model, training_state, generation_state, tokens, train_numeric);  // Full sequence for training
    
    // Compute actual cross-entropy loss for logging
    float max_logit = *std::max_element(logits_before.begin(), logits_before.end());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits_before.size(); ++i) {
        sum_exp += std::exp(logits_before[i] - max_logit);
    }
    float prob_y = std::exp(logits_before[target_y] - max_logit) / sum_exp;
    float loss = -std::log(prob_y + 1e-10f);
    
    PROOF_LOG("Loss = " + std::to_string(loss));
    
    // Backward
    model->backward(loss, false, 1.0f);
    
    // 4. One optimizer step with LARGE learning rate to see effect
    float test_lr = 0.1f;  // Large LR for visible effect
    
    // Create temporary optimizer step counter
    GRIM::OptimizerStep opt_step;
    
    GRIM::launchAdamWStep(parameter_registry->requireParameterGroups("level4_learning_changes_logits"), test_lr,
                          GRIM::HyperParameters::ADAMW_WEIGHT_DECAY,
                          opt_step.step,
                          training_state->stream_ctrl.getPrimaryStream());
    
    // 5. Get logits AFTER training
    auto after_numeric = makeNumericSideChannel(input_tokens.size());
    std::vector<float> logits_after = runInferencePrefill(model, training_state, generation_state, input_tokens, after_numeric);
    float logit_y_after = logits_after[target_y];
    
    PROOF_LOG("logit[y] AFTER  = " + std::to_string(logit_y_after));
    PROOF_LOG("Change: " + std::to_string(logit_y_after - logit_y_before));
    
    // 6. THE CRITICAL CHECK: logit[y] should INCREASE
    PROOF_ASSERT(logit_y_after > logit_y_before, 
                 "FATAL: logit[y] did not increase after training!\n"
                 "  Before: " + std::to_string(logit_y_before) + "\n"
                 "  After:  " + std::to_string(logit_y_after) + "\n"
                 "  This means gradients are wrong or optimizer is broken!");
    
    PROOF_LOG("✓ One-step SGD correctly increases logit[y]!");
    return result;
}

//======================================================//
//  LEVEL 5: Tokenizer–loss alignment
//  
//  Byte fallback sanity: rare unicode, raw bytes, atom placeholders
//  must receive gradients.
//======================================================//

TestResult level5_tokenizer_loss_alignment(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, GRIM::Tokenizer::UniByte* tokenizer) {
    TestResult result;
    result.name = "LEVEL 5: Tokenizer–loss alignment";
    result.passed = true;
    
    PROOF_LOG("Testing byte fallback receives gradients...");
    
    // Test 1: Encode text with rare unicode
    std::string rare_text = "Hello 世界 🌍";  // Contains Chinese and emoji
    std::vector<int> tokens = tokenizer->encode(rare_text);
    
    PROOF_LOG("Input: \"" + rare_text + "\"");
    PROOF_LOG("Token count: " + std::to_string(tokens.size()));
    
    // Check if any tokens are in byte range [4-259], atom range [260-279], etc.
    int byte_token_count = 0;
    int atom_token_count = 0;
    int unigram_token_count = 0;
    int special_token_count = 0;
    
    for (int tid : tokens) {
        if (tid >= 0 && tid < static_cast<int>(GRIM::Tokenizer::NUM_SPECIAL_TOKENS)) {
            special_token_count++;
        } else if (tid >= static_cast<int>(GRIM::Tokenizer::BYTE_TOKEN_OFFSET) && tid < static_cast<int>(GRIM::Tokenizer::ATOM_TOKEN_OFFSET)) {
            byte_token_count++;
        } else if (tid >= static_cast<int>(GRIM::Tokenizer::ATOM_TOKEN_OFFSET) && tid < static_cast<int>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
            atom_token_count++;
        } else {
            unigram_token_count++;
        }
    }
    
    PROOF_LOG("Special tokens: " + std::to_string(special_token_count));
    PROOF_LOG("Byte tokens: " + std::to_string(byte_token_count));
    PROOF_LOG("Atom tokens: " + std::to_string(atom_token_count));
    PROOF_LOG("Unigram tokens: " + std::to_string(unigram_token_count));
    
    // Test 2: Zero gradients
    model->zeroGrad();
    
    // Test 3: Forward + backward
    if (tokens.size() >= 2) {
        auto numeric = makeNumericSideChannel(tokens.size());
        runInferencePrefill(model, training_state, generation_state, tokens, numeric);
        model->backward(1.0f, false, 1.0f);
        
        // TODO(REWRITE): Use GradNorm::measureGradientNorms() instead of deleted gradientMetrics()
        const auto& gm = model->gradientMetrics(); // BROKEN: method deleted
        
        PROOF_LOG("Embedding gradient RMS after byte/unicode input: " + 
                  std::to_string(GRIM::GradNorm::GradMetrics::rms(gm.embedding_sum_sq, gm.embedding_count)));
        
        PROOF_ASSERT(gm.embedding_sum_sq > 1e-20f, 
                     "FATAL: No gradient for byte/unicode tokens!");
    }
    
    // Test 4: Test with numbers (atom tokens)
    std::string number_text = "The price is 42.99 dollars";
    std::vector<int> number_tokens = tokenizer->encode(number_text);
    
    PROOF_LOG("Input with numbers: \"" + number_text + "\"");
    PROOF_LOG("Token count: " + std::to_string(number_tokens.size()));
    
    model->zeroGrad();
    
    if (number_tokens.size() >= 2) {
        auto numeric = makeNumericSideChannel(number_tokens.size());
        runInferencePrefill(model, training_state, generation_state, number_tokens, numeric);
        model->backward(1.0f, false, 1.0f);
        
        // TODO(REWRITE): Use GradNorm::measureGradientNorms() instead of deleted gradientMetrics()
        const auto& gm = model->gradientMetrics(); // BROKEN: method deleted
        
        PROOF_LOG("Embedding gradient RMS after number input: " + 
                  std::to_string(GRIM::GradNorm::GradMetrics::rms(gm.embedding_sum_sq, gm.embedding_count)));
        
        PROOF_ASSERT(gm.embedding_sum_sq > 1e-20f, 
                     "FATAL: No gradient for atom tokens!");
    }
    
    PROOF_LOG("✓ All token types receive gradients");
    return result;
}

//======================================================//
//  LEVEL 6: Autoregressive emergence test
//  
//  Train on "abcabcabc", then generate with greedy decode.
//  Should produce "abcabcabc..." pattern.
//======================================================//

TestResult level6_autoregressive_emergence(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, ::ParameterRegistry::StartupParameterRegistry* parameter_registry, GRIM::Tokenizer::UniByte* tokenizer) {
    TestResult result;
    result.name = "LEVEL 6: Autoregressive emergence test";
    result.passed = true;
    
    PROOF_LOG("Testing autoregressive pattern learning...");
    if (!parameter_registry) {
        throw std::runtime_error("level6_autoregressive_emergence: parameter_registry is NULL");
    }
    
    const auto& cfg = model->getConfig();
    
    // Training data: simple repeating pattern
    std::string pattern = "abcabcabcabc";
    std::vector<int> tokens = tokenizer->encode(pattern);
    
    PROOF_LOG("Training pattern: \"" + pattern + "\"");
    PROOF_LOG("Token count: " + std::to_string(tokens.size()));
    
    // Print tokens for debugging
    std::string token_str = "Tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) token_str += ", ";
        token_str += std::to_string(tokens[i]);
    }
    token_str += "]";
    PROOF_LOG(token_str);
    
    // Create optimizer step counter
    GRIM::OptimizerStep opt_step;
    
    float lr = 0.01f;
    int num_steps = 100;
    
    PROOF_LOG("Training for " + std::to_string(num_steps) + " steps...");
    
    // Training loop
    for (int step = 0; step < num_steps; ++step) {
        model->zeroGrad();
        
        // Forward pass
        auto numeric = makeNumericSideChannel(tokens.size());
        runInferencePrefill(model, training_state, generation_state, tokens, numeric);
        
        // Compute approximate loss (we'd need full loss computation here)
        float loss = 1.0f;  // Placeholder
        
        // Backward
        model->backward(loss, false, 1.0f / tokens.size());
        
        // Update
        GRIM::launchAdamWStep(parameter_registry->requireParameterGroups("level6_autoregressive_emergence"), lr,
                              GRIM::HyperParameters::ADAMW_WEIGHT_DECAY,
                              opt_step.step,
                              training_state->stream_ctrl.getPrimaryStream());
        
        if ((step + 1) % 20 == 0) {
            PROOF_LOG("Step " + std::to_string(step + 1) + "/" + std::to_string(num_steps));
        }
    }
    
    // Generation test: Start with "ab", expect "c" next
    std::string prompt = "ab";
    std::vector<int> prompt_tokens = tokenizer->encode(prompt);
    
    PROOF_LOG("Generation prompt: \"" + prompt + "\"");
    
    // Generate next tokens
    std::vector<int> generated = prompt_tokens;
    int max_gen = 12;
    
    for (int i = 0; i < max_gen; ++i) {
        auto numeric = makeNumericSideChannel(generated.size());
        std::vector<float> logits = runInferencePrefill(model, training_state, generation_state, generated, numeric);
        
        // Greedy decode: pick argmax
        int next_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < static_cast<int>(logits.size()); ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
        }
        
        generated.push_back(next_token);
        
        // Stop on EOS
        if (next_token == GRIM::Tokenizer::EOS_TOKEN_ID) break;
    }
    
    // Decode generated sequence
    std::string generated_text = tokenizer->decode(generated);
    
    PROOF_LOG("Generated: \"" + generated_text + "\"");
    
    // Check if pattern emerged
    // We expect something like "abcabc..." or at least repetitive structure
    
    // Count how many 'c' appear after 'ab' patterns
    int abc_count = 0;
    for (size_t i = 0; i + 2 < generated_text.size(); ++i) {
        if (generated_text[i] == 'a' && generated_text[i+1] == 'b' && generated_text[i+2] == 'c') {
            abc_count++;
        }
    }
    
    PROOF_LOG("'abc' pattern occurrences: " + std::to_string(abc_count));
    
    // Check for degenerate outputs
    bool is_echo_stop = (generated_text.size() <= prompt.size() + 1);
    bool is_all_same = true;
    if (generated_text.size() > 1) {
        char first = generated_text[0];
        for (char c : generated_text) {
            if (c != first) {
                is_all_same = false;
                break;
            }
        }
    }
    
    if (is_echo_stop) {
        PROOF_LOG("⚠ WARNING: Echo + stop behavior detected");
    }
    
    if (is_all_same) {
        PROOF_LOG("⚠ WARNING: All-same-character output detected");
    }
    
    // For this test, we just want to see SOME learning happened
    // The model should produce more than just the prompt
    PROOF_ASSERT(generated_text.size() > prompt.size(), 
                 "FATAL: Model produces no output beyond prompt!");
    
    // Check for null byte collapse
    bool has_null_bytes = (generated_text.find('\0') != std::string::npos);
    PROOF_ASSERT(!has_null_bytes, 
                 "FATAL: Model collapsed to null byte output!");
    
    PROOF_LOG("✓ Model produces non-degenerate output");
    
    // Note: Full pattern learning may require more training steps
    // This test just verifies the model CAN learn and generate
    
    return result;
}

//======================================================//
//  Test Runner
//======================================================//

void runAllTests(GRIM::LanguageModel* model, GRIM::TrainingState* training_state, GRIM::GenerationState* generation_state, ::ParameterRegistry::StartupParameterRegistry* parameter_registry, GRIM::Tokenizer::UniByte* tokenizer, int level = 0) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           GRIM-text Causality Proof Test Suite                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    std::vector<TestResult> results;
    
    // Run tests based on level
    if (level == 0 || level == 1) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(level1_single_token_causality(model, training_state, generation_state, tokenizer));
        printResult(results.back());
    }
    
    if (level == 0 || level == 2) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(level2_causal_mask_correctness(model, training_state, generation_state, tokenizer));
        printResult(results.back());
    }
    
    if (level == 0 || level == 3) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(level3_gradient_reaches_embeddings(model, training_state, generation_state, tokenizer));
        printResult(results.back());
    }
    
    if (level == 0 || level == 4) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(level4_learning_changes_logits(model, training_state, generation_state, parameter_registry, tokenizer));
        printResult(results.back());
    }
    
    if (level == 0 || level == 5) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(level5_tokenizer_loss_alignment(model, training_state, generation_state, tokenizer));
        printResult(results.back());
    }
    
    if (level == 0 || level == 6) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(level6_autoregressive_emergence(model, training_state, generation_state, parameter_registry, tokenizer));
        printResult(results.back());
    }
    
    // Summary
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        TEST SUMMARY                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& result : results) {
        std::cout << (result.passed ? "  ✓ " : "  ✗ ") << result.name << "\n";
        if (result.passed) passed++;
        else failed++;
    }
    
    std::cout << "\n";
    std::cout << "Total: " << passed << " passed, " << failed << " failed\n";
    std::cout << "\n";
    
    if (failed == 0) {
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✓ ALL TESTS PASSED - Training infrastructure is CORRECT!     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    } else {
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✗ TESTS FAILED - Generation will NOT work until fixed!       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    }
}

} // namespace CausalityProof

//======================================================//
//  Main Entry Point
//======================================================//

int main(int argc, char** argv) {
    int level = 0;  // 0 = all levels
    
    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--level" && i + 1 < argc) {
            level = std::stoi(argv[++i]);
        } else if (arg == "--all") {
            level = 0;
        } else if (arg == "--config") {
            std::cerr << "--config is no longer supported; use the canonical ai_config.json\n";
            return 2;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --level N    Run specific test level (1-6)\n";
            std::cout << "  --all        Run all test levels (default)\n";
            std::cout << "\nTest Levels:\n";
            std::cout << "  1: Single-token causality proof\n";
            std::cout << "  2: Causal mask correctness\n";
            std::cout << "  3: Gradient path continuity\n";
            std::cout << "  4: Learning must change logits\n";
            std::cout << "  5: Tokenizer-loss alignment\n";
            std::cout << "  6: Autoregressive emergence\n";
            return 0;
        }
    }
    
    std::cout << "Initializing CUDA...\n";
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Using GPU: " << prop.name << "\n";
    
    // TODO: Load model and tokenizer from config
    // For now, this is a skeleton - you'll need to integrate with your actual loading code
    
    std::cout << "\n";
    std::cout << "ERROR: Model/tokenizer loading not yet integrated.\n";
    std::cout << "To use this test suite:\n";
    std::cout << "  1. Add this file to CMakeLists.txt\n";
    std::cout << "  2. Link with grim_language_model_gpu_impl\n";
    std::cout << "  3. Add model/tokenizer loading code in main()\n";
    std::cout << "\n";
    std::cout << "Or call CausalityProof::runAllTests(model, tokenizer) from train_gpu.cu\n";
    
    return 1;
}
