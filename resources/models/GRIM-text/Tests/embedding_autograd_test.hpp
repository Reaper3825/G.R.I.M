//======================================================//
//  embedding_autograd_test.hpp
//  Comprehensive autograd-enabled embedding test suite
//  with verbose logging (13 tests total)
//======================================================//

#pragma once

#include <string>

namespace GRIM {
namespace Test {

//======================================================//
//  Core Autograd Tests (4 tests)
//======================================================//

/**
 * Test 1: Basic autograd forward pass
 * Verifies embedding lookup works with Tensor objects
 * and that requires_grad flag is properly set.
 */
bool testAutogradForwardBasic(std::string& message);

/**
 * Test 2: Autograd backward with loss
 * Tests gradient computation through loss function
 * and gradient accumulation via Tensor.backward().
 */
bool testAutogradBackwardWithLoss(std::string& message);

/**
 * Test 3: Finite difference gradient verification
 * Compares autograd gradients against numerical gradients
 * computed via finite differences (max relative error check).
 */
bool testFiniteDifferenceVerification(std::string& message);

/**
 * Test 4: Integration with TrainingState
 * Verifies TrainingState.tensors_ initialization and that
 * embedding weights are registered with autograd system.
 */
bool testTrainingStateIntegration(std::string& message);

//======================================================//
//  Position & Weight Tying Tests (2 tests)
//======================================================//

/**
 * Test 5: Position embedding gradients
 * Tests position embeddings receive correct gradients
 * and that unused positions have zero gradient.
 */
bool testPositionEmbeddingGradients(std::string& message);

/**
 * Test 6: Weight tying stress test (Issue #22 regression)
 * Verifies gradient accumulation when embedding/LM-head share
 * same buffer (tied weights). Tests atomicAdd correctness.
 */
bool testWeightTyingStressTest(std::string& message);

//======================================================//
//  Stress & Performance Tests (1 test)
//======================================================//

/**
 * Test 7: High contention atomicAdd (4096-way)
 * Tests 4096 positions all updating same token gradient
 * to verify atomicAdd determinism under extreme contention.
 */
bool testHighContentionAtomicAdd(std::string& message);

//======================================================//
//  Integration Tests (2 tests)
//======================================================//

/**
 * Test 8: RMSNorm integration
 * Tests embedding + RMSNorm forward/backward pipeline
 * and verifies gradients flow correctly through layers.
 */
bool testRMSNormIntegration(std::string& message);

/**
 * Test 9: Real GRMT data integration
 * Loads UnigramByte tokenizer and processes real text
 * to verify end-to-end training data compatibility.
 */
bool testRealGRMTDataIntegration(std::string& message);

//======================================================//
//  Edge Case Tests (4 tests)
//======================================================//

/**
 * Test 10: Out of bounds token ID (Rule 20 Fail Loud)
 * Tests behavior with invalid token IDs to verify
 * Rule 20 compliance or graceful error handling.
 */
bool testOutOfBoundsTokenID(std::string& message);

/**
 * Test 11: Empty sequence
 * Tests seq_len=0 edge case to verify no crashes
 * and graceful handling of empty input.
 */
bool testEmptySequence(std::string& message);

/**
 * Test 12: Very long sequence (8192 tokens)
 * Stress test with max sequence length to verify
 * stability, performance, and no NaN/Inf generation.
 */
bool testVeryLongSequence(std::string& message);

/**
 * Test 13: Batch independence
 * Verifies batched forward pass produces identical results
 * to separate single-batch runs (determinism check).
 */
bool testBatchIndependence(std::string& message);

}  // namespace Test
}  // namespace GRIM
