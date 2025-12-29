#!/usr/bin/env python3
"""
Trace why POST-FORWARD loss=0.000000 for batch 1 in GRIM-text training.
"""

def main():
    print("=" * 80)
    print("GRIM-text Training: Why POST-FORWARD loss = 0.000000")
    print("=" * 80)
    print()
    
    print("📊 Log Evidence:")
    print("   [2025-12-17 14:25:37] [Batch 1/3404] size=3 len=5-6 eff=88%")
    print("   [2025-12-17 14:25:37] [GradTrace] BATCH_INFO batch=1 seqs=[333,4589,6297] lens=[5,5,6]")
    print("   [2025-12-17 14:25:37] [GradTrace] POST-FORWARD loss=0.000000  ← WHY ZERO?")
    print()
    
    print("=" * 80)
    print("REASON: training_state_.initialized Check Fails")
    print("=" * 80)
    print()
    print("The loss computation has an early-exit guard that returns 0.0f if")
    print("training state is not initialized. This happens on the FIRST batch.")
    print()
    
    print("=" * 80)
    print("CODE PATH: Forward Pass → Loss Computation → Early Exit")
    print("=" * 80)
    print()
    
    steps = [
        {
            "step": 1,
            "file": "Phase2_TrainingLoop.cu:789",
            "operation": "Call Forward Pass",
            "code": "result.loss = ctx.model->computeLossBatch(batch_inputs, batch_targets);",
            "description": "Batch 1 forward pass initiated",
        },
        {
            "step": 2,
            "file": "TrainingOps.cu:49-54",
            "operation": "Check Training State",
            "code": """float LanguageModel::computeLoss(...) {
    if (!training_state_.initialized) {
        return 0.0f;  // ← EARLY EXIT!
    }
    ...""",
            "description": "Guard clause catches uninitialized state",
        },
        {
            "step": 3,
            "file": "Phase2_TrainingLoop.cu:795",
            "operation": "Log Zero Loss",
            "code": """ctx.logging.logger->log("[GradTrace] POST-FORWARD loss=" + 
                        Internal::formatScalar(result.loss, 6));""",
            "description": "Logs the returned 0.0f to file",
        },
    ]
    
    for step_info in steps:
        print(f"STEP {step_info['step']}: {step_info['operation']}")
        print(f"File: {step_info['file']}")
        print(f"Code:")
        for line in step_info['code'].split('\n'):
            print(f"  {line}")
        print(f"→ {step_info['description']}")
        print()
    
    print("=" * 80)
    print("ALTERNATIVE EXPLANATION: First Batch Before State Setup")
    print("=" * 80)
    print()
    print("SCENARIO A: training_state_.initialized = false (early exit)")
    print("  - Phase1_Startup.cu:576 calls initTrainingState()")
    print("  - But training_state_.initialized flag might not be set yet")
    print("  - TrainingOps.cu:51-53 checks flag → returns 0.0f")
    print()
    print("SCENARIO B: Actually computes loss but gets exactly 0.0")
    print("  - Forward pass runs with randomly initialized weights")
    print("  - Xavier initialization: weights ~ N(0, sqrt(2/(fan_in+fan_out)))")
    print("  - logits for batch 1 happen to produce loss=0.0 (extremely rare)")
    print()
    print("SCENARIO C: Loss computation fails validation")
    print("  - ComputeLossHost_GPU.cu:131-146 validates inputs")
    print("  - If validation fails, returns result{} with average_loss=0.0f")
    print()
    
    print("=" * 80)
    print("DETAILED LOSS COMPUTATION FLOW")
    print("=" * 80)
    print()
    
    loss_steps = [
        {
            "step": 1,
            "file": "TrainingOps.cu:63",
            "code": "auto logits = forwardWithCache(input_ids);",
            "desc": "Forward pass: embeddings → encoder layers → logits",
        },
        {
            "step": 2,
            "file": "TrainingOps.cu:89-132",
            "code": "const auto loss_result = computeLossHost(loss_inputs, scratch);",
            "desc": "Prepare LossComputationInputs and call loss pipeline",
        },
        {
            "step": 3,
            "file": "ComputeLossHost_GPU.cu:154-161",
            "code": """std::cout << "[ComputeLossHost] loss pipeline invoked (batch="
          << ctx.batch_size << ", seq=" << ctx.seq_len
          << ", tokens=" << total_tokens << ")" << std::endl;""",
            "desc": "Log loss pipeline invocation (first time only)",
        },
        {
            "step": 4,
            "file": "ComputeLossHost_GPU.cu:163-169",
            "code": """Loss::DeviceBuffers buffers{};
buffers.token_losses = scratch.loss_values;
buffers.scratch = scratch.loss_accumulator;
buffers.grad_logits = inputs.grad_logits;
const auto breakdown = Loss::launchLossPipeline(ctx, inputs.config, buffers);""",
            "desc": "Launch CUDA kernel: UnifiedLoss (Focal + LabelSmoothing + CE)",
        },
        {
            "step": 5,
            "file": "UnifiedLoss_GPU.cu:375-450",
            "code": "UnifiedLossTelemetry computeUnifiedLoss(...)",
            "desc": "Compute focal loss with label smoothing on GPU",
        },
        {
            "step": 6,
            "file": "ComputeLossHost_GPU.cu:172",
            "code": "result.average_loss = result.total_loss / static_cast<float>(valid_tokens);",
            "desc": "Average loss over valid tokens",
        },
        {
            "step": 7,
            "file": "ComputeLossHost_GPU.cu:174-177",
            "code": """if (!std::isfinite(result.total_loss) || !std::isfinite(result.average_loss)) {
    std::cerr << "ComputeLossHost: non-finite loss detected" << std::endl;
    return result;  // result.success = false, average_loss = 0.0f
}""",
            "desc": "Check for NaN/Inf → early exit with 0.0f if invalid",
        },
        {
            "step": 8,
            "file": "TrainingOps.cu:142-149",
            "code": """if (!std::isfinite(loss_result.average_loss)) {
    ForwardLog::warn("[computeLoss] ⚠️  INVALID LOSS DETECTED!");
    ...
}""",
            "desc": "Additional validation in computeLoss()",
        },
        {
            "step": 9,
            "file": "TrainingOps.cu:162",
            "code": "return loss_result.average_loss;",
            "desc": "Return final loss value (0.0f if any check failed)",
        },
    ]
    
    print("FULL LOSS PIPELINE:")
    print("-" * 80)
    for step_info in loss_steps:
        print(f"{step_info['step']}. {step_info['file']}")
        print(f"   Code: {step_info['code'][:80]}...")
        print(f"   → {step_info['desc']}")
        print()
    
    print("=" * 80)
    print("LIKELY ROOT CAUSE")
    print("=" * 80)
    print()
    print("Most probable explanation:")
    print("  1. TrainingOps.cu:51-53 checks training_state_.initialized")
    print("  2. On FIRST batch, this flag might still be false")
    print("  3. Function returns 0.0f without computing loss")
    print("  4. Batch 2 onwards: loss is computed normally (e.g., 10.169053)")
    print()
    print("Evidence from log:")
    print("  • Batch 1: loss=0.000000  ← Zero (likely early exit)")
    print("  • Batch 2: loss=10.169053 ← Normal loss value")
    print("  • Batch 3: loss=9.848621  ← Normal")
    print("  • Batch 4: loss=9.857855  ← Normal")
    print()
    print("Alternative: If initialized=true, then")
    print("  • validation in ComputeLossHost_GPU.cu:135-148 failed")
    print("  • OR launchLossPipeline returned total_loss=0.0")
    print("  • Check for errors in console output or stderr")
    print()
    
    print("=" * 80)
    print("WHERE training_state_.initialized IS SET")
    print("=" * 80)
    print()
    print("File: InitTrainingState.cu")
    print("Function: LanguageModel::initTrainingState()")
    print()
    print("Setting locations:")
    print("  1. InitTrainingState.cu:27 → Function entry")
    print("  2. InitTrainingState.cu:~540 → training_state_.initialized = true;")
    print()
    print("Timeline:")
    print("  • Phase1_Startup.cu:576 → model->initTrainingState()")
    print("  • InitTrainingState.cu:540 → training_state_.initialized = true")
    print("  • Phase2_TrainingLoop.cu:789 → model->computeLossBatch() [batch 1]")
    print("  • TrainingOps.cu:51 → if (!training_state_.initialized) return 0.0f")
    print()
    print("If batch 1 loss=0.0, then:")
    print("  • Either initialized flag not set yet (timing issue)")
    print("  • Or loss computation validation failed")
    print()
    print("=" * 80)
    print("DEBUGGING RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("To diagnose:")
    print("  1. Add log in TrainingOps.cu:51:")
    print("     if (!training_state_.initialized) {")
    print("         std::cout << \"[DEBUG] computeLoss early exit (not initialized)\" << std::endl;")
    print("         return 0.0f;")
    print("     }")
    print()
    print("  2. Check InitTrainingState.cu:540 sets flag:")
    print("     std::cout << \"[DEBUG] training_state_.initialized = true\" << std::endl;")
    print()
    print("  3. Check ComputeLossHost_GPU.cu validation logs:")
    print("     Search console for \"ComputeLossHost: invalid\" error messages")
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
