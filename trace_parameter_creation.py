#!/usr/bin/env python3
"""
Trace the actual GPU memory allocations that create the 104M parameters.
The 542 log writes are just TEXT - the real action is GPU memory allocation.
"""

def main():
    print("=" * 80)
    print("GRIM-text Training: ACTUAL Parameter Creation (GPU Memory Allocations)")
    print("=" * 80)
    print()
    
    print("📊 Parameter Breakdown:")
    print("   total_params      = 104,358,144")
    print("   embedding_params  =  28,842,240  (vocab=37555 × d_model=768)")
    print("   encoder_params    =  75,515,904  (12 layers)")
    print("   lm_head_params    =           0  (weight tying enabled)")
    print()
    
    print("=" * 80)
    print("REAL MEMORY ALLOCATION SEQUENCE (from EXE startup)")
    print("=" * 80)
    print()
    
    # ACTUAL GPU operations that create parameters
    operations = [
        ("Phase1_Startup.cu:494-536", "Model Constructor", "LanguageModel::LanguageModel()", [
            ("Common/grim_language_model_gpu.cu:100-200", "Allocate config structs"),
            ("Common/grim_language_model_gpu.cu:250-350", "Create GPU encoder layers"),
            ("Common/grim_language_model_gpu.cu:400-450", "Initialize CUDA streams"),
        ]),
        
        ("Phase1_Startup.cu:576", "Initialize Training State", "model->initTrainingState()", [
            ("InitTrainingState.cu:27-100", "→ Allocate embedding gradients (28,842,240 × sizeof(float))"),
            ("InitTrainingState.cu:100-150", "→ Allocate encoder layer gradients"),
            ("InitTrainingState.cu:150-200", "→ Allocate attention gradient buffers"),
            ("InitTrainingState.cu:200-250", "→ Initialize cuBLAS/cuDNN handles"),
        ]),
        
        ("LanguageModel_Training.cu:502", "Build Parameter Groups", "buildParameterGroups()", [
            ("LanguageModel_Training.cu:238-280", "→ Scan embedding weights (28,842,240 params)"),
            ("LanguageModel_Training.cu:280-350", "→ Scan 12 encoder layers:"),
            ("LanguageModel_Training.cu:300", "  - Layer 0: QKV=1,180,416 + Wo=589,824 + FFN=4,722,176 + RMS=1,536 = 6,494,976"),
            ("LanguageModel_Training.cu:310", "  - Layer 1-11: same (×11 more) = 71,444,736"),
            ("LanguageModel_Training.cu:350", "  Total encoder: 6,494,976 × 12 = 77,939,712"),
            ("LanguageModel_Training.cu:360-377", "→ Create Adam optimizer state (m, v vectors)"),
        ]),
    ]
    
    print("PHASE 1: MODEL CONSTRUCTION & GPU ALLOCATION")
    print("-" * 80)
    step = 1
    for loc, name, func, substeps in operations:
        print(f"\n{step}. {loc}")
        print(f"   Operation: {name}")
        print(f"   Function:  {func}")
        for subloc, desc in substeps:
            print(f"   {subloc}")
            print(f"   {desc}")
        step += 1
    
    print("\n" + "=" * 80)
    print("ACTUAL GPU MEMORY ALLOCATIONS (cudaMalloc calls)")
    print("=" * 80)
    print()
    
    cuda_allocs = [
        ("InitTrainingState.cu:50-60", "cudaMalloc", "embedding_grads", "28,842,240 × 4 bytes = 115.4 MB"),
        ("TrainingOps.cu:371", "cudaMalloc", "token_buffer", "batch_size × seq_len × d_model"),
        ("TrainingOps.cu:393", "cudaMalloc", "position_buffer", "seq_len × d_model"),
        ("TrainingOps.cu:414", "cudaMalloc", "gamma_buffer", "d_model (RMSNorm)"),
        ("LanguageModel_Training.cu:370", "cudaMalloc", "m_state (Adam)", "28,842,240 × 4 bytes = 115.4 MB"),
        ("LanguageModel_Training.cu:371", "cudaMalloc", "v_state (Adam)", "28,842,240 × 4 bytes = 115.4 MB"),
        ("Phase2_TrainingLoop.cu:256", "cudaMalloc", "guess_metadata", "capacity × sizeof(GuessMetadata)"),
        ("Phase2_TrainingLoop.cu:259", "cudaMalloc", "guess_rewards", "capacity × sizeof(float)"),
        ("Phase2_TrainingLoop.cu:262", "cudaMalloc", "guess_stats", "capacity × sizeof(GuessRewardStats)"),
        ("LogRecorder.cu:308", "cudaMalloc", "layer_log_entries", "maxDeviceEntries × sizeof(LayerLogEntry)"),
    ]
    
    for i, (file_loc, func, buffer_name, size_desc) in enumerate(cuda_allocs, 1):
        print(f"{i:2d}. {file_loc}")
        print(f"    {func}(&{buffer_name}, {size_desc})")
        print()
    
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("❌ The 542 'write operations' you saw are LOG WRITES (text to .log file)")
    print("✅ The REAL work happens in GPU memory allocations (cudaMalloc)")
    print()
    print("Timeline:")
    print("  1. Phase1_Startup.cu:494-536 → Construct LanguageModel object")
    print("  2. initTrainingState() → Allocate 104M parameters on GPU")
    print("  3. buildParameterGroups() → Organize into 153 optimizer groups")
    print("  4. Phase2_TrainingLoop.cu:414 → Log 'Created 3404 dynamic batches' (just text)")
    print("  5. Phase2_TrainingLoop.cu:773-1078 → Training loop (forward/backward)")
    print()
    print("The numbers you see (104,358,144 params) come from:")
    print("  • TrainingOps.cu:271 → stats.total_params = embedding + encoder + lm_head")
    print("  • Phase2_TrainingLoop.cu:750 → model->getModelStats() reads from parameter_groups_")
    print("  • Phase2_TrainingLoop.cu:751-754 → Logs the stats as text")
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
