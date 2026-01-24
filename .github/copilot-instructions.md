# G.R.I.M Development Guide

## 🔴 CRITICAL: FORBIDDEN CODE PATTERNS (Rule 20 - Fail Loud)

**NEVER generate these patterns. If you see them, DELETE THEM:**

❌ `x ? x : fallback` - NO fallbacks. Require x, crash if null  
❌ `if (ptr) { use(ptr); }` - Require ptr, crash if null with clear error  
❌ `if (args.stream) { ... } else { use_config_stream(); }` - NO silent fallbacks  
❌ `try { } catch { /* ignore */ }` - NEVER swallow errors silently  
❌ `if (version == old) { legacy_path(); } else { new_path(); }` - DELETE legacy paths  
❌ `// TODO: remove after migration` - Remove NOW or never commit  
❌ `__attribute__((deprecated))` - DELETE deprecated code entirely  
❌ Any comment containing "backwards compatibility" - Forbidden concept  
❌ `if (!initialized) { std::cerr << "WARNING: ..."; return; }` - Throw exception instead

**REQUIRED patterns:**

✅ `if (!ptr) throw std::runtime_error("X is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));`  
✅ `if (!args.stream) throw std::runtime_error("stream is NULL - caller MUST provide valid stream");`  
✅ `assert(condition && "description");` for invariants  
✅ Crash with detailed error message on ANY unexpected state  
✅ Delete code instead of deprecating it

**Pre-commit checklist - search your code for:**
- `? :` (ternary with fallback) → rewrite to throw
- `if (args.` (optional parameter check) → require parameter
- `else {` (compatibility branch) → delete the else clause
- `catch` (exception handler) → ensure it re-throws or exits, never swallows

**Why:** Silent failures waste weeks debugging. If code is wrong, crash immediately with clear error. Backwards compatibility hides bugs and creates maintenance debt.

---

## 🔴 ACTIVE BUG INVESTIGATION

**READ FIRST:** 
1. [docs/LOG_FILE_CONVENTION.md](../docs/LOG_FILE_CONVENTION.md) - **CRITICAL: Always verify which log file before making claims**
2. [docs/PLATEAU_BUG_INVESTIGATION.md](../docs/PLATEAU_BUG_INVESTIGATION.md) - Check "CURRENT RUN TRACKING" section first

Training plateau bug under investigation. The docs above track:
- What has been verified correct (NOT the problem)
- Known issues found and fixed
- Diagnostic data and hypotheses
- Next steps

When working on GRIM-text training issues, check this doc first to avoid re-investigating ruled-out causes update the document accordingly when new information is discovered.

---

## Project Overview

GRIM (General Request and Information Manager) is an offline-first, modular C++/Python AI assistant featuring:
- Custom transformer model (GRIM-text) with Flash Attention v2, cuBLAS, custom fused CUDA kernels
- Multi-modal perception (screen capture, OCR, object detection)
- Voice I/O (Whisper.cpp STT, Coqui XTTS v2 via Python bridge)
- Hot-reloadable plugin architecture
- GPU-accelerated training and inference (CUDA, GELU activation, FlatBuffers serialization)
- DeBERTa BERT model for training data quality verification

**Core Philosophy:** All features work offline by default. Only browser commands and external APIs require internet.

**Project Status:** Personal Jarvis-style assistant, not intended for distribution. Optimized for single-user experience.

**Planned:** Multi-model orchestration system, VR/Quest headset overlay mode

## Architecture Overview

### Main Components

**C++ Core** ([main.cpp](main.cpp))
- `ai/` - Intent classification, task planning, GRIM-text integration
- `commands/` - Command routing and execution (see [commands/commands_core.hpp](commands/commands_core.hpp))
- `memory/` - FlatBuffer-serialized memory storage (unified schema in [memory/unified_memory.hpp](memory/unified_memory.hpp))
- `nlp/` - Pattern matching and rule-based NLP
- `perception/` - Screen capture, OCR (Tesseract), visual context caching
- `voice/` - Whisper.cpp integration for STT
- `wake/` - Porcupine wake word detection
- `ui/` - BGFX-based console UI

**Python Bridges** (`resources/python/`)
- `coqui_bridge.py` - TTS synthesis (XTTS v2)
- `osit_bridge.py` - Sherlock OSINT integration

**GRIM-text Model** (`resources/models/GRIM-text/`)
- Custom transformer with GPU-accelerated training/inference
- **ScratchBlock Reasoning Layer** - Internal structured reasoning with atom detection (numbers, URLs, emails, paths, dates, code literals)
- **Unigram + Byte Fallback Tokenizer** - High-quality subword segmentation with 100% UTF-8 coverage
  - Token layout: [0-255] = bytes, [256-511] = atom placeholders, [512+] = unigram vocab
- **Quantization Support** - Int8/FP16 activation quantization for inference
- **Personality Layer** - RL-weighted response selection governed by reward learning
- Separate build from main GRIM executable
- Runs as HTTP server (`grim_text_server.exe`) on port 11435
- Training controlled via `training_control_server.exe` on port 11436

### Key Integration Points

**Backend Configuration** ([ai_config.json](ai_config.json))
```json
{
  "backend": "grim_native",  // Uses GRIM-text instead of Ollama/OpenAI
  "grim_text_url": "http://127.0.0.1:11435"
}
```

**Server Managers**
- `GRIM::startGRIMTextServer()` - Launches inference server ([ai/grim_text_server_manager.hpp](ai/grim_text_server_manager.hpp))
- `GRIM::startTrainingServer()` - Launches training control server ([ai/training_server_manager.hpp](ai/training_server_manager.hpp))
- Both auto-start in [bootstrap/bootstrap.cpp](bootstrap/bootstrap.cpp), auto-stop on Ctrl+C

## Build System

**CMake Configuration**
```bash
# Use presets for vcpkg + CUDA integration
cmake --preset=release
cmake --build --preset=release

# Or traditional approach
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

**Key Build Options** ([cmake/Options.cmake](cmake/Options.cmake))
- `GRIM_USE_CUDA=ON` - GPU acceleration for Whisper/GRIM-text
- `GRIM_USE_PERCEPTION=ON` - Enable vision/OCR features
- `GRIM_PORTABLE_ONLY=OFF` - Resources in fixed paths vs portable layout

**Critical Build Notes**
- Main GRIM executable: `GRIM.exe`
- GRIM-text builds separately in `resources/models/GRIM-text/training/`
- Core plugin (`plugins/core_plugin.cpp`) compiles directly into GRIM.exe
- Hot-reloadable plugins (osint_plugin, network_test_plugin) build as DLLs

## Development Workflows

### Training Data Pipeline

**Data Collection** (`DataCollection/`)
1. **Web Collection** - Scrapes text from academic papers, technical docs, Wikipedia, GitHub
2. **BERT Verification** - DeBERTa v3 model (`resources/models/GRIM-text/quality/deberta-v3-base-mnli/`) validates quality
   - Language detection (English word frequency)
   - Spam pattern detection
   - Duplicate content filtering (hash-based)
   - UI artifact removal
   - Encoding error detection
3. **HTML Cleaning** - DataLoader.cu automatically strips HTML tags/entities from training texts
   - `stripHtmlTags()` - Removes `<tag>` patterns via std::regex
   - `decodeHtmlEntities()` - Converts `&lt;`, `&gt;`, `&amp;`, `&nbsp;`, etc.
   - `normalizeWhitespace()` - Collapses multiple spaces, trims edges
   - Filters texts shorter than 20 characters after cleaning
4. **Data Splits** - Splits into train/val/test sets
5. **Binary Format** - Converts to `.grmt` FlatBuffer format for training

**Training Workflow**
```bash


# 1. Start training control server (port 11436)
cd control/build/Release
.\training_control_server.exe

# 2. Monitor training via WebSocket (port 9002) or training_status.fb
# Training metrics: resources/models/GRIM-text/training/training_status.fb
```

**Training Features**
- **Flash Attention v2** - O(N) memory complexity for long sequences (512+ tokens)
- **Custom Fused Kernels** - Combined QKV projection, FFN+GELU, LayerNorm
- **Unified Loss System** - Single kernel combining focal loss + label smoothing with telemetry
- **TelemetryLattice** - Hierarchical streaming statistics (8 levels, 5 metric streams)
- **GPU Accelerated** - Full backward pass on CUDA (cuBLAS for matrix ops)
- **AdamW Optimizer** - Decoupled weight decay, AMSGrad variant support
- **FlatBuffers** - Zero-copy checkpoint serialization (8.4x faster loading)

**Three-Phase Training Architecture**
Training loop refactored into debuggable phases ([resources/models/GRIM-text/training/Phases/](resources/models/GRIM-text/training/Phases/)):

- **Phase 1: Startup** ([Phase1_Startup.cu/hpp](resources/models/GRIM-text/training/Phases/Phase1_Startup.cu))
  - Configuration loading from `ai_config.json`
  - Path validation (model, data, checkpoint dirs)
  - Tokenizer initialization (GrimTokenizer with scratch blocks)
  - Training data loading with sliding windows
  - Model initialization with Xavier weight seeding
  - Optimizer/gradient controller setup
  - Returns `TrainingContext` struct with all initialized state

- **Phase 2: Training Loop** ([Phase2_TrainingLoop.cu/hpp](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu))
  - Epoch iteration with dynamic batching
  - Batch construction (content weighting, rare token scoring)
  - Forward/backward passes with gradient accumulation
  - Gradient clipping (token-normalized + adaptive)
  - Optimizer steps with dynamic learning rate
  - Validation and checkpointing
  - Auto-stop detection (plateau, high loss)
  - Soft restart and micro-validation

- **Phase 3: Cleanup** ([Phase3_Cleanup.cu/hpp](resources/models/GRIM-text/training/Phases/Phase3_Cleanup.cu))
  - Final model checkpoint save
  - Training summary generation
  - Status file updates (FlatBuffer)
  - Resource cleanup (GPU/CPU)

**Orchestrator** ([train_gpu_orchestrator.cu](resources/models/GRIM-text/training/train_gpu_orchestrator.cu))
- Simple `main()` that calls `executePhase1()` → `executePhase2()` → `executePhase3()`
- Clear phase boundaries for debugging
- Data flow via `TrainingContext` struct (no globals)

**Benefits:**
- Each phase is a separate compilation unit → faster incremental builds
- Set breakpoints at phase boundaries without navigating 3500+ line file
- Clear contracts: `TrainingContext` passed explicitly between phases
- If training fails, logs show which phase failed
- Easier to test phases independently

### Testing GRIM-text Model

```bash
# 1. Build the training executable (three-phase architecture)
cd resources/models/GRIM-text/training/TrainingLoop
cmake --build build --config Release --target train_gpu

# 2. Run training
cd ..
.\TrainingLoop\build\Release\train_gpu.exe

# 3. Build inference server (separate target)
cd TrainingLoop
cmake --build build --config Release --target grim_text_server

# 4. Test inference
cd ..
python test_grim_model.py  # Automated health checks

# 5. Monitor training
# Training status stored as FlatBuffer at:
# resources/models/GRIM-text/training/training_status.fb
```

### Adding Commands

Commands registered in [commands/commands_core.hpp](commands/commands_core.hpp):
```cpp
extern std::unordered_map<std::string, CommandFunc> commandMap;

// In implementation file:
commandMap["mycommand"] = [](const std::string& arg) -> CommandResult {
    return {true, "Response", Intent::INFORM};
};
```

### Hot-Reloadable Plugins

Example: [plugins/osint_plugin/osint_plugin.cpp](plugins/osint_plugin/osint_plugin.cpp)
- Export `plugin_init()`, `plugin_commands()`, `plugin_unload()`
- Place DLL in `plugins/` directory
- GRIM checks for file changes every 2 seconds
- See [core/plugin_manager.hpp](core/plugin_manager.hpp)

### Memory System

**RL-Gated Dual Memory Architecture**
- **Short-Term Memory** - Recent context window (max 50 items), cached in RAM
- **Long-Term Memory** - Persistent storage with FlatBuffer serialization
- **RL Gating** - Reward learning system decides what gets promoted from STM → LTM
- Context decay with confidence scoring to prevent memory bloat

Uses FlatBuffers for zero-copy serialization ([memory/unified_memory_generated.h](memory/unified_memory_generated.h)):
```cpp
GRIM::UnifiedMemoryObject obj(
    GRIM::SourceType::USER_VOICE,
    GRIM::TypeTag::FACT,
    GRIM::IntentType::INFORM,
    GRIM::ContextType::CONVERSATION,
    "raw input",
    1.0f
);
auto buffer = obj.toFlatBuffer();  // Serialize

// Short-term storage (transient)
g_memoryStorage.storeShortTerm(obj);

// Long-term storage (persistent, RL-gated)
g_memoryStorage.storeLongTerm(obj);
```

**Memory Components** ([memory/](memory/))
- `memory_storage.{cpp,hpp}` - Dual-store with STM/LTM separation
- `context_manager.{cpp,hpp}` - Contextual snapshots for intent classification
- `unified_memory.{hpp,fbs}` - FlatBuffer schema and C++ API

### Python Bridge Communication

C++ launches Python with `BridgeManager` ([core/bridge_manager.hpp](core/bridge_manager.hpp)):
```cpp
BridgeManager::start("tts", "resources/python/coqui_bridge.py");
json request = {{"text", "Hello"}, {"voice", "default"}};
json response = BridgeManager::send("tts", request);
```

## GRIM-text Architecture

### ScratchBlock Reasoning Layer
**Critical Feature:** Internal structured reasoning before text generation
- Detects and tokenizes structural elements: numbers, URLs, emails, file paths, dates, code literals
- Uses **AtomTable** system with dedicated embeddings for each atom type
- Token layout: `[0-255]` = byte fallback, `[256-511]` = atom placeholders, `[512+]` = unigram vocab
- Enables model to reason about structure before converting to text
- Configurable via `ai_config.json` → `tokenizer.scratch_block_reasoning`

### Tokenization: Unigram + Byte Fallback
- **Unigram Language Model** - Statistical subword segmentation for quality (via Viterbi decoding)
- **Byte Fallback** - Raw UTF-8 bytes (0x00-0xFF) for 100% coverage of unknown chars/emojis
- **Trie-Based Lookup** - Fast prefix matching for Viterbi encoding; trie auto-built in constructor
- **Aho-Corasick DFA** - O(n) multi-pattern matching for structural token detection (50-100x faster than std::regex)
  - Detects URL prefixes (http://, https://, www., ftp://)
  - Email indicators (@, mailto:)
  - Number prefixes (-, +, digits)
- **Zero-copy design** - `std::string_view` references, no allocations during tokenization
- **Files**: `resources/models/GRIM-text/Shared/UnigramByte/`
- **Key Implementation**: GrimTokenizer.hpp (alias to UniByte), AhoCorasick.cu (pattern matching), Unigram.cu (vocab + Viterbi)
- **Test Suite**: `resources/models/GRIM-text/Tests/unigrambyte_self_test.cu` (37 tests covering Byte, Unigram, UniByte, AtomTable, Integration)

### Quantization System
- **Activation Quantization** - Int8 symmetric quantization for inference speedup
- Configurable per-layer: embeddings, encoder outputs, QKV cache, logits
- See `ai_config.json` → `training.config.activation_quantization`
- Files: `resources/models/GRIM-text/Layers/Quantization/`

### Grouped Query Attention (GQA)
**Production Implementation** - Efficient attention mechanism reducing KV cache memory
- **Configuration**: num_heads=12, num_kv_heads=4, heads_per_kv_group=3 (3:1 reduction ratio)
- **Memory Efficiency**: KV cache reduced by 3x compared to Multi-Head Attention (MHA)
- **Backward Compatibility**: Old MHA checkpoints cannot be loaded into GQA models (config validation enforces this)
- **Gradient Scaling**: GQA backward kernel applies `gqa_grad_scale = 1.0f / heads_per_kv_group` to dV, dK accumulation to prevent gradient explosion
- **Weight Layout**: W_qkv shape is `[(num_heads + 2*num_kv_heads) * head_dim, d_model]` = `[1280, 768]` vs MHA `[2304, 768]`
- **Serialization Support**: 
  - FlatBuffer schema includes `num_kv_heads` in both `ModelConfig` and `AttentionWeights` tables
  - Checkpoint save/load automatically handles GQA dimensions
  - Old checkpoints have `num_kv_heads=0` (treats as MHA); new checkpoints store actual `num_kv_heads` value
- **Files**: 
  - Forward/backward kernels: `Layers/Flash_Attention_Kernal.cu`
  - Serialization: `Layers/Serialization/Serialization_GPU.cu/hpp`
  - FlatBuffer schema: `training/schemas/grim_transformer_model.fbs`
  - Training state: `Training/TrainingState_GPU.hpp`
- **Test Results**: Training with GQA gradient scaling produces stable gradients (no explosion), loss converges normally

### Personality & RL Weighting
- **Reward Learning Integration** - Response selection weighted by RL feedback
- Pre-dispatch command suggestions based on learned user preferences
- Feedback loop: user interaction → sentiment scoring → RL model update
- Files: `Reward_Learning/grim_rl.{hpp,cpp}`
- Config: `ai_config.json` → `personality.custom_prompt`

### Unified Loss System
**Production Architecture** - Autograd-enabled loss combining focal loss + label smoothing + entropy regularization
- **Mathematical Formula**: `L = α(1-p_t)^γ * CE_smooth + λ * H(p)` where:
  - `CE_smooth = -(1-ε)log(p_t) - ε/(V-1)Σlog(p_i)` (label-smoothed cross entropy)
  - `H(p) = Σ p_i * log(p_i)` (negative entropy, penalizes low entropy/mode collapse)
  - `p_t` = probability of correct class
  - `α` = focal weight (default 1.0), `γ` = focal gamma (default 0.0 for plain CE)
  - `ε` = smoothing epsilon (default 0.0 for plain CE)
  - `λ` = entropy regularization lambda (default 0.0)
- **Gradient Formula**: Standard CE gradient `∂L/∂z = (softmax - one_hot) / N`
  - Focal/smoothing/entropy only affect loss magnitude, gradient direction unchanged
  - N = valid_count (mean reduction like PyTorch)
- **Files**: 
  - Core: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.{cu,hpp}`
  - Call Site: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`
- **Config**: `ai_config.json` → `training.config.loss`
  - `focal.enabled`, `focal.alpha`, `focal.gamma`
  - `label_smoothing.enabled`, `label_smoothing.epsilon`
  - `entropy_reg.enabled`, `entropy_reg.lambda`
- **API**: 
  - `autograd::unified_loss(logits, targets, mask, num_tokens, vocab_size, config, stream)` - Full loss
  - `autograd::cross_entropy_loss(...)` - Legacy wrapper using plain CE config
- **DELETED**: `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` (old system disconnected from autograd gradients)

### TelemetryLattice System
**Hierarchical Telemetry Tracking** - Multi-resolution streaming statistics
- **Architecture**: 8 temporal levels, exponential stride (2^k), pure GPU computation
- **Per-Level State**: 10-dimensional telemetry vector per metric stream:
  - `μ` (magnitude), `σ_tilde` (coefficient of variation), `v_σ` (volatility-of-volatility)
  - `Δ_bar` (normalized slope), `p` (directional persistence)
  - `r_out` (soft outlier gate), `ℓ_out` (regime persistence)
  - `μ_ex` (excess magnitude), `Δμ`/`Δσ` (anchor drift)
- **Metric Streams**: LOSS, GRAD_NORM_MEAN, GRAD_NORM_MAX, LEARNING_RATE, TOKENS_PER_BATCH
- **Use Cases**: 
  - Loss divergence detection (r_out spike + ℓ_out persistence)
  - Gradient explosion early warning (v_σ increase)
  - Adaptive learning rate (delta_bar trend)
  - Training regime shifts (anchor drift)
- **API**: `initTelemetryLattice()`, `updateTelemetryLattice()`, `readTelemetryVector()`
- **Files**: `resources/models/GRIM-text/Shared/Telemetry/TelemetryLattice_GPU.{cu,hpp}`
- **Integration**: UnifiedLoss outputs feed directly into lattice

## Project-Specific Conventions

### Logging & Remote Monitoring
- Use `LOG_PHASE()` for major initialization steps
- Use `LOG_DEBUG(module, message)` for debugging
- Logs written to `grim.log` via [logger.hpp](logger.hpp)
- **WebSocket Server** (port 9002) - Real-time log streaming and UI integration
  - Broadcasts all log messages to connected clients
  - Started automatically in [main.cpp](main.cpp) via `wsServer.start()`
  - See [net/websocket_server.hpp](net/websocket_server.hpp)

### Intent Classification
- Fast path: Rule-based NLP ([nlp/nlp.cpp](nlp/nlp.cpp))
- Slow path: FastClassifier ([ai/fast_classifier.cpp](ai/fast_classifier.cpp))
- LLM fallback: GRIM-text via `ai_interpret()` ([ai/ai.cpp](ai/ai.cpp))

### Perception Context Caching
- Visual context cached for 2 seconds ([perception/perception_context.hpp](perception/perception_context.hpp))
- Prevents redundant OCR/object detection on unchanged screens
- Access via `PerceptionContext::getInstance().getContext()`

### Signal Handling
Clean shutdown of child processes (GRIM-text/training servers) on Ctrl+C:
```cpp
// Installed in main.cpp
SetConsoleCtrlHandler(consoleHandler, TRUE);
```

## Common Pitfalls

1. **vcpkg OpenAL Bug**: `CMakeLists.txt` filters out corrupted `\openal32.lib` from onnxruntime-gpu
2. **GRIM-text Paths**: Use `ai_config.json` paths section - avoid hardcoding absolute paths
3. **Memory Schema**: Use `UnifiedMemoryObject` not legacy `MemoryObject` (deprecated)
4. **Plugin Exports**: Must use `extern "C" __declspec(dllexport)` on Windows
5. **Python venv**: Activate `.venv` before running Python bridges/tests
6. **C++ Vector Invalidation**: NEVER hold references (`auto& node`) across vector mutations (`emplace_back`, `push_back`, `insert`) - reallocation invalidates all references. Always re-access elements after mutations.
7. **Training Data Quality**: HTML artifacts corrupt tokenizer vocab - always strip tags before training. DataLoader.cu now handles this automatically.
8. **Aho-Corasick Patterns**: Pattern detection automata must be built during DetectorState construction, not lazily. Structural token detection is critical for model reasoning.
9. **C++ Array Initialization**: `int arr[256] = {-1}` only sets first element to -1, rest are 0! Use `std::fill()` in constructor or explicit loop for full initialization.
10. **AtomTable Token IDs**: Token IDs include `ATOM_TOKEN_BASE` (256) offset. When accessing `entries_[]` array, subtract the base: `uint32_t idx = id - ATOM_TOKEN_BASE`.
11. **UnigramLM Trie**: Must call `buildTrie()` after adding pieces with `addPiece()` before encoding. Constructor now auto-builds trie with special tokens.
12. **GQA Checkpoint Incompatibility**: MHA and GQA models cannot share checkpoints. Serialization validates that checkpoint `num_kv_heads` matches model config. Old MHA checkpoints (num_kv_heads=0) cannot load into GQA model; must retrain or convert weights offline. Serialization logs clear error messages for mismatches.
13. **GQA Gradient Scaling**: Backward kernel MUST apply `gqa_grad_scale = 1.0f / heads_per_kv_group` to dV/dK accumulation. Without this, gradients explode (26M+ spike observed). Scaling ensures proper gradient normalization when multiple Q heads accumulate to same KV head.
14. **Data Quality vs Gradient Alignment**: High gradient alignment (99.45% cosine similarity) does NOT automatically mean low data diversity. GRIM-text training data has 6,733 unique sequences with only 10% token overlap (Jaccard), 0% duplicates, and 98-99% prefix diversity. The gradient alignment comes from batch composition strategy (`SIMILARITY_GROUPED`) or model architecture, NOT the data. Always run empirical data quality analysis before assuming data issues.

15. **Production Recommendation — Prefer Standard Attention + RMSNorm**: For production training stability, prefer standard scaled dot-product attention together with pre-norm `RMSNorm` (i.e., disable per-head L2 normalization of `Q` and `K`). To switch:
  - Set `QK_NORMALIZATION_ENABLED = false` in `resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp`.
  - Keep `RMSNorm` as the pre-normalization layer (no changes required if already in use).
  - Rationale: Standard scaled dot-product attention (score = `Q · K^T / sqrt(d)`) avoids the `1/||Q||` or `1/||K||` singularities introduced by per-head L2 normalization, and is the approach used in large production models (LLaMA, Mistral, Gemma). Use this when you prioritize robustness over the representational geometry benefits of QK-normalization.
20. **NEVER Keep Backwards Compatibility**: When removing functionality, DELETE all compatibility shims, legacy APIs, and fallback code paths. If code fails after removal, that's GOOD - it exposes misconnects and incorrect assumptions. Backwards compatibility hides bugs and creates maintenance debt. Let it fail loud and fix the root cause. Example: Removed `computeOptimalBlockSizes()` from Flash Attention - any caller should be updated to use constants directly from `HyperParameters::FLASH_ATTN_BLOCK_Q/KV`.
16. **Three-Phase Training Files**: The old monolithic `train_gpu.cu` (3569 lines) is kept as backup. The new build uses `train_gpu_orchestrator.cu` + `Phases/Phase{1,2,3}_{Startup,TrainingLoop,Cleanup}.{cu,hpp}`. If modifying training logic, edit the appropriate phase file, not the old train_gpu.cu. CMakeLists.txt in `training/TrainingLoop/` defines the build.

17. **Unified Loss System**: Use `autograd::unified_loss()` in `AutogradLoss.cu` for training loss computation. This is the ONLY loss path - it combines focal loss, label smoothing, entropy regularization, and cross-entropy into a single autograd-enabled kernel. The `cross_entropy_loss()` function is a convenience wrapper that calls `unified_loss()` with plain CE config. **DELETED**: `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` - these old modules had the bug where loss computation was disconnected from gradients.

19. **Batched Embedding Position Bug (FIXED)**: Prior to fix, when `positions=nullptr` was passed to embedding kernels, position was computed as `token_idx` (global index) instead of `token_idx % seq_len` (within-sequence position). This caused 2/3 of every batch to use wrong position embeddings (e.g., batch 1 tokens got positions 720-1439 instead of 0-719). Symptoms: loss starts at 10.5 (correct random baseline) but increases to 15-20 as sequence length grows, with catastrophic spikes (77-259) for long sequences. Fix in `Embedding_GPU.cu` now computes `pos_id = token_idx % seq_len` when positions is null.

18. **CMake Cache After Removing Files**: When removing .cu files from CMakeLists.txt, must clean CMake cache to remove stale device-link objects. Linker errors like `unresolved external symbol __fatbinwrap_*` indicate cached library still references deleted files. Fix: `Remove-Item -Recurse -Force build\CMakeFiles\grim_training_kernels.dir` then rebuild, or use `cmake --build build --config Release --clean-first`.

60. **PCGrad for Tied Embedding/LM Head Weights (CRITICAL FIX Jan 2026)**: When `tie_embeddings=true`, the LM head backward and embedding backward produce **OPPOSITE** gradients that cancel to ZERO! The diagnostic showed: `LM_HEAD: sum=+4.05, EMBEDDING: sum=-4.05, COMBINED: sum=0.00, COSINE=-1.0 (CANCELING)`. 
  - **Root Cause**: LM head wants token prediction probability to INCREASE, embedding wants input representation to DECREASE. With shared weights, these are opposite directions!
  - **Fix**: PCGrad (Projecting Conflicting Gradients) - `g_final = g_lm + (g_emb - proj_{g_lm}(g_emb))`. This projects out the conflicting component, keeping the LM head direction + any orthogonal embedding information.
  - **Implementation**: `kernel_pcgrad_combine` in `TensorContract_GPU.cu`, `pcgrad_temp_buffer` in TrainingState, allocated in Phase1_Startup.cu when `tie_embeddings=true`.
  - **Effect**: When cosine=-1 (opposing): uses g_lm only. When cosine=0 (orthogonal): uses g_lm + g_emb. When cosine=+1 (aligned): avoids double-counting.

21. **Weight Tying Grad Buffer Aliasing**: When `tie_embeddings=true`, `embedding_grads` and `lm_head_weight_grads` are the **same pointer** (aliased). NEVER:
  - Register both in GradAccumulationController (double-zeroing)
  - Add both to parameter_groups_ (double optimizer update)
  - Free both in destructor (double-free crash)
  - The ownership flags `lm_head_weights_owned` and pointer comparison `embedding_grads == lm_head_weight_grads` track aliasing.
  - **NOTE**: With PCGrad fix (Issue #60), embedding backward writes to TEMP buffer first, then orthogonal component is added to shared grad buffer.

44. **per_token_grad_scale is REQUIRED (NOT a bug)**: When loss is averaged (`loss_mean = sum(losses) / valid_tokens`), the backward pass gradient is already scaled by `1/valid_tokens`. The `per_token_grad_scale=true` setting in `ai_config.json` is MANDATORY to ensure gradients match this scaling. DO NOT disable this thinking "tiny gradients" is a bug - gradient RMS of ~1e-6 is CORRECT when `valid_tokens ~ 3000`. The tiny gradients combine correctly in AdamW. Disabling this causes gradient magnitude explosion (effective LR becomes 3000x larger).

38. **Issue #38 WAS WRONG - SUPERSEDED BY #60**: The original "fix" that SKIPPED embedding backward was wrong. The CORRECT fix is PCGrad (Issue #60) which preserves both gradient sources without cancellation.

43. **Encoder Weight Gradient Centering (Issue #43, Jan 2026)**: Encoder backward uses cached activations (`cached_ln1_output`, `cached_ffn_input`, `cached_ffn_hidden`, `cached_attn_output`) for weight gradient GEMMs. These activations have NON-ZERO MEAN, creating systematic gradient bias: `grad_W[i,j] = Σ_t ((centered[t,i] + mean_t) × grad[t,j])` - the mean term doesn't cancel. Fix in `BackwardPhase2_Encoder.cu`: Added `centerActivationsKernel` to center each cached activation BEFORE weight gradient GEMMs (grad_W_qkv, grad_W_o, grad_W1, grad_W2). Uses `centered_activation_scratch` buffer allocated in TrainingState. This matches the Issue #37 pattern applied to LM head.

22. **Centralized Controller Pattern - MANDATORY**: All GPU resource management MUST go through centralized controllers in TrainingState. VIOLATIONS ARE BUGS:
  - **CUDA Streams**: Use `training_state.stream_ctrl.getPrimaryStream()` - NEVER create raw `cudaStream_t` locals or store in other structs
  - **Gradient Buffers**: Use `training_state.grad_ctrl` - NEVER call `cudaMalloc`/`cudaFree` for gradients outside TrainingState
  - **Optimizer States**: Use `training_state.optimizer_m_states/optimizer_v_states` - ParameterGroup holds pointers, does NOT allocate
  - **cuBLAS**: Use `training_state.cublas_handle` - NEVER create separate handles
  - Pattern: Structs store pointers only. TrainingState owns allocations. Lifecycle methods (`allocate*`/`free*`) are centralized.
  - Rationale: Prevents memory leaks, double-frees, stream synchronization bugs, and makes resource lifetime explicit.

25. **FFN Post-GELU Cache CRITICAL (Fixed Dec 2025)**: EncodingLayer::forward() MUST write post-GELU activations to `args.cache_ffn_output`. Bug history: field existed in `EncodingForwardArgs` but was NEVER written, causing `cached_ffn_outputs[]` to contain garbage. Backward pass then used garbage for `grad_W2 = ffn_hidden^T @ grad_output`, corrupting W2 weight gradients. Symptom: FFN gradient norm was 4.3x smaller than expected. Fix: Added cudaMemcpyAsync after ffn_->forward() to copy `post_gelu` to `args.cache_ffn_output`. If you add new forward caches, VERIFY they're actually written!

23. **GRIM-text vs G.R.I.M Delegate Systems - CRITICAL SEPARATION**: GRIM-text is a SEPARATE PROGRAM from G.R.I.M and MUST NOT depend on G.R.I.M's core libraries:
  - **WRONG**: `#include "../../../../core/delegate.hpp"` in GRIM-text files - this is G.R.I.M's main program delegate system
  - **CORRECT**: GRIM-text has its own GPU delegate system at `resources/models/GRIM-text/Shared/Delegate/Delegate.hpp` (GPU-side `__device__` functions only)
  - **YAGNI Principle**: If a GRIM-text component declares delegate/callback functionality but NO code registers callbacks, DELETE the delegate code entirely
  - **Files affected**: StreamController, GradAccumulationController, and any Shared/ components - they should NOT use any delegate pattern
  - **Rationale**: GRIM-text trains/runs as standalone executable (`train_gpu.exe`, `grim_text_server.exe`), completely independent of G.R.I.M's main process
  - **Build separation**: GRIM-text builds in `resources/models/GRIM-text/training/`, G.R.I.M builds in root `build/` - NO cross-dependencies allowed

24. **GPU Gradient Norm Sync Performance**: `computeGradNorm()` has GPU kernel implementation but syncs CPU-GPU by default for logging. This adds **7+ seconds per batch** on 3080Ti when called every iteration:
  - **Root cause**: `cudaStreamSynchronize()` forces CPU to wait for GPU completion to read metrics (total_norm, component norms)
  - **Solution**: Pass `sync_for_host_read=false` to skip sync when metrics not needed (9 out of 10 batches)
  - **When to sync**: Only when gradient component logging needed (every 10 batches) or for debugging
  - **Gradient clipping**: Does NOT need sync - clipping happens on GPU using device-resident norms
  - **Performance gain**: 10-15x speedup per batch (from 15s to 1-2s) by eliminating unnecessary CPU-GPU synchronization
  - **Implementation**: `Phase2_TrainingLoop.cu` syncs every 10th batch via `computeGradNorm(batch_idx % 10 == 0)`

56. **Autograd Function Return Statements (Issue #56, Jan 2026)**: When writing autograd-enabled forward functions that build computation graphs, **ALWAYS return the output Tensor**. A missing return statement causes the local Tensor's destructor to run at function end, which cascades deletion of the entire `grad_fn` chain DURING the forward pass. Symptom: "illegal memory access" in backward kernels because `grad_fn` pointers point to freed memory. Fix: Always explicitly `return output;` from forward functions. C++ compilers often don't warn about missing return statements returning non-void (undefined behavior). The FFN::forward() bug in `Feed_Forward_GPU.cu` caused training crashes for weeks until discovered.
  - **Performance gain**: 10-15x speedup per batch (from 15s to 1-2s) by eliminating unnecessary CPU-GPU synchronization
  - **Implementation**: `Phase2_TrainingLoop.cu` syncs every 10th batch via `computeGradNorm(batch_idx % 10 == 0)`

## Testing & Debugging

**Run GRIM**
```bash
# From repo root (sets working directory correctly)
./build/Release/GRIM.exe
```

**Tokenizer Tests**
```powershell
# Build and run tokenizer self-test (37 tests)
cd resources/models/GRIM-text/training/build
cmake --build . --config Release --target unigrambyte_self_test
.\Release\unigrambyte_self_test.exe
```

**Debug GRIM-text Server**
```powershell
# Test server health
python test_grim_model.py

# Check training status
python verify_model_config.py
```

**Voice Testing**
```bash
# STT only (skip if no microphone)
./GRIM.exe --voice-test

# TTS test (requires Python venv + Coqui)
python -m resources.python.coqui_bridge --test
```

## Key Files to Reference

- [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) - GRIM-text backend integration strategy
- [README.md](README.md) - Quick start, dependencies, build instructions
- [docs/OSINT_SETUP.md](docs/OSINT_SETUP.md) - Sherlock OSINT plugin setup
- [docs/VOICE_ANIMATION_SYSTEM.md](docs/VOICE_ANIMATION_SYSTEM.md) - Voice visualization
- [resources/models/GRIM-text/README.md](resources/models/GRIM-text/README.md) - Model architecture

## External Dependencies

**C++ Dependencies (vcpkg manifest mode)**
- **vcpkg.json** - Manifest mode for reproducible builds (not classic mode)
- **CUDA 12.5** - Required for GPU acceleration (cuBLAS, custom kernels)
- **FlatBuffers** - Zero-copy serialization (memory, checkpoints, training status)
- **OpenCV** - Image processing for perception system
- **Tesseract** - OCR engine
- **ONNX Runtime GPU** - For DeBERTa quality verification model
- **Whisper.cpp** - `external/whisper.cpp` (submodule)
- **Porcupine** - `external/porcupine` (wake word engine)
- **BGFX** - `external/bgfx.cmake` (rendering backend)

**Python Dependencies**
- **Coqui TTS** (XTTS v2) - High-quality voice synthesis
- **Sherlock** - OSINT username lookups
```bash
pip install -r resources/python/requirements.txt
```

**GPU Technology Stack**
- **Flash Attention v2** - Memory-efficient attention (48x48 block tiling)
- **cuBLAS** - NVIDIA BLAS library for matrix operations
- **Custom CUDA Kernels** - Fused operations (QKV projection, FFN+GELU, LayerNorm)
- **GELU Activation** - GPU-accelerated via custom kernel
- **Tensor Cores** - FP16 precision support (Ampere/Ada architectures)
