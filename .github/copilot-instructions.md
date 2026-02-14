# G.R.I.M Development Guide

## 🔴 CRITICAL: FORBIDDEN CODE PATTERNS (Rule 20 - Fail Loud)

**NEVER generate these patterns. If you see them, DELETE THEM:**

❌ `x ? x : fallback` - NO fallbacks. NO Stubs. Require x, crash if null  
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

**Severity Hierarchy:** Architectural problems (wrong data flow, dead code paths, disconnected subsystems, silent fallbacks at system boundaries) are the **MOST SEVERE** class of bug. A single architectural misconnect (e.g., loss not connected to gradients, embeddings not scaled, weights tied but gradients canceling) can waste weeks of GPU compute producing meaningless training runs. Always prioritize architectural correctness over local code quality. Rule 26 (YAGNI/delete dead code) is the architectural arm of Rule 20 — dead code creates false confidence that a subsystem is functional when it isn't.

---

## 🟡 MANDATORY: Equation-Based Diagnostic Logging (Rule 21)

**When adding diagnostic logging for AI/ML math operations, ALWAYS use the `[*_EQUATION]` format:**

This format is non-negotiable for training/inference debugging because **you can't argue with hard mathematical facts**.

**Required Format Structure:**

```
[OPERATION_EQUATION] CONTEXT: equation = mathematical_formula
  INPUT_A (description): shape=[dims] min=X max=Y rms=Z
  INPUT_B (description): shape=[dims] min=X max=Y rms=Z
  PARAMETERS: param1=value1, param2=value2
  EXPECTED result = formula_with_values = computed_expectation
  ACTUAL result: min=X max=Y rms=Z mean=W
  [ANOMALY] if actual differs significantly from expected: explanation
```

**Examples of Good Logging Tags:**

- `[GRAD_A_EQUATION]` - Matrix A weight gradient computation
- `[GRAD_B_EQUATION]` - Matrix B weight gradient computation
- `[ATTN_SCORE_EQUATION]` - Attention score computation (Q @ K^T / sqrt(d))
- `[LSE_EQUATION]` - Log-Sum-Exp computation
- `[SOFTMAX_EQUATION]` - Softmax forward/backward
- `[RMSNORM_EQUATION]` - RMSNorm computation
- `[EMBEDDING_EQUATION]` - Embedding lookup with scaling
- `[LOSS_EQUATION]` - Loss computation (cross-entropy, focal, etc.)

**Mandatory Elements:**

1. **Mathematical equation** - The EXACT formula being computed
2. **Input shapes** - Tensor dimensions for dimensional analysis
3. **Input statistics** - min, max, rms (and optionally mean, std)
4. **Expected value** - What the result SHOULD be based on input stats
5. **Actual value** - What was actually computed
6. **Anomaly detection** - Flag when actual >> expected or contains NaN/Inf

**Example Implementation (Attention Scores):**

```cpp
// [ATTN_SCORE_EQUATION] format for attention computation
fprintf(stderr, "[ATTN_SCORE_EQUATION] FLASH_ATTENTION_FWD: score = (Q @ K^T) / sqrt(head_dim) + alibi_bias\n");
fprintf(stderr, "  Q (sample tokens, head 0): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
        n_sample, head_dim, q_min, q_max, q_rms);
fprintf(stderr, "  K (sample tokens, head 0): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
        n_sample, head_dim, k_min, k_max, k_rms);
fprintf(stderr, "  PARAMETERS: scale=1/sqrt(%d)=%.10f, alibi_slope=%.10f, max_bias=%.10f\n",
        head_dim, scale, alibi_slope, max_alibi_bias);
fprintf(stderr, "  EXPECTED score_magnitude = Q_row_norm * K_row_norm * scale = %.10f * %.10f * %.10f ≈ %.10f\n",
        q_row_norm, k_row_norm, scale, expected_score);
fprintf(stderr, "  ACTUAL score: min=%.10f max=%.10f rms=%.10f\n", score_min, score_max, score_rms);
if (fabsf(score_max) > 100.0f) {
    fprintf(stderr, "  [ANOMALY] score_max=%.10f >> expected=%.10f, indicates Q/K magnitude explosion\n",
            score_max, expected_score);
}
```

**Why This Format:**

1. **Irrefutable evidence** - Math doesn't lie; discrepancies reveal bugs immediately
2. **Self-documenting** - Log shows exactly what computation is being performed
3. **Root cause isolation** - Expected vs actual comparison pinpoints where explosion/collapse starts
4. **Audit trail** - Can trace anomaly propagation through computation graph
5. **Reproducible debugging** - Same inputs + formula = same expected output

**When to Add Equation Logging:**

- ✅ ANY new forward/backward kernel implementation
- ✅ When debugging gradient explosion/vanishing
- ✅ When investigating loss anomalies
- ✅ When verifying weight initialization
- ✅ Any GEMM operation (grad_A = C^T @ B, grad_B = A^T @ C, etc.)

**Pre-merge checklist for ML code:**

- [ ] All tensor operations have `[*_EQUATION]` logging available (can be behind `#ifdef DEBUG_EQUATIONS`)
- [ ] Expected values computed from input statistics
- [ ] Anomaly thresholds defined (e.g., score > 100, LSE > 50, gradient > 1e6)
- [ ] Log includes tensor shapes for dimensional analysis

---

## �🔴 ACTIVE BUG INVESTIGATION

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
- GPU-accelerated training and inference (CUDA, GELU activation, FlatBuffers serialization)
- DeBERTa BERT model for training data quality verification

**Core Philosophy:** All features work offline by default. Only browser commands and external APIs require internet.

**Project Status:** Personal Jarvis-style assistant/companion/reasoning engine, not intended for distribution. Optimized for single-user experience made to adapt to user use cases and model's environment and systems it has access to.

**Planned:** Multi-model orchestration system, VR/Quest headset overlay mode, crossplatform support.

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
    "backend": "grim_native", // Uses GRIM-text instead of Ollama/OpenAI
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

**Orchestrator** ([train_gpu.cu](resources/models/GRIM-text/training/train_gpu.cu))

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
21. **Three-Phase Training Files**: The entry point is `train_gpu.cu` which orchestrates `Phases/Phase{1,2,3}_{Startup,TrainingLoop,Cleanup}.{cu,hpp}`. If modifying training logic, edit the appropriate phase file, not the orchestrator. CMakeLists.txt in `training/TrainingLoop/` defines the build.

22. **Unified Loss System**: Use `autograd::unified_loss()` in `AutogradLoss.cu` for training loss computation. This is the ONLY loss path - it combines focal loss, label smoothing, entropy regularization, and cross-entropy into a single autograd-enabled kernel. The `cross_entropy_loss()` function is a convenience wrapper that calls `unified_loss()` with plain CE config. **DELETED**: `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` - these old modules had the bug where loss computation was disconnected from gradients.

23. **Batched Embedding Position Bug (FIXED)**: Prior to fix, when `positions=nullptr` was passed to embedding kernels, position was computed as `token_idx` (global index) instead of `token_idx % seq_len` (within-sequence position). This caused 2/3 of every batch to use wrong position embeddings (e.g., batch 1 tokens got positions 720-1439 instead of 0-719). Symptoms: loss starts at 10.5 (correct random baseline) but increases to 15-20 as sequence length grows, with catastrophic spikes (77-259) for long sequences. Fix in `kernel_embedding_forward` (TensorContract_GPU.cu) now computes `pos_id = token_idx % seq_len` when positions is null.

24. **Embedding Scale Missing (AIAYN √d_model) - Issue #92, Jan 2026 — SUPERSEDED BY #140**: The "Attention Is All You Need" paper states: "we multiply those weights by √d_model" for embedding scaling. GRIM-text was MISSING this scaling - embeddings were used raw (scale=1.0). Without scaling, embedding vectors are too small relative to attention scores (which scale by 1/√d), causing the softmax to be overly flat.

- **Root Cause**: Legacy `Embedding_GPU.cu` was DEAD CODE - never called in production. The embedding_scale parameter was added to the legacy path but production uses `autograd::embedding()` in `TensorContract_GPU.cu`.
- **Fix**: Added `float embedding_scale = 1.0f` parameter to `autograd::embedding()` with proper chain rule handling in backward: `grad_weight[token_id] += grad_output * scale`.
- **Implementation**: `AutogradTraining.cu` passes `sqrt(d_model)` for both token and position embeddings.
- **Legacy Cleanup (Rule 20)**: Deleted ALL dead code from `Embedding_GPU.cu` (kernels, launchers, EmbeddingLayer class). Also deleted `embedding_self_test.cu` and `embedding_autograd_test.cu`. File now only contains `destroyEmbeddingRuntime()` for memory management.
- **⚠️ SUPERSEDED**: Issue #140 removed the sqrt(d_model) scaling entirely because GRIM-text uses ALiBi/RoPE (position info inside attention, NOT residual stream), making the AIAYN scaling purposeless. With tied weights, the scaling created a 27.7x gradient asymmetry causing structural-token weight row divergence.

48. **Hardcoded Defaults in Config Structs Violate Rule 20 (Feb 2026)**: Configuration structs with hardcoded default values create silent fallbacks that hide initialization failures. Example: `StartupConfig` had `int max_seq_len = 512` which violated Rule 20 (No Fallbacks).

- **Problem**: If config loading fails to set `max_seq_len`, the struct silently uses 512 instead of failing loud. This hides bugs and creates wrong memory allocations.
- **Rule 20 Enforcement**: Derived/critical parameters should default to `0` (or invalid sentinel) and validation MUST throw if still 0 after loading.
- **Fix Pattern**:
  ```cpp
  struct StartupConfig {
      // WRONG: int max_seq_len = 512;  // Silent fallback!
      // RIGHT:
      int max_seq_len = 0;  // MUST be set during loadConfiguration, throw if 0
      int sliding_window_stride = 0;  // Derived from max_seq_len
  };
  ```
- **Detection**: Search for `= [non-zero]` in config struct declarations. If the value is algorithmic (not a flag), it probably violates Rule 20.
- **Files Affected**: `Phase1_Startup.hpp` - Fixed `StartupConfig.max_seq_len` and `sliding_window_stride` defaults from 512/256 → 0.
- **Related**: See MAX_SEQ_LEN_AUDIT.md for comprehensive analysis of `max_seq_len` usage throughout codebase.

105. **RMSNorm Diagnostic False Anomaly (NOT A BUG) - Issue #105, Jan 2026**: RMSNorm diagnostic was flagging Layer 0 as anomalous (`output_rms=0.89` instead of expected `1.0`). The diagnostic expectation formula was **WRONG** - it assumed `output_rms = gamma_rms`, but the **correct** formula is: `output_rms = input_rms × gamma_rms / sqrt(input_rms² + eps)`. When input_rms is small (~0.006 for Xavier-init embeddings) and eps=1e-5, epsilon contributes ~20% of the denominator, resulting in output_rms ≈ 0.89 which is mathematically correct!

- **Root Cause**: Diagnostic in `Encoding_GPU.cu` assumed `output_rms = gamma_rms`, which only holds when `input_rms² >> eps`. With small embeddings (per_row_rms=0.006), eps=1e-5 becomes significant.
- **Fix**: Changed diagnostic to compute correct expected value: `expected_output_rms = input_rms * gamma_rms / sqrt(input_rms² + eps)`. Added epsilon contribution percentage to help identify when this matters.
- **Note**: This is NOT a training bug - RMSNorm is computing correctly. The diagnostic was generating false positives.

107. **Xavier Init LCG PRNG Correlation (CRITICAL FIX Jan 2026)**: The `kernel_xavier_uniform` kernel in `TensorContract_GPU.cu` produced highly correlated random values with avg|cosine| ≈ 0.37 instead of expected ≈ 0.036 (12x too high!). Combined with Issue #106, this caused W_qkv initialization to NOT be fixed even after applying the scaling.

- **Root Cause**: The LCG PRNG computed state as a LINEAR function of thread index:
    ```cuda
    uint64_t state = seed + idx * A;  // Linear in idx!
    state = state * A + C;            // Only ONE iteration
    ```
    Since `state[idx+1] - state[idx] = A²` (constant), consecutive elements had correlated outputs.
- **Symptom**: Diagnostic showed row_rms=0.0312 (expected 0.0011 with Issue #106 fix), avg|cosine|=0.374 (expected ~0.036), individual cosines as high as 0.847 (85% aligned!).
- **Why Issue #106 Wasn't Applied**: `Phase1_Startup.cu` only reseeded when `sample_rms < 1e-7f`, but `TrainingTensors.cu` already initialized weights with the BUGGY kernel.
- **Fix (2 parts)**:
    1. Fixed LCG in `kernel_xavier_uniform`: Use splitmix64-style mixing for per-element seed, run 16 LCG iterations before extracting value.
    2. Removed `sample_rms < 1e-7f` check in `Phase1_Startup.cu` - ALWAYS reinitialize encoder weights with corrected scaling.
- **Files Modified**: `TensorContract_GPU.cu`, `Phase1_Startup.cu`
- **Expected Result**: avg|cosine| ≈ 1/sqrt(768) ≈ 0.036, row_rms ≈ 0.0011 (with Issue #106 scaling)

106. **W_qkv Initialization Too Large (LSE Explosion) - Issue #106, Jan 2026**: QKV projection output was 33x larger than expected, causing attention score explosion (800 instead of ~1) and LSE explosion (700+ instead of 6-10).

- **Root Cause Chain**:
    1. RMSNorm output has row_norm ≈ sqrt(d_model) ≈ 27.7
    2. W_qkv uses standard Xavier init with stddev ≈ 0.031 → row_norm ≈ 0.87
    3. **Isotropic columns** (variance ratio 2.01x) + **high cosine alignment** (154x expected) cause **coherent summation** instead of partial cancellation
    4. QKV GEMM output row_norm = 265.6 (33.2x larger than expected 0.77)
    5. Q/K row norms ≈ 80 instead of target 8 → attention scores ≈ 800 → LSE explodes to 700+
- **Fix**: Scale W_qkv initialization by `1/sqrt(d_model)` ≈ 0.036 to compensate for coherent summation
- **Implementation**: `Phase1_Startup.cu` - `qkv_std = qkv_std_base * (1.0f / sqrt(d_model))`
- **Expected Result**: Q/K row_norm ≈ 265 × 0.036 ≈ 9.6 (close to target 8), LSE in normal range 6-10

98. **LM Head Scale Mismatch with Tied Embeddings - Issue #98, Jan 2026 — REVERTED, SUPERSEDED BY #140**: When `tie_embeddings=true`, Issue #92 scales embedding forward by `sqrt(d_model)≈27.7`, but LM head used RAW weights. This created **27.7x gradient attenuation** in backward pass.

- **Root Cause**: AIAYN states "multiply weights by sqrt(d_model)" - this applies to BOTH embedding lookup AND output projection with tied weights. GRIM only did the embedding side.
- **Symptom**: FlashAttention dQ gradients were ~0.0000004 (should be ~0.00001). LM head backward `grad_encoder = grad_logits @ weights` used raw weights rms≈0.006 instead of scaled weights.
- **Fix**: Added `autograd::scale(logits, sqrt(d_model))` after LM head matmul when `tie_embeddings=true`.
- **Implementation**:
    - `TensorContract_GPU.cu`: Added `ScaleGradFn` struct and `autograd::scale()` function
    - `TensorContract_GPU.hpp`: Added `scale()` declaration
    - `AutogradTraining.cu`: Apply scaling to LM head output when tie_embeddings=true
- **Math**: `logits_scaled = logits * scale` → backward: `grad_input = grad_output * scale`. This restores gradient magnitude to match embedding forward scaling.
- **PyTorch Note**: PyTorch baseline does NOT scale embeddings at all (plain lookup), so no mismatch exists there.
- **⚠️ REVERTED**: `ScaleGradFn` and `autograd::scale()` deleted as dead code (Rule 20). Issue #140 takes the opposite approach: removing embedding scaling entirely (like PyTorch) instead of adding matching LM head scaling.

97. **Encoder Biases Frozen (No Autograd Tracking) - Issue #97, Jan 2026**: All encoder biases (b_qkv, b_o, b1, b2) received **ZERO gradients** because bias addition used raw CUDA kernels (`launchFFNBiasAdd`) that bypassed autograd completely.

- **Root Cause**: The forward pass did `qkv_out = matmul(ln1_out, W_qkv)` (autograd ✓) then `qkv_out.data += b_qkv` (raw kernel, NO GradFn!). Backward: MatMulGradFn has NO knowledge of b_qkv, so bias gradients = 0.
- **Symptom**: Biases stayed at initial values (zeros or Xavier) throughout training. Diagnostic showed `b_qkv: [0.0, 0.0, 0.0, ...]` never changing.
- **Fix**: Created `autograd::broadcast_add(Tensor input[N×D], Tensor bias[D])` with proper `BiasAddGradFn`:
    - Forward: `output[i,j] = input[i,j] + bias[j]` (uses existing `launchFFNBiasAdd`)
    - Backward for input: `grad_input = grad_output` (pass-through, standard for addition)
    - Backward for bias: `grad_bias[j] = Σᵢ grad_output[i,j]` (sum over batch dimension, uses existing `launchFFNBiasBackward`)
- **Implementation**: `BiasAddGradFn` struct in `TensorContract_GPU.cu`, following patterns from Issue #48 (stable data caching), #50 (grad_fn ownership), #54 (result ownership), #56 (non-leaf buffers).
- **Files Modified**:
    - `TensorContract_GPU.cu`: Added `BiasAddGradFn` struct and `autograd::broadcast_add` function
    - `TensorContract_GPU.hpp`: Added `broadcast_add` declaration
    - `Encoding_GPU.cu`: Changed b_qkv and b_o from `launchFFNBiasAdd` to `autograd::broadcast_add`
    - `Feed_Forward_GPU.cu`: Changed b1 and b2 from `launchFFNBiasAdd` to `autograd::broadcast_add`

18. **CMake Cache After Removing Files**: When removing .cu files from CMakeLists.txt, must clean CMake cache to remove stale device-link objects. Linker errors like `unresolved external symbol __fatbinwrap_*` indicate cached library still references deleted files. Fix: `Remove-Item -Recurse -Force build\CMakeFiles\grim_training_kernels.dir` then rebuild, or use `cmake --build build --config Release --clean-first`.

19. **PCGrad for Tied Embedding/LM Head Weights (CRITICAL FIX Jan 2026)**: When `tie_embeddings=true`, the LM head backward and embedding backward produce **OPPOSITE** gradients that cancel to ZERO! The diagnostic showed: `LM_HEAD: sum=+4.05, EMBEDDING: sum=-4.05, COMBINED: sum=0.00, COSINE=-1.0 (CANCELING)`.

- **Root Cause**: LM head wants token prediction probability to INCREASE, embedding wants input representation to DECREASE. With shared weights, these are opposite directions!
- **Fix**: PCGrad (Projecting Conflicting Gradients) - `g_final = g_lm + (g_emb - proj_{g_lm}(g_emb))`. This projects out the conflicting component, keeping the LM head direction + any orthogonal embedding information.
- **Implementation**: `kernel_pcgrad_combine` in `TensorContract_GPU.cu`, `pcgrad_temp_buffer` in TrainingState, allocated in Phase1_Startup.cu when `tie_embeddings=true`.
- **Effect**: When cosine=-1 (opposing): uses g_lm only. When cosine=0 (orthogonal): uses g_lm + g_emb. When cosine=+1 (aligned): avoids double-counting.

72. **FlashAttention GQA dK/dV Buffer Overflow (CRITICAL FIX Jan 2026)**: The Dao-AILab FlashAttention backward kernel writes dK/dV gradients using the **query head index** (`bidh`), NOT the KV head index (`bidh / h_h_k_ratio`). With GQA (12 Q heads, 4 KV heads), query heads 4-11 write **outside** a buffer sized for num_kv_heads=4. This causes STATUS_STACK_BUFFER_OVERRUN (exit code -1073740791 / 0xC0000409) and corrupts adjacent memory, leading to gradient explosion.

- **Symptom**: Training crashes after ~100-120 batches with exit code -1073740791. Gradients show massive spikes (attn=4,501,145, rms=520,659) before crash.
- **Root Cause**: `flash_bwd_kernel.h` line 758 writes to `bidh * dk_head_stride` where `bidh` = 0..11. Buffer allocated for 4 KV heads → heads 4-11 write out-of-bounds.
- **Fix**: Allocate `dk_bf16`/`dv_bf16` for `num_heads` (not `num_kv_heads`), then reduce gradients afterward using `kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32` which sums grouped Q head gradients to get proper KV head gradients.
- **Implementation**: Fix must be applied in **TWO places** in `TensorContract_GPU.cu`:
    1. `ScaledDotProductAttentionGradFn::save()` - allocates dk_bf16/dv_bf16 with `dk_dv_alloc_elems = b * s * num_heads * hd`
    2. `scaled_dot_product_attention()` - allocates grad_fn->dk_bf16/dv_bf16 with same sizing
- **CRITICAL**: Both allocation sites MUST use num_heads (12), not num_kv_heads (4). Initial fix only patched save(), leaving scaled_dot_product_attention() with wrong size → continued crashes.

73. **GQA Gradient Scaling Missing in Reduction Kernel (CRITICAL FIX Jan 2026)**: Issue #72's reduction kernel **SUMMED** gradients from 3 Q heads but did NOT apply GQA gradient scaling. The external Dao-AILab FlashAttention library (unlike our old custom kernel) does NOT apply `gqa_grad_scale = 1.0 / heads_per_kv_group` internally. Result: K/V gradients were **3x too large**, causing attention gradient explosion (16K-600K instead of ~1-5).

- **Symptom**: After Issue #72 fix, training no longer crashes but attention gradients are still massive (attn=28,488 to 610,297). Some batches are normal (attn=2.7) - these happen to have lower loss.
- **Root Cause**: `kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32` did `dst[idx] = sum` without dividing by `heads_per_kv_group`.
- **Fix**: Apply `gqa_grad_scale = 1.0f / heads_per_kv_group` in the reduction kernel: `dst[idx] = sum * gqa_grad_scale`.
- **Mathematical basis**: When 3 Q heads each produce dK/dV for the same KV head, the total contribution should be averaged, not summed. This matches the old custom Flash kernel behavior which had `dv *= gqa_grad_scale` and `dk *= gqa_grad_scale`.

78. **ALiBi Causes dQ/dK vs dV Asymmetry Through Softmax Backward Math (Issue #78, Jan 2026)**: FlashAttention backward produces exploded dQ/dK (100,000x) while dV stays normal. This is due to the fundamental difference in how softmax backward treats these gradients:

- **dV = P^T @ dO** - Uses softmax probabilities P directly, which are bounded [0,1] and sum to 1
- **dS = P \* (dP - dP_sum)**, then **dQ = dS @ K**, **dK = dS^T @ Q** - Uses softmax Jacobian which can amplify
- **ALiBi effect**: With `m_max=0.25` and `max_seq_len=1024`, head 0 applies bias `-0.25 * 1024 = -256` to distant positions. This creates near-zero attention weights `P[i,j] ≈ exp(-256) ≈ 0` for distant positions.
- **Why dV stays bounded**: `dV = P^T @ dO` is a weighted average of dO values, bounded by max(abs(dO))
- **Why dQ/dK explode**: `dS = P * (dP - dP_sum)` involves subtraction. When P is highly localized (most weight on ~32 tokens), `dP_sum` is dominated by high-P positions. For positions where `P[i,j] ≈ 0` but `dP[i,j] ≠ 0`, the term `-P[i,j] * dP_sum` creates residual gradients that don't cancel.
- **Compounding through layers**: Backward starts at L11 (near output) with normal gradients. Each layer's dQ/dK flow through residual connections to the previous layer's input. Any amplification COMPOUNDS: L11→L10→L9...→L0. By L6-L0, amplification reaches 100,000x.
- **GQA is NOT the cause**: `kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32` applies SAME `gqa_grad_scale = 1/3` to both dK and dV. The asymmetry originates INSIDE FlashAttention backward, not our reduction code.
- **FIX IMPLEMENTED**: Clamped ALiBi bias via `ALIBI_MAX_BIAS = -10.0f` in HyperParameters_GPU.hpp. Slopes are capped so `abs(slope) * max_seq_len <= abs(ALIBI_MAX_BIAS)`. This ensures `exp(-10) ≈ 0.000045` (computable) instead of `exp(-256) ≈ 0` (underflow). Implementation in `PositionalBiasMethod.cu::computeAlibiSlopes()`.

21. **Weight Tying Grad Buffer Aliasing**: When `tie_embeddings=true`, `embedding_grads` and `lm_head_weight_grads` are the **same pointer** (aliased). NEVER:

- Zero both buffers separately (they're the same memory - will destroy accumulated gradients)
- Add both to parameter*groups* (double optimizer update)
- Free both in destructor (double-free crash)
- The ownership flags `lm_head_weights_owned` and pointer comparison `embedding_grads == lm_head_weight_grads` track aliasing.
- **NOTE**: With PCGrad fix (Issue #60), embedding backward writes to TEMP buffer first, then orthogonal component is added to shared grad buffer.

44. **per_token_grad_scale is REQUIRED (NOT a bug)**: When loss is averaged (`loss_mean = sum(losses) / valid_tokens`), the backward pass gradient is already scaled by `1/valid_tokens`. The `per_token_grad_scale=true` setting in `ai_config.json` is MANDATORY to ensure gradients match this scaling. DO NOT disable this thinking "tiny gradients" is a bug - gradient RMS of ~1e-6 is CORRECT when `valid_tokens ~ 3000`. The tiny gradients combine correctly in AdamW. Disabling this causes gradient magnitude explosion (effective LR becomes 3000x larger).

45. **Issue #38 WAS WRONG - SUPERSEDED BY #60**: The original "fix" that SKIPPED embedding backward was wrong. The CORRECT fix is PCGrad (Issue #60) which preserves both gradient sources without cancellation.

46. **Encoder Weight Gradient Centering (Issue #43, Jan 2026)**: Encoder backward uses cached activations (`cached_ln1_output`, `cached_ffn_input`, `cached_ffn_hidden`, `cached_attn_output`) for weight gradient GEMMs. These activations have NON-ZERO MEAN, creating systematic gradient bias: `grad_W[i,j] = Σ_t ((centered[t,i] + mean_t) × grad[t,j])` - the mean term doesn't cancel. Fix in `BackwardPhase2_Encoder.cu`: Added `centerActivationsKernel` to center each cached activation BEFORE weight gradient GEMMs (grad_W_qkv, grad_W_o, grad_W1, grad_W2). Uses `centered_activation_scratch` buffer allocated in TrainingState. This matches the Issue #37 pattern applied to LM head.

47. **ALiBi Slopes: Context-Length-Aware Scaling (Issue #69 + multi-model fix)**: ALiBi slopes MUST scale relative to `max_seq_len` for multi-model orchestration. The formula in `PositionalBiasMethod.cu` computes:

- `d_min = rotary_dim/2` (or 16): locality distance for strongest head
- `d_max = max_seq_len`: context length being trained/inferred on
- `m_max = target_bias / d_min`: penalty slope for NEAR distances (head 0)
- `m_min = target_bias / d_max`: penalty slope for FAR distances (head N-1)
- Slopes interpolated **geometrically** from `-m_max` to `-m_min`
- **CRITICAL**: FlashAttention expects NEGATIVE slopes (library uses `+= slope * col_idx`)
- **WARNING**: If you set `max_seq_len=2048` but run inference at 8192 tokens, weakest heads will be too weak! Always match `max_seq_len` to actual context length.

70. **RoPE: NTK-Aware Context Scaling**: RoPE frequencies auto-scale when `max_seq_len > 2048` (base context). Formula in `PositionalBiasMethod.cu`:

- `effective_theta = theta * (max_seq_len / 2048)^(rotary_dim / (rotary_dim - 2))`
- This is "dynamic NTK" / "Code Llama style" scaling for extending context
- `rope_scaling` field provides additional manual scaling (default 1.0)

113. **Sinusoidal Position Embeddings DELETED (Feb 2026)**: Issue #103 correctly removed **learned** position embeddings (they were isotropic → QKV explosion). Issue #113 originally added sinusoidal position embeddings as a replacement, but the `addSinusoidalPositionEmbeddingsKernel` was subsequently **DELETED**. With ALiBi/RoPE, position information is injected **INSIDE attention** (via bias/rotary), not in the residual stream. Same tokens at different positions DO have identical residual-stream representations — this is by design for ALiBi/RoPE architectures. Position differentiation comes from the attention mechanism itself.

- **Current state**: No position embeddings added to embeddings when using ALiBi/RoPE. Only learned position embeddings are supported (via `positional_encoding.use_learned=true` in config), and those are only activated when `positional_encoding` resolves to `NONE`.
- **WARNING**: The `positional_encoding` config was previously hardcoded and not parsed from JSON. Issue #141 fixes this.

125. **LM Head Centering Used WRONG CENTERING FUNCTION (ROOT CAUSE FIX - Issue #125, Feb 2026)**: The `launchCenterHiddenStates()` function did **ROW-WISE** centering (`mean(hidden[t,:])` - per-token mean across features) instead of **COLUMN-WISE** centering (`mean(hidden[:,d])` - per-feature mean across tokens). Row centering does NOT reduce hidden state correlation - it only makes each row sum to 0 without changing the angle between vectors!

- **Root Cause**: `lm_head_GPU.cu::centerHiddenStatesKernel` computed `mean = s_sum / d_model` (dividing by 768 features) with one CUDA block per token → row centering. The kernel was designed wrong from the start.
- **Symptom**: Despite centering being "enabled", diagnostic still showed `avg|cos(h_i,h_j)| = 0.958` (26.6x higher than expected 0.036)
- **Mathematical Proof**:
    - **Row centering**: `centered[t,d] = x[t,d] - mean_d(x[t,:])` → Subtracting same scalar from all features is a TRANSLATION, which preserves angles
    - **Column centering**: `centered[t,d] = x[t,d] - mean_t(x[:,d])` → Removes the SHARED DIRECTION, directly reduces cos(h_i, h_j)
- **Fix (AutogradTraining.cu)**: Replaced `launchCenterHiddenStates()` with `autograd::center_columns()` which correctly centers across positions
- **Files Modified**: `AutogradTraining.cu` - LM head input centering now uses column-wise centering
- **Expected Result**: `avg|cos(h_i,h_j)|` should drop from ~0.96 to ~0.036 (1/√768), mode collapse resolved

126. **RMSNormGradFn Stale input_grad Pointer (CRITICAL FIX - Issue #126, Feb 2026)**: RMSNormGradFn stored a raw pointer to `x.grad_data()` during `capture_inputs()`. When the input tensor `x` was a temporary (created by `autograd::add()` or similar), it would be destroyed before `backward()` ran, leaving `input_grad` as a dangling pointer.

- **Root Cause**: `capture_inputs()` called `x.ensure_grad()` which allocated a gradient buffer owned by `x`, then stored `input_grad = x.grad_data()`. When `x` destructed, the buffer was freed.
- **Symptom**: `[RMS-BWD-VALIDATE] input_grad read: err=1 (invalid argument)` - CUDA error when accessing freed memory
- **Fix**: RMSNormGradFn now allocates its OWN gradient buffer via `cudaMalloc()` instead of borrowing from the input tensor.
- **New Member**: `bool owns_input_grad = false` - tracks whether we own the buffer and need to free it
- **Files Modified**: `TensorContract_GPU.cu` - `RMSNormGradFn::capture_inputs()` and destructor

127. **center_columns() Causes Use-After-Free of RMSNormGradFn (CRITICAL FIX - Issue #127, Feb 2026)**: When `lm_head_center_hidden_states=true`, `autograd::center_columns()` creates a `CenterColumnsGradFn` that takes ownership of `encoder_output_tensor.grad_fn` (the RMSNormGradFn). If the centered tensor was a local variable, it would be destroyed after forward(), deleting the RMSNormGradFn. But `lm_input_tensor.grad_fn` still pointed to it → crash in backward.

- **Root Cause Chain**:
    1. `center_columns(ctx.encoder_output_tensor)` creates CenterColumnsGradFn
    2. CenterColumnsGradFn::capture_input() transfers ownership of RMSNormGradFn
    3. Local `Tensor centered_encoder_output` goes out of scope after forward()
    4. CenterColumnsGradFn destructor DELETES RMSNormGradFn
    5. `ctx.lm_input_tensor.grad_fn` (set to encoder_output_tensor.grad_fn) is now dangling
    6. Backward calls `a_grad_fn->apply()` → crash on invalid vtable
- **Symptom**: Crash immediately after `[MATMUL-BWD-TO-A] About to call a_grad_fn->apply()` with no ENTRY log
- **Fix (AutogradTraining.cu + hpp)**:
    1. Added `Tensor centered_encoder_output` member to ForwardContext (persists until clearIntermediates)
    2. Changed `lm_input_tensor.grad_fn` to use `centered_encoder_output.grad_fn` when centering enabled
    3. Same fix for numeric_head's encoder_for_numeric tensor
- **Files Modified**: `AutogradTraining.cu`, `AutogradTraining.hpp`

128. **Weight Paradox - Token 277 Gradient Sign Conflict (ANALYSIS - Issue #128, Feb 2026)**: Despite Issue #126 centering fix working (avg_cos ≈ -0.001), Token 277 increasingly dominates predictions because gradient sign conflicts cause ||W[277]|| to INCREASE when it should decrease.

- **Root Cause**: `grad_W[277,i] = Σ_t hidden[t,i] × grad_logits[t,277]`
    - At target=277 positions: `grad_logits = p(277) - 1 ≈ -0.94` (negative)
    - At target≠277 positions: `grad_logits = p(277) + entropy_term ≈ -0.02` (negative due to entropy reg)
    - When `hidden_sum[t] < 0` AND `grad < 0`: negative × negative = **POSITIVE** contribution
    - Net gradient can have WRONG sign: `from_277_targets=+0.08, from_other=-0.03 → total=+0.05`
- **Symptom**: Training log shows `[ANOMALY] WEIGHT_PARADOX_SOURCE` - gradient contribution positive despite model trying to decrease W[277]
- **Evidence from Session 17702644885411807**:
    - Loss NOT converging: 11.75→10.85-12.74 (no improvement)
    - Token 277 in argmax: 0/50 (batch 1) → 6-8/50 (batch 95+)
    - unique_argmax decreasing: 50 → 43-45 (diversity loss)
    - But avg_cos stable at -0.001 (centering IS working!)
- **Potential Fixes**: Per-token gradient clipping, focal loss gamma increase, hidden centering per-position
- **Status**: Under investigation - centering fixes hidden correlation but not weight gradient sign conflicts

129. **LayerScale=0.1 Causes Gradient Vanishing (ROOT CAUSE FIX - Issue #129, Feb 2026)**: Training loss not converging despite correct forward pass. LayerScale initialized to 0.1 (CaiT paper recommendation) causes catastrophic gradient attenuation through encoder layers.

- **Gradient Flow Analysis**:
    1. LM_HEAD grad_C: max=0.000147, rms=1.47e-6 (healthy)
    2. LM_HEAD grad_A (→encoder): max=1.64e-6, rms=8.77e-7 (90x drop from weight magnitude)
    3. LayerScale: grad_attn = 0.1 × grad_output (additional 10x drop)
    4. Result: Encoder weight gradients have RMS ~1e-8
- **Effective Learning**: `update = lr × grad = 3e-4 × 1e-8 = 3e-12` (essentially ZERO!)
- **Root Cause**: LayerScale forward is `y = x * 0.1`, backward is `grad_x = grad_y * 0.1`. Combined with LM head weight scaling (~90x drop), gradients vanish.
- **Fix**: Changed `ai_config.json` → `layer_scale.init_value` from 0.1 to 1.0
- **Why**: `init_value=1.0` disables gradient attenuation while keeping learnable scalar (can adapt during training)
- **Files Modified**: `ai_config.json`
- **Alternative**: Set `layer_scale.enabled=false` to completely remove LayerScale

115. **Diagnostic Buffer Mismatch (DIAGNOSTIC BUG! - Issue #115, Feb 2026)**: Diagnostic functions in `Phase2_TrainingLoop.cu` were reading the WRONG buffer (`cached_encoder_output` = pre-centering) instead of the centered buffer (`centering_scratch_tensor` = post-centering). This caused all diagnostic logs to show incorrect values even though training was using centered data correctly.

- **Root Cause**: `AutogradTraining.cu` populates `cached_encoder_output` at line 947 (BEFORE centering) and `centering_scratch_tensor` at line 1055 (AFTER centering). Training correctly uses the centered buffer, but all 3 diagnostic functions read the pre-centering buffer.
- **Symptom**: `[HIDDEN_STATE_EQUATION]` showed `hidden_mean ≈ -0.001` (non-zero) when it should show `hidden_mean ≈ 0` (centered)
- **Impact**: Led to false conclusions that centering wasn't working, wasted investigation time on Issues #108-#114
- **Fix (Phase2_TrainingLoop.cu)**:
    - `computeHiddenState277Analysis()` - Added `bool use_centering` parameter with conditional buffer selection
    - `computeFeedbackLoopDiagnostic()` - Added `bool use_centering` parameter with conditional buffer selection
    - `[HiddenCosine]` diagnostic - Added inline conditional buffer selection using `ctx.model->getConfig().lm_head_center_hidden_states`
    - Call sites now pass `cfg.lm_head_center_hidden_states` as the `use_centering` parameter
- **Verification**: After fix, `[HIDDEN_STATE_EQUATION]` should show `hidden_mean ≈ 0` when centering is enabled

130. **QKV "Expected" Diagnostic Formula Was WRONG (DIAGNOSTIC BUG - Issue #130, Feb 2026)**: The QKV_PROJECTION_EQUATION diagnostic showed `expected_row_norm=0.864` vs `actual_row_norm=24.0`, claiming a "28x mismatch". This was **NOT a training bug** - the diagnostic formula was wrong.

- **What the diagnostic computed**: `expected = ln1_row_norm × wqkv_row_norm / sqrt(d_model)` ≈ 0.864
- **Correct GEMM output formula**: `||Y_row|| ≈ sqrt(d_model) × σ_x × σ_w × ||x_row||` ≈ 24
- **Conclusion**: The actual value (24.0) is **mathematically correct**. The diagnostic expected formula was wrong.
- **Impact**: Led to false conclusions about QKV projection being broken. Individual GEMM operations are working correctly.
- **Files**: `Encoding_GPU.cu` lines ~2150 has the wrong expected formula in diagnostic code

131. **LibTorch Gradient Comparison Was INVALID (DIAGNOSTIC BUG - Issue #131, Feb 2026)**: The gradient comparison showed GRIM QKV grad norm (0.118) vs LibTorch (13.77) = "116x mismatch". This comparison was **invalid** due to completely different model configurations.

- **LibTorch Baseline Config**: d_model=512, num_layers=6, num_heads=8, batch_tokens=1,536
- **GRIM Config**: d_model=768, num_layers=12, num_heads=12, batch_tokens=6,000
- **Why invalid**: Different model depth/width affects gradient magnitude. Different batch size affects gradient scale from mean reduction.
- **Corrected Analysis**: Expected ~4x from batch size (6000/1536), remaining ~30x from model size difference
- **To fix**: Run LibTorch baseline with IDENTICAL config to GRIM before claiming gradient mismatch
- **Files**: `Tools/libtorch_baseline/main.cpp` - needs config matching GRIM's `ai_config.json`

132. **Hidden State Gradient Sign Flip Due to Row-Sum Variance (FIXED - Issue #132, Feb 2026)**: The gradient sign flip was caused by row sums varying across positions.

- **Root Cause**: Column centering makes `Σ_t hidden[t,d] = 0` but does NOT make `Σ_d hidden[t,d] = 0` (row sums still vary)
- **Mechanism**: At 277-target positions, `hidden_sum[t]` correlates with being more NEGATIVE than average
- **Sign Flip**: `grad_logits[t,277]` is negative (want to decrease) × `hidden_sum[t]` is negative = POSITIVE contribution
- **Result**: W[277] weight INCREASES when it should decrease → mode collapse feedback loop
- **Evidence**: Training log shows `[ANOMALY] WEIGHT_PARADOX_SOURCE` with positive total contribution despite negative grad
- **FIX**: Apply ROW centering AFTER column centering in `AutogradTraining.cu`:
    1. Column centering: `Σ_t h[t,d] = 0` for each feature d → reduces hidden state correlation (Issue #125)
    2. Row centering: `Σ_d h[t,d] = 0` for each position t → eliminates gradient sign flip (Issue #132)
- **Implementation**: `ctx.centered_encoder_output = autograd::center_rows(autograd::center_columns(encoder_output))`
- **Why This Works**: With `hidden_sum[t] = 0` for all positions t, the contribution formula `Σ_t hidden_sum[t] × grad[t,277]` becomes 0, preventing systematic sign errors
- **Files Modified**: `AutogradTraining.cu` (added center_rows after center_columns)

133. **Flat text_ce = Uniform Softmax + Entropy Reg Mask (ROOT CAUSE FIX - Issue #133, Feb 2026)**: text_ce stuck at 9.76 was NOT "near random" — it IS the random baseline `ln(50377)=10.83` minus entropy regularization offset `0.1×10.83=1.08=9.75`. Model predictions were UNIFORM because logit std≈0.17 over 50K vocab.

- **Root Cause (3 compounding issues)**:
    1. **Logit magnitude too small**: Xavier-init embeddings (rms≈0.006) + RMSNorm-normalized hidden states (rms≈1.0/dim) → logit std = 0.006 × sqrt(768) ≈ 0.17. With 50K vocab, softmax≈uniform.
    2. **Entropy reg (λ=0.1) masks true CE**: Subtracts 0.1×ln(V)=1.08 from loss, making 10.83 look like 9.76. Also actively FIGHTS CE gradient by pushing back toward uniform.
    3. **Weight decay=0.3 too aggressive**: Standard is 0.01-0.1.
- **Mathematical Proof**: For uniform p_v=1/V: CE=ln(V)=10.83, entropy_penalty=-0.1×ln(V)=-1.08, total=9.75≈observed 9.76 ✓
- **Why logit scaling is NOT the fix**: Softmax gradient at uniform is `(p - y) ≈ -1` regardless of logit magnitude. Scaling logits just amplifies backward by sqrt(d_model)=27.7x without helping escape the uniform basin.
- **Fixes (config only)**:
    - `entropy_reg.enabled=false` (was masking true CE and fighting learning)
    - `weight_decay=0.1` (was 0.3, too aggressive)
    - `focal.enabled=false` (no benefit at uniform predictions)
- **Expected**: Initial loss ~10.83 (true random baseline, no entropy mask), then decreasing

114. **Hidden State Norm Explosion Feedback Loop (ROOT CAUSE - Issue #114, Feb 2026)**: The persistent Token 277 (SPACE) mode collapse is caused by a **self-reinforcing feedback loop** with 5 interlocking anomalies. The mathematical root cause is:

- **Root Cause Equation**: `logit[277] = h · W[277]^T = ||h|| × ||W[277]|| × cos(h, W[277])`
- **The Paradox**: Even when gradient says "decrease W[277]", logit INCREASES because ||h|| growth outpaces any W[277] reduction!
- **Five Critical Anomalies**:
    1. **Hidden Norm Explosion (+111%)**: ||h|| grows 24.79→52.34+ because encoder amplifies the aligned direction. RMSNorm normalizes variance but NOT magnitude.
    2. **Cosine Collapse (+232%)**: avg_cos(h_i, h_j) grows 0.25→0.84 - hidden states converge to a single common direction.
    3. **Weight Paradox (+32%)**: ||W[277]|| GROWS from 0.17→0.22 despite NEGATIVE gradients! Hidden mean bias corrupts gradient direction.
    4. **Mean Drift (→0)**: h_mean drifts from -0.011→-0.0004 as hidden states center while norms explode.
    5. **Logit Explosion (+25x)**: logit[277] grows 0.20→5.12 as the product of all factors compounds.
- **Feedback Loop Mechanism**:
    ```
    logit[277] = ||h|| × ||W|| × cos(h,W) high → p(277)↑ → negative grad_logits[277]
    → AdamW tries to shrink W[277] → BUT encoder learns h ALIGNED with W[277]
    → ||h||↑ and cos(h,W)↑ OUTPACE ||W||↓ → logit[277]↑ → REPEAT
    ```
- **Diagnostic Logging**: Added `[FEEDBACK_LOOP_EQUATION]` logging in `Phase2_TrainingLoop.cu` with:
    - Per-batch decomposition: ||h||, ||W[277]||, cos(h, W[277]), growth rates
    - Expected vs Actual logit comparison
    - Anomaly flags for divergence detection
- **Related Issues**: #37 (hidden centering), #43 (encoder centering), #108 (cosine collapse), #111 (centering disabled), #113 (sinusoidal position)
- **Files Modified**: `Phase2_TrainingLoop.cu` (diagnostic), `PLATEAU_BUG_INVESTIGATION.md` (documentation)

22. **Centralized Controller Pattern - MANDATORY**: All GPU resource management MUST go through centralized controllers in TrainingState. VIOLATIONS ARE BUGS:

- **CUDA Streams**: Use `training_state.stream_ctrl.getPrimaryStream()` - NEVER create raw `cudaStream_t` locals or store in other structs
- **Gradient Buffers**: Gradients are managed via autograd system - use `ctx.model->zeroGradients()` before accumulation window, `ctx.model->backward()` computes and accumulates
- **Optimizer States**: Use `training_state.optimizer_m_states/optimizer_v_states` - ParameterGroup holds pointers, does NOT allocate
- **cuBLAS**: Use `training_state.cublas_handle` - NEVER create separate handles
- **Gradient Accumulation**: Track via `ctx.optimizer.current_micro_step` (int), reset to 0 after optimizer step. No traffic light state machine.
- Pattern: Structs store pointers only. TrainingState owns allocations. Lifecycle methods (`allocate*`/`free*`) are centralized.
- Rationale: Prevents memory leaks, double-frees, stream synchronization bugs, and makes resource lifetime explicit.

90. **ScratchBlock Buffer Desync (Issue #90, Jan 2026)**: After `autograd::add(emb, pos_emb)` creates a NEW Tensor with its own buffer, `ctx.embedding_tensor.data` points to this NEW buffer. Then `ts->cached_embeddings` is synced from `emb_output.data`. ScratchBlock operates IN-PLACE on `ts->cached_embeddings`, but `ctx.embedding_tensor.data` is NOT updated! Layer 0 receives STALE pre-ScratchBlock data while Layer 1+ are correct (they use Layer 0's output).

- **Symptom**: LSE explosion (300-600 instead of 6-10) ONLY in Layer 0. K/V values ~8x larger than expected after QKV projection.
- **Root Cause**: Two different buffers - `ctx.embedding_tensor.data` (from autograd::add) vs `ts->cached_embeddings` (ScratchBlock output)
- **Fix**: After ScratchBlock completes, copy `ts->cached_embeddings` back to `ctx.embedding_tensor.data`
- **Implementation**: `AutogradTraining.cu` - added cudaMemcpyAsync after `ctx.scratch_block->forward()`

25. **FFN Post-GELU Cache CRITICAL (Fixed Dec 2025)**: EncodingLayer::forward() MUST write post-GELU activations to `args.cache_ffn_output`. Bug history: field existed in `EncodingForwardArgs` but was NEVER written, causing `cached_ffn_outputs[]` to contain garbage. Backward pass then used garbage for `grad_W2 = ffn_hidden^T @ grad_output`, corrupting W2 weight gradients. Symptom: FFN gradient norm was 4.3x smaller than expected. Fix: Added cudaMemcpyAsync after ffn\_->forward() to copy `post_gelu` to `args.cache_ffn_output`. If you add new forward caches, VERIFY they're actually written!

26. **GRIM-text vs G.R.I.M Delegate Systems - CRITICAL SEPARATION**: GRIM-text is a SEPARATE PROGRAM from G.R.I.M and MUST NOT depend on G.R.I.M's core libraries:

- **WRONG**: `#include "../../../../core/delegate.hpp"` in GRIM-text files - this is G.R.I.M's main program delegate system
- **CORRECT**: GRIM-text does NOT use delegates. The GPU delegate system (`Shared/Delegate/Delegate.hpp`) was deleted (Rule 26: zero registered callbacks). If GPU-side event callbacks are needed in the future, re-implement from scratch with actual callers.
- **YAGNI Principle**: If a GRIM-text component declares delegate/callback functionality but NO code registers callbacks, DELETE the delegate code entirely
- **Files affected**: StreamController and any Shared/ components - they should NOT use any delegate pattern
- **Rationale**: GRIM-text trains/runs as standalone executable (`train_gpu.exe`, `grim_text_server.exe`), completely independent of G.R.I.M's main process
- **Build separation**: GRIM-text builds in `resources/models/GRIM-text/training/`, G.R.I.M builds in root `build/` - NO cross-dependencies allow

24. **GPU Gradient Norm Sync Performance**: `computeGradNorm()` has GPU kernel implementation but syncs CPU-GPU by default for logging. This adds **7+ seconds per batch** on 3080Ti when called every iteration:

- **Root cause**: `cudaStreamSynchronize()` forces CPU to wait for GPU completion to read metrics (total_norm, component norms)
- **Solution**: Pass `sync_for_host_read=false` to skip sync when metrics not needed (9 out of 10 batches)
- **When to sync**: Only when gradient component logging needed (every 10 batches) or for debugging
- **Gradient clipping**: Does NOT need sync - clipping happens on GPU using device-resident norms
- **Performance gain**: 10-15x speedup per batch (from 15s to 1-2s) by eliminating unnecessary CPU-GPU synchronization
- **Implementation**: `Phase2_TrainingLoop.cu` syncs every 10th batch via `computeGradNorm(batch_idx % 10 == 0)`

56. **Autograd Function Return Statements (Issue #56, Jan 2026)**: When writing autograd-enabled forward functions that build computation graphs, **ALWAYS return the output Tensor**. A missing return statement causes the local Tensor's destructor to run at function end, which cascades deletion of the entire `grad_fn` chain DURING the forward pass. Symptom: "illegal memory access" in backward kernels because `grad_fn` pointers point to freed memory. Fix: Always explicitly `return output;` from forward functions. C++ compilers often don't warn about missing return statements returning non-void (undefined behavior). The FFN::forward() bug in `Feed_Forward_GPU.cu` caused training crashes for weeks until discovered.

57. **NumericHead Backward NEVER Called (Issue #71, Jan 2026)**: When using a multi-task loss (e.g., text CE + numeric Huber), each loss path needs its own backward() call! The numeric loss kernel (`launchNumericLoss`) computed `grad_predictions` and wrote them to `grad_numeric_tensor.data`, but `numeric_head_output.backward()` was NEVER called! Result: `numeric_head_weights` received ZERO gradients for entire training - the numeric head parameters were NOT trained at all. Fix in `LanguageModel_Training.cu::backward()`: After `loss_tensor.backward(nullptr)`, check if numeric head is enabled and call `numeric_head_output.backward(&grad_numeric_tensor)`. **PATTERN**: For ANY auxiliary loss head (numeric, classification, etc.), you MUST explicitly call `.backward()` on that head's output tensor with the corresponding loss gradients.

58. **Numeric Loss Gradients Missing Mean Reduction (Issue #74, Jan 2026)**: The numeric loss was computed as `loss_avg = loss_sum / count` (proper mean reduction), but the **gradients** were NOT scaled by `1/count`! The `numericLossKernel` wrote `grad_predictions[idx] = grad * loss_weight` without the `1/count` factor. By chain rule: `d(loss_avg)/d(pred) = (1/count) * d(loss_sum)/d(pred)`. Without scaling, numeric gradients were ~3500x too large (matching the token count). This caused `num` gradient norm to dominate (2641 vs ~3 for other components). Fix in `NumericLoss_GPU.cu`: Added `scaleNumericGradKernel` that runs after the loss kernel and multiplies all gradients by `1.0f / count`. **PATTERN**: When using mean reduction for loss, gradients MUST also be scaled by `1/N` where N is the count of valid elements.

137. **Numeric Head Gradient Norm Dominated by Dense Accumulation (Issue #137, Feb 2026)**: `num=10.4` was caused by THREE compounding issues:

- **Bug 1 — `scaleNumericGradKernel` had bogus `1/sqrt(valid_text_tokens)` factor**: Issue #136 added this heuristic to "compensate" for text/numeric token count mismatch. The comment claimed formula was `1/(valid_count * valid_text_tokens)^0.5` but code computed `(1/valid_count) * (1/sqrt(valid_text_tokens))` — neither is correct. Proper mean reduction is just `1/valid_text_tokens`. The `1/sqrt(N)` over-suppressed numeric head weight gradients by ~82x, making the numeric head barely learn.
- **Bug 2 — `log_var` gradients are unscaled O(loss)**: `grad = -loss * weight + 0.5` gives magnitude ~11.5 when loss=12. These single scalars registered as `ParamGroupType::NUMERIC_HEAD` dominated the norm, causing the actual numeric head weights to get clipped every batch despite having tiny gradients (~0.009).
- **Bug 3 — Dense accumulation asymmetry**: After fixing bugs 1-2, `num` was still ~1.18 (vs `emb_lm_tied=0.495`). Root cause: numeric head's matmul backward sums ~500 atom gradient contributions into just 768 params (DENSE), while text LM head distributes across 50K vocab rows (SPARSE). With highly correlated encoder outputs at atom positions (cos≈0.85), the sum grows almost linearly with N_atoms instead of sqrt(N_atoms).
- **Fix (3 parts)**:
    1. `scaleNumericGradKernel`: Uses `1/valid_text_tokens` (same denominator as text CE mean reduction). `valid_text_tokens` passed directly from host (no device sync).
    2. `AutogradTraining.cu`: Normalized log_var gradients by `1/(1 + loss²)` to bound their contribution to ~0.11 (negligible vs weight norm).
    3. `AutogradTraining.cu`: After `numeric_head_output.backward()`, post-scales weight+bias gradients by `sqrt(N_atoms / valid_tokens)`. This normalizes the dense accumulation variance to match text's effective accumulation density. Does NOT affect grad_encoder (already propagated). `cached_numeric_count` stored from `computeLossBatch()`.
- **Files Modified**: `NumericLoss_GPU.cu`, `NumericLoss_GPU.hpp`, `ComputeLossBatch.cu`, `AutogradTraining.cu`, `TrainingState_GPU.hpp`
- **Expected Result**: With ~500 atoms and ~6669 valid tokens: `sqrt(500/6669)≈0.27`, giving `num≈0.32` (comparable to `emb_lm_tied=0.495`). `numeric_clipped=NO`.

74b. **Issue #74 Scaling Kernel Race Condition (Jan 2026)**: The Issue #74 fix added `scaleNumericGradKernel` after `numericLossKernel`, but CUDA kernels launch asynchronously! The scaling kernel was reading `count` BEFORE the loss kernel finished writing it via `atomicAdd`. Result: `count=0` or partial count → scaling either skipped (n>0 check fails) or incorrect. Fix: Added `cudaStreamSynchronize(stream)` between the two kernel launches to ensure the loss kernel completes before the scaling kernel reads the count. **PATTERN**: When kernel B reads data written by kernel A via atomics, you MUST sync between them even on the same stream.

82. **RMSNorm Gamma Gradient Token Normalization (Issue #82, Jan 2026) - REVERTED**: The original Issue #82 added `1/tokens` scaling to RMSNorm gamma gradients, claiming they were ~400x larger than FFN gradients. This was **WRONG** because: (1) the loss backward already applies `1/tokens` through mean reduction (Issue #58), (2) the incoming `dy` to RMSNorm already reflects this scaling, (3) adding another `1/tokens` made RMS gradients `1/tokens²` too small (~300,000x too small with 3500 tokens). The correct gradient is simply: `atomicAdd(&grad_gamma[i], dy[i] * x[i] * inv_rms)` - NO `inv_tokens` scaling. **PATTERN**: Don't double-apply mean reduction scaling - if loss backward already scales by `1/N`, don't scale again in parameter gradient kernels.

83. **FlashAttention dQ/dK vs dV Magnitude Asymmetry (Issue #83, Jan 2026) - SUPERSEDED BY #87**: FlashAttention backward produces dQ/dK that are **500,000x larger** than dV! Root cause: Softmax backward Jacobian asymmetry with ALiBi localized attention:

- **dV = P^T @ dO** - Uses softmax probabilities P directly (bounded 0-1)
- **dQ = dS @ K, dK = dS^T @ Q** - Where `dS = P * (dP - dP_sum)` involves Jacobian (unbounded)
- With ALiBi's ~32-token attention window, the Jacobian creates residual gradients that don't cancel
- Result: dQ_rms ≈ 0.06, dK_rms ≈ 0.02, dV_rms ≈ 0.000001 (500,000x ratio!)
- This asymmetry flows into W_qkv weight gradients → attention grad norm ~30,000 vs FFN ~2
- **Original Fix**: After FlashAttention backward, scale dQ/dK to match dV magnitude: `dq_scale = dv_rms / dq_rms`.
- ⚠️ **SUPERSEDED**: Issue #84 fixed the ROOT CAUSE (missing preprocessing kernel). Issue #83's normalization then CRUSHED gradients instead of fixing them. **Issue #87 removed Issue #83 normalization.**

84. **FlashAttention Missing Preprocessing Kernel (Issue #84, Jan 2026)**: **ROOT CAUSE OF GRADIENT EXPLOSION!** GRIM's FlashAttention backward was missing the preprocessing kernel that computes `dP_sum = dot(dO, O)` for all query positions. The main backward kernel (`flash_bwd_dq_dk_dv_loop_kernel`) uses `Is_first` template parameter to conditionally call `dot_do_o()` inline, but `Is_first=true` only for the FIRST column block. The kernel then reads `gdPsum` for ALL m_blocks, expecting valid pre-computed values. Without the preprocessing kernel, `dsoftmax_sum` buffer (allocated via `cudaMalloc` with NO zeroing) contained garbage for most positions.

- **Symptom**: dQ/dK explode 100,000-500,000x while dV stays normal. First 2 layers per batch are correct (starting m_blocks get valid data from inline `dot_do_o`), layers 3-12 explode.
- **Evidence**: `training_run.log` shows calls 1-2 with dQ_max=0.000001, calls 3-12 with dQ_max=0.4-0.7 despite identical dO_max=0.000002.
- **Fix in `Flash_Attention_Kernal.cu`**: Added `flash_bwd_dot_do_o_kernel` template and modified `run_flash_bwd` to launch preprocessing kernel FIRST with grid `(num_m_block, batch, heads)` before main backward kernel.
- **CRITICAL**: This preprocessing kernel call is REQUIRED by Tri Dao's FlashAttention library. The reference implementation (`flash_bwd_launch_template.h`) clearly shows `flash_bwd_dot_do_o_kernel<<<grid_m, ...>>>` launched before `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<<<grid_n, ...>>>`.

85. **Validation Token Budget Exceeds Training Buffer Size (CRITICAL FIX Jan 2026)**: Training allocates GPU buffers based on `batch_size * max_seq_len` (e.g., 7×1024=7168 tokens). But validation used hardcoded `kDefaultMaxTokensPerBatch = 8192`. When validation batch had >7168 tokens, it wrote past buffer boundaries → STATUS_STACK_BUFFER_OVERRUN crash (exit code -1073740791 / 0xC0000409).

- **Symptom**: Training completes successfully (3000+ batches), then crashes immediately after "Created N validation batches" message.
- **Root Cause**: `Phase2_TrainingLoop.cu` used `val_opts.max_tokens_per_batch = kDefaultMaxTokensPerBatch` (8192) which exceeds `max_cached_tokens` (7168 for batch_size=7, seq_len=1024).
- **Fix**: Changed validation to use `ctx.model->getConfig().max_tokens_per_batch` instead of hardcoded constant. Added logging: `"[Val] Token budget: X (model limit: Y)"`.
- **PATTERN**: When reusing buffers allocated for training, validation MUST respect the same size limits. Never use hardcoded constants that could exceed allocated buffer sizes.

87. **Issue #83 dQ/dK Normalization CRUSHING Gradients (Issue #87, Jan 2026)**: After Issue #84 fixed the root cause of gradient explosion (missing preprocessing kernel), Issue #83's dQ/dK normalization became **HARMFUL**. It was scaling dQ/dK DOWN to match tiny dV magnitude, causing vanishing gradients:

- **Symptom**: attn gradients 1.96→0.08 (24x decrease), ffn gradients 1.83→0.07 (26x decrease), loss NOT improving (actually increasing!)
- **Root Cause**: Issue #83 computed `dq_scale = dv_rms / dq_rms`. When dQ/dK were at proper magnitude (Issue #84 fix) but dV is tiny (~1e-6), this CRUSHED attention gradients.
- **Fix in `TensorContract_GPU.cu`**: Disabled Issue #83 normalization entirely. Now all gradients use scale=1.0.
- **PATTERN**: When a bandaid fix is superseded by a root cause fix, REMOVE the bandaid immediately. It will become harmful.

88. **Embedding Backward Skip Flag Was WRONG (Issue #88→#109→#110, Feb 2026)**: Issue #88 set `g_skip_embedding_backward_for_tied_weights = true` thinking LM head and embedding gradients "cancel". Issue #109 set it to `false`, but that STILL caused cancellation without PCGrad. Issue #110 re-enables PCGrad buffer allocation.

- **CRITICAL**: The ROOT CAUSE was Issue #87 incorrectly removing PCGrad allocation!
- **Why #88/#109 Were Both Wrong**:
    - LM head: `grad_W[i,j] = sum_t hidden[t,i] * grad_logits[t,j]` - DENSE MATMUL
    - Embedding: `atomicAdd(&grad_W[token_id[t], :], grad_encoder[t, :])` - SPARSE SCATTER to SAME buffer
    - With tied weights writing to SAME buffer, these gradients ARE OPPOSITE and DO CANCEL!
    - Issue #60 correctly implemented PCGrad to fix this, but Issue #87 removed it
- **Issue #110 Fix in `Phase1_Startup.cu`**: Call `allocatePCGradBuffer()` for tied weights.
- **Three EmbeddingGradFn paths**: (1) PCGrad mode [correct], (2) Skip mode [Issue #88], (3) Normal mode [cancels!]
- **Evidence**: Log showed `pcgrad_buffer=0000000000000000` (NULL) for all calls, forcing PATH 3 which says "(will cancel!)" in comment.

138. **computeGradNorm Timing Variance Was Pipeline Drain (DIAGNOSTIC FIX Feb 2026)**: `computeGradNorm` wall time varied 3ms-53ms between batches. The 50ms spikes occurred because `cudaStreamSynchronize` inside `computeGradNorm` was draining leftover backward pass kernels from the GPU pipeline — NOT because the norm kernels themselves were slow.

- **Fix**: Added CUDA event-based timing (`cudaEventRecord` before/after norm kernels, `cudaEventElapsedTime`) to decompose the log into `kernel=Xms` (actual GPU norm computation) and `drain=Yms` (waiting for backward pipeline).
- **Files Modified**: `LanguageModel_Training.cu` (event creation + timing), `grim_language_model_cuda.hpp` (added `last_gpu_norm_ms_` member), `Phase2_TrainingLoop.cu` (updated log format)
- **Lesson**: Wall-time measurements of GPU operations are misleading when the stream has pending work from prior stages.

139. **Per-Component Gradient Clipping (emb_lm_tied Crushing Encoder Gradients - Issue #139, Feb 2026)**: `emb_lm_tied` gradient norm dominated 88-99.6% of total gradient norm across ALL batches, drowning out attention, FFN, and RMSNorm components. The old clipping (Issue #134) grouped ALL text parameters (emb+attn+ffn+rms) and applied a single clip coefficient — when emb was 99% of text_norm, the clip_coef ≈ 0.2 scaled ALL text params equally, crushing encoder gradients to near-zero.

- **Root Cause**: LM head gradient `grad_B = hidden^T @ grad_logits` has shape [50377×768] = 38.7M parameters. With high loss (~10-12), softmax gradients are large. The matmul across all token positions amplifies this. Encoder gradients (attn/ffn/rms combined) were 100-1000x smaller in magnitude.
- **Symptom**: Batches 1-8: emb% = 60-72% (acceptable). Batches 9-14: emb% climbs to 83-95%. Batches 15+: emb% = 99.4-99.6%, attn+ffn+rms combined < 0.5, encoder layers barely learning.
- **Fix**: Three independent clips instead of two:
    1. **emb clip** — clips LM_HEAD (+ EMBEDDING if untied) independently
    2. **enc clip** — clips ATTENTION + FFN + RMSNORM + SCRATCHBLOCK independently
    3. **num clip** — clips NUMERIC_HEAD independently
- Each component gets its own full budget (`effective_per_token_limit`). Encoder gradients can no longer be crushed by embedding spikes.
- **Files Modified**: `Phase2_TrainingLoop.cu` (post-accumulation clipping section)
- **Expected Result**: Encoder components maintain their natural gradient magnitude. `attn=0.03, ffn=0.07, rms=0.01` survive clipping intact instead of being scaled by 0.2.
- **PATTERN**: When one parameter group dominates L2 norm, NEVER clip it jointly with smaller groups. Use per-component independent clipping.

140. **Embedding Scale sqrt(d_model) Removed for Tied Weights (ROOT CAUSE FIX - Issue #140, Feb 2026)**: The `sqrt(d_model) = 27.7` embedding scaling (Issue #92, AIAYN paper) created a **27.7x gradient asymmetry** between the embedding and LM head backward paths when using tied weights. This was the ROOT CAUSE of escalating `emb_lm_tied` gradient magnitude (spikes growing from 0.61→5.23 over 28 batches) and structural-token weight row divergence (`||W||_max` monotonically growing 0.18→0.32).

- **Root Cause**: With `tie_embeddings=true`, both embedding and LM head operate on the SAME weight W:
    - **Embedding forward**: `emb = W[tok] * sqrt(d_model)` → backward chain rule: `grad_W_emb[tok] += grad_encoder[t] * 27.7`
    - **LM head forward**: `logits = centered @ W^T` (raw W, no scaling) → backward: `grad_W_lm = centered^T @ grad_logits` (no extra factor)
    - The embedding path carries a **27.7x amplification** that the LM head path does not
    - For structural tokens (BOS=token 0, atom placeholder=token 400) appearing every batch at fixed positions, the amplified embedding gradient accumulates via AdamW momentum
    - This caused those weight rows to diverge: `||W||_max` grew monotonically while `||W||_mean` stayed flat at 0.175
- **Why AIAYN scaling doesn't apply to GRIM-text**:
    - AIAYN's sqrt(d_model) scaling was designed to make embeddings large relative to **sinusoidal position encodings** in the residual stream
    - GRIM-text uses ALiBi/RoPE which inject position **INSIDE attention**, NOT in the residual stream
    - No competing position encoding exists → the scaling has no purpose
    - Modern LLMs with tied weights (GPT-2, LLaMA, Mistral, Gemma) do NOT scale embeddings by sqrt(d_model)
- **Fix**: Changed `embedding_scale = sqrt(d_model)` → `embedding_scale = 1.0f` in `AutogradTraining.cu`
- **Effect**: Embedding backward gradient magnitude matches LM head backward magnitude (both operate on raw W). Eliminates the 27.7x asymmetry that caused structural-token weight row divergence.
- **History**: Issue #92 added scaling → Issue #102 disabled (LSE explosion) → Issue #106 re-enabled after W_qkv init fix → **Issue #140 removes permanently** (gradient asymmetry for tied weights)
- **Files Modified**: `AutogradTraining.cu` (embedding_scale = 1.0f)
- **IMPORTANT**: Requires retraining from scratch (weight magnitudes fundamentally change). Old checkpoints are incompatible.

141. **ScratchBlock Backward Never Called + Config Hardcoded (CRITICAL FIX - Issue #141, Feb 2026)**: Three bugs fixed:

- **BUG A (CRITICAL): ScratchBlock backward NEVER executed.** `AutogradTraining.cu` guarded ScratchBlock backward with `intermediates.embedding_tensor.has_grad()` which was ALWAYS false. The embedding_tensor is a dropout output (non-leaf). Autograd writes gradients to `DropoutGradFn::input_grad` (internal buffer), not to `tensor.grad_data()`. `atom_projection_`, `atom_type_embeddings_`, `text_feature_projection_` received ZERO gradients for entire training.
    - **Fix**: Added `grad_output_tap` field to base `GradFn` struct. `DropoutGradFn::apply()` copies `grad_output` (encoder input gradient) into the tap buffer before applying dropout mask. `Phase1_Startup.cu` allocates `scratchblock_grad_tap` buffer in TrainingState. Backward code sets tap before `loss_tensor.backward()`, then uses captured gradient for ScratchBlock backward.
    - **Files Modified**: `TensorContract_GPU.hpp` (GradFn tap fields), `TensorContract_GPU.cu` (DropoutGradFn tap copy), `AutogradTraining.cu` (tap setup + ScratchBlock backward rewrite), `TrainingState_GPU.hpp` (tap buffer), `Phase1_Startup.cu` (tap allocation)

- **BUG B: `positional_encoding` config HARDCODED.** `Phase1_Startup.cu` line 748 always set `model_config.positional_encoding = DEFAULT_POSITIONAL_ENCODING` (ALIBI_ROPE). JSON config's `positional_encoding.use_learned/use_rope/use_alibi` were NEVER parsed. `parsePositionalEncodingType()` was dead code. Rule 20 violation.
    - **Fix**: Added parsing of `positional_encoding` object from JSON in `loadConfiguration()`. Added `PositionalEncodingType positional_encoding` field to `ModelArchitecture` struct. `initializeModel()` uses parsed value.
    - **Files Modified**: `Phase1_Startup.cu` (JSON parsing + model init), `HyperParameters_GPU.hpp` (ModelArchitecture field)

- **BUG C: PCGrad NaN on zero-norm rows.** `kernel_pcgrad_combine` divided by `||g_lm||²` with only `assert()` guard (compiled out in Release). Zero-norm rows produce `0/0 = NaN`.
    - **Fix**: Replaced `assert` with `if (total_norm_sq < 1e-12f) { s_proj_coef = 0.0f; }` guard.
    - **Files Modified**: `TensorContract_GPU.cu`

- **Also deleted**: ~170 lines of dead Issue #93/#95 diagnostic code inside `if (use_learned_pos_emb)` block in `AutogradTraining.cu` that never executed with ALIBI_ROPE.

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
