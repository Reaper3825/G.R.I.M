# GRIM Self-Training Pipeline

A complete 5-stage autonomous training system for the GRIM AI model. This pipeline automatically collects, verifies, parses, trains, and deploys updated field adapters.

## Architecture

### End-to-End Flow
```
fetch_online_data()
   → verify_sources()
      → parse_verified_data()
         → train_field_adapter()
            → deploy_adapter()
```

## Stages

### Stage 1 — Collect (`collector.cpp/hpp`)

**Goal:** Fetch fresh online data for potential training.

**Functions:**
- `fetch_online_data()` - Main entry point
- Pulls from pre-approved sources (news APIs, tech docs, GitHub READMEs)
- Stores raw text and metadata (source URL, timestamp, author)
- Writes output to `data/raw/*.jsonl`

**Output:** `data/raw/*.txt` or `*.jsonl` → raw text corpus

**Key Features:**
- Configurable data sources
- API authentication support
- Keyword filtering
- Rate limiting and timeout handling
- Multiple source types (News API, GitHub, Tech Docs)

---

### Stage 2 — Verify (`verifier.cpp/hpp`)

**Goal:** Filter and validate information accuracy and reliability.

**Functions:**
- `verify_sources()` - Main entry point
- Checks source domain against whitelist/blacklist
- Cross-checks identical claims across ≥ 2 sources
- Drops low-confidence or malformed text
- Assigns reliability score (0–1), discards entries < 0.8

**Output:** `data/verified/verified.jsonl` — clean verified text with metadata

**Key Features:**
- Domain whitelisting/blacklisting
- Cross-reference validation
- Content quality checks
- Reliability scoring
- Similarity detection for corroboration

---

### Stage 3 — Parse (`parser.cpp/hpp`)

**Goal:** Convert verified text into structured, model-trainable examples.

**Functions:**
- `parse_verified_data()` - Main entry point
- Extracts key fields (topic, summary, source, timestamp, reliability)
- Formats records as JSONL or FlatBuffer
- Runs tokenizer sanity checks
- Supports multiple parsing strategies:
  - Question-Answer generation
  - Summarization
  - Text completion
  - Instruction-following

**Output:** `data/parsed/parsed.jsonl` — structured dataset ready for training

**Key Features:**
- Keyword and entity extraction
- Multiple parsing strategies
- Token length validation
- Metadata preservation
- Summary generation

---

### Stage 4 — Train (`trainer.cpp/hpp`)

**Goal:** Fine-tune GRIM's field/user model with new parsed data.

**Functions:**
- `train_field_adapter()` - Main entry point
- Loads `grim_user_current.gguf` or base weights
- Fine-tunes with parsed dataset
- Uses low learning rate (1e-5) and 1–2 epochs
- Saves adapter as `grim_user_temp.pt` or quantized `.gguf`
- Logs training stats and validation accuracy

**Output:** `models/grim_user_temp.pt` — fine-tuned adapter weights

**Key Features:**
- LoRA/adapter-based fine-tuning
- Configurable hyperparameters
- Validation split
- Checkpoint saving
- Learning rate warmup
- Training progress logging

---

### Stage 5 — Deploy (`trainer.cpp/hpp`)

**Goal:** Safely integrate the new adapter into active GRIM instance.

**Functions:**
- `deploy_adapter()` - Main entry point (in Trainer class)
- Runs evaluation on test prompts
- Confirms no regression (accuracy > threshold)
- Replaces current field model if passed
- Archives previous adapter to `models/archive/`
- Cleans up temporary data

**Output:**
- `models/grim_user_current.gguf` — updated GRIM field model
- `models/archive/grim_user_TIMESTAMP.gguf` — archived backups

**Key Features:**
- Regression testing
- Automated archiving
- Safe model replacement
- Cleanup of temporary data
- Rollback support

---

## Pipeline Orchestrator (`pipeline.cpp/hpp`)

Coordinates all 5 stages with:
- Sequential execution with error handling
- Partial pipeline runs (specific stage ranges)
- Configuration management (load/save JSON)
- Stage callbacks for progress monitoring
- Comprehensive statistics tracking
- Logging to file and console

### Usage Examples

#### 1. Run Complete Pipeline (Default Config)
```cpp
#include "pipeline.hpp"

auto pipeline = std::make_unique<Pipeline>();
bool success = pipeline->run();
```

#### 2. Run with Custom Configuration
```cpp
PipelineConfig config;
config.collector.max_entries_per_source = 200;
config.verifier.min_reliability_threshold = 0.85;
config.trainer.learning_rate = 5e-6;
config.trainer.num_epochs = 3;
config.auto_deploy = false;  // Manual deployment

auto pipeline = std::make_unique<Pipeline>(config);
bool success = pipeline->run();
```

#### 3. Run Individual Stage
```cpp
auto pipeline = std::make_unique<Pipeline>();
pipeline->run_stage(Pipeline::Stage::COLLECT);
pipeline->run_stage(Pipeline::Stage::VERIFY);
```

#### 4. Run Partial Pipeline
```cpp
auto pipeline = std::make_unique<Pipeline>();
// Run only verify → parse → train
pipeline->run_partial(Pipeline::Stage::VERIFY, Pipeline::Stage::TRAIN);
```

#### 5. Load Configuration from File
```cpp
auto pipeline = std::make_unique<Pipeline>();
pipeline->load_config("config/training_config.json");
pipeline->run();
```

#### 6. Set Stage Callbacks
```cpp
auto pipeline = std::make_unique<Pipeline>();
pipeline->set_stage_callback([](Pipeline::Stage stage, bool success) {
    std::cout << "Stage " << stage_to_string(stage) 
              << (success ? " ✓" : " ✗") << std::endl;
});
pipeline->run();
```

---

## Configuration File Format

Example `training_config.json`:

```json
{
  "collector": {
    "output_dir": "data/raw",
    "max_entries": 200,
    "timeout": 30,
    "save_as_jsonl": true
  },
  "verifier": {
    "input_dir": "data/raw",
    "output_dir": "data/verified",
    "min_reliability": 0.85,
    "cross_check": true,
    "min_cross_refs": 3
  },
  "parser": {
    "input_dir": "data/verified",
    "output_dir": "data/parsed",
    "max_tokens": 2048,
    "extract_entities": true,
    "extract_keywords": true
  },
  "trainer": {
    "input_dir": "data/parsed",
    "model_dir": "models",
    "learning_rate": 0.00001,
    "num_epochs": 2,
    "batch_size": 4
  },
  "deployment": {
    "min_accuracy": 0.85,
    "run_tests": true,
    "cleanup": true
  },
  "auto_deploy": true,
  "stop_on_error": true,
  "log_file": "logs/pipeline.log"
}
```

---

## Directory Structure

```
G.R.I.M/
├── training/
│   ├── collector.hpp/cpp       # Stage 1: Data Collection
│   ├── verifier.hpp/cpp        # Stage 2: Source Verification
│   ├── parser.hpp/cpp          # Stage 3: Data Parsing
│   ├── trainer.hpp/cpp         # Stage 4 & 5: Training & Deployment
│   ├── pipeline.hpp/cpp        # Pipeline Orchestrator
│   └── main_pipeline.cpp       # Example Usage
├── data/
│   ├── raw/                    # Stage 1 output
│   ├── verified/               # Stage 2 output
│   ├── parsed/                 # Stage 3 output
│   └── test_prompts.json       # Test cases for deployment
├── models/
│   ├── grim_user_current.gguf  # Active field model
│   ├── grim_user_temp.pt       # Newly trained adapter
│   └── archive/                # Previous model backups
└── logs/
    ├── pipeline.log            # Pipeline execution log
    └── training/               # Training logs
```

---

## Dependencies

- **nlohmann/json** - JSON parsing
- **libcurl** - HTTP requests for data collection
- **C++17** or later
- **CMake** 3.15+

---

## Building

Add to your `CMakeLists.txt`:

```cmake
# Training pipeline
add_library(grim_training
    training/collector.cpp
    training/verifier.cpp
    training/parser.cpp
    training/trainer.cpp
    training/pipeline.cpp
)

target_link_libraries(grim_training
    nlohmann_json::nlohmann_json
    CURL::libcurl
)

# Example executable
add_executable(grim_pipeline training/main_pipeline.cpp)
target_link_libraries(grim_pipeline grim_training)
```

Build:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

---

## Running the Pipeline

### Command Line Usage

```bash
# Run with defaults
./grim_pipeline

# Run with config file
./grim_pipeline config/training_config.json

# Run single stage
./grim_pipeline --stage collect
./grim_pipeline --stage verify
./grim_pipeline --stage train
./grim_pipeline --stage deploy

# Run with custom settings
./grim_pipeline --custom
```

---

## Pipeline Statistics

After running, get comprehensive stats:

```cpp
auto stats = pipeline->get_stats();

std::cout << "Data collected: " << stats.data_collected << std::endl;
std::cout << "Data verified: " << stats.data_verified << std::endl;
std::cout << "Examples parsed: " << stats.examples_parsed << std::endl;
std::cout << "Training accuracy: " << stats.training_stats.best_val_accuracy << std::endl;
std::cout << "Deployment: " << (stats.deployment_successful ? "SUCCESS" : "FAILED") << std::endl;
```

---

## Safety Features

1. **Domain Whitelisting** - Only trusted sources
2. **Cross-Verification** - Multiple source corroboration
3. **Reliability Scoring** - Automatic quality filtering
4. **Regression Testing** - Pre-deployment validation
5. **Automatic Archiving** - Rollback capability
6. **Configurable Thresholds** - Customizable safety levels
7. **Error Handling** - Graceful failures with logging
8. **Manual Override** - Disable auto-deploy for review

---

## Advanced Features

### Custom Data Sources

```cpp
Collector collector;

DataSource github_source;
github_source.url = "https://api.github.com/repos/org/repo/readme";
github_source.source_type = "github";
github_source.requires_auth = true;
github_source.api_key = "ghp_xxxxx";
github_source.priority = 9;

collector.add_source(github_source);
```

### Custom Parsing Strategies

```cpp
ParserConfig config;
config.strategy = ParserConfig::ParseStrategy::QUESTION_ANSWER;
// or SUMMARIZATION, COMPLETION, INSTRUCTION

Parser parser(config);
```

### LoRA Configuration

```cpp
TrainerConfig config;
config.use_lora = true;
config.lora_rank = 16;
config.lora_alpha = 32.0;
config.target_modules = {"q_proj", "v_proj", "k_proj", "o_proj"};
```

---

## Future Enhancements

- [ ] Real model loading (GGUF integration)
- [ ] Actual tokenizer integration
- [ ] GPU training support
- [ ] Distributed training
- [ ] Real-time monitoring dashboard
- [ ] Automatic hyperparameter tuning
- [ ] A/B testing of models
- [ ] Multi-language support

---

## License

Part of the GRIM AI system.

---

## Contact

For issues or questions about the self-training pipeline, please refer to the main GRIM documentation.
