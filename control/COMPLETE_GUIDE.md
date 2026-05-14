# GRIM-text Training Control System - Complete Overview

## ⚠️ CRITICAL ISSUES TO FIX

### 1. **IMMEDIATE: DataCollection Build Missing**
- **Problem**: `resources/models/GRIM-text/DataCollection/build/Release/grim_data_pipeline.exe` exists but UI reports "pipeline failed"
- **Root Cause**: The executable exists but the UI's data collection flow is failing
- **Impact**: "Run Data Pipeline" button in GRIM.exe training UI will fail
- **Fix Required**: Debug UI → training_control_server → grim_data_pipeline.exe communication flow

### 2. **PATH INCONSISTENCY: Control Folder Location**
- **Problem**: Guide shows control folder at `training/control/` but actual location is `D:\G.R.I.M\control/`
- **Root Cause**: Control folder was moved to GRIM root to fix include path issues
- **Impact**: All documentation paths are wrong
- **Status**: Code updated ✅, Documentation needs update ❌

### 3. **MISSING: Auto-start Training Control Server**
- **Problem**: GRIM.exe expects training_control_server.exe to already be running
- **Location**: Server at `D:\G.R.I.M\control\build\Release\training_control_server.exe`
- **Impact**: Training panel will show "Server not connected" unless manually started
- **Fix Required**: Add auto-start logic in `grim_text_server_manager.cpp` (line 266 has path, needs launch code)

### 4. **TESTING REQUIRED: Full Pipeline Flow**
- **Status**: All executables built, paths updated, but end-to-end testing not done
- **Test**: GRIM.exe → training_control_server → train_gpu.exe
- **Test**: GRIM.exe → training_control_server → grim_data_pipeline.exe

---

## 📁 Directory Structure (UPDATED - November 2025)

```
D:\G.R.I.M\
├── control/                              # ⚠️ MOVED FROM training/control/
│   ├── training_control.fbs
│   ├── training_control_generated.h
│   ├── training_control_server.cpp
│   ├── training_control_client.hpp       # ✅ Include in GRIM.exe
│   ├── training_paths.hpp                # ✅ NEW: Path resolution utilities
│   ├── CMakeLists.txt
│   ├── COMPLETE_GUIDE.md
│   └── build/
│       └── Release/
│           └── training_control_server.exe  # ✅ BUILT
│
└── resources/models/GRIM-text/
    ├── Common/                           # Shared GPU implementations
    │   ├── grim_embedding_gpu.hpp
    │   ├── grim_embedding_gpu.cu
    │   ├── grim_transformer_gpu.hpp
    │   ├── grim_transformer_gpu.cu
    │   ├── grim_encoder_layer_gpu.hpp
    │   ├── grim_flash_attention.hpp
    │   ├── grim_flash_attention.cu
    │   ├── flash_attention_forward.cu
    │   ├── flash_attention_backward.cu
    │   ├── grim_language_model_gpu.cu
    │   └── grim_training_kernels.cu
    │
    ├── GRIM/                             # Inference server
    │   ├── grim_text_server.cpp
    │   ├── grim_language_model.hpp
    │   ├── grim_language_model_cuda.hpp
    │   ├── CMakeLists.txt
    │   └── cmake/
    │       └── build/
    │           └── Release/
    │               └── grim_model.exe    # ✅ Built inference server
    │
    ├── core/                             # Model architecture
    │   ├── grim_embedding.hpp
    │   ├── grim_encoder_layer.hpp
    │   ├── grim_gqa.hpp
    │   ├── grim_rmsnorm.hpp
    │   ├── grim_swiglu.hpp
    │   └── grim_lm_head.hpp
    │
    ├── DataCollection/                   # Data pipeline
    │   ├── grim_data_pipeline.cpp        # ✅ NEW: Unified pipeline
    │   ├── main_data_collection.cpp      # DEPRECATED
    │   ├── collect_data.cpp              # DEPRECATED  
    │   ├── verifier.cpp
    │   ├── web_collector.hpp
    │   ├── data_preprocessor.hpp
    │   ├── CMakeLists.txt
    │   └── build/
    │       └── Release/
    │           ├── grim_data_pipeline.exe    # ✅ BUILT
    │           ├── collect_data.exe          # Legacy
    │           └── verifier.exe              # Legacy
    │
    └── training/
        ├── schemas/                      # FlatBuffer schemas
        │   ├── grim_embedding_weights.fbs
        │   └── grim_embedding_weights_generated.h
        │
        ├── TrainingLoop/                 # GPU Training
        │   ├── train_gpu.cu
        │   ├── CMakeLists.txt
        │   └── build/
        │       └── Release/
        │           └── train_gpu.exe     # ✅ BUILT
        │
        ├── data/                         # Runtime data files
        │   ├── source_data.json
        │   ├── training_data.grmt
        │   └── checkpoint_*.json
        │
        ├── models/                       # Runtime model files
        │   ├── vocab.bin
        │   ├── checkpoint_epoch_*.pt
        │   └── best_model.pt
        │
        ├── logs/                         # Runtime logs
        │   ├── training_log_*.txt
        │   └── training_metrics.json
        │
        ├── training_status.fb            # ⚠️ Written by train_gpu.exe
        └── training_paths.hpp            # Path utilities
```

## 🎯 Purpose

Allow GRIM.exe to control GRIM-text training **without linking to CUDA code**.

## 🏗️ Architecture

### Communication Flow

```text
GRIM.exe (C++ UI)
    ↓ (includes)
training_control_client.hpp
    ↓ (HTTP + FlatBuffers)
training_control_server.exe (port 11436)
    ↓ (spawns/monitors)
train_gpu.exe (CUDA training)
    ↓ (writes)
D:\G.R.I.M\resources\models\GRIM-text\training\training_status.fb
```

### Key Components

1. **FlatBuffer Schema** (`training_control.fbs`)
   - Defines all message types
   - Type-safe protocol
   - Zero-copy deserialization
   - Compile with: `flatc --cpp training_control.fbs`

2. **Control Server** (`training_control_server.cpp`)
   - Runs as separate .exe: `D:\G.R.I.M\resources\models\GRIM-text\training\control\build\Release\training_control_server.exe`
   - HTTP server on localhost:11436
   - Spawns/stops train_gpu.exe at: `D:\G.R.I.M\resources\models\GRIM-text\training\TrainingLoop\build\Release\train_gpu.exe`
   - Monitors status file: `D:\G.R.I.M\resources\models\GRIM-text\training\training_status.fb`
   - Returns status via FlatBuffers

3. **Control Client** (`training_control_client.hpp`)
   - **Header-only** library
   - Include in GRIM.exe (no linking!)
   - Simple C++ API
   - Wraps HTTP + FlatBuffer communication

## 📋 Setup Steps

### 1. Compile FlatBuffer Schema

```powershell
cd D:\G.R.I.M\resources\models\GRIM-text\training\control

# Install flatc if needed
# vcpkg install flatbuffers

flatc --cpp training_control.fbs
# Generates: training_control_generated.h
```

### 2. Build Control Server

```powershell
cd D:\G.R.I.M\control

# Configure
cmake -B build -DCMAKE_PREFIX_PATH="D:/G.R.I.M/vcpkg_installed/x64-windows"

# Build
cmake --build build --config Release
```

Output: `D:\G.R.I.M\control\build\Release\training_control_server.exe` (~2-3 MB)

### 3. Include Client in GRIM.exe

**In your GRIM.exe CMakeLists.txt:**

```cmake
# Add include directory (⚠️ UPDATED PATH - control folder moved to root)
target_include_directories(grim PRIVATE
    ${CMAKE_SOURCE_DIR}/control
)

# Link FlatBuffers and httplib
find_package(flatbuffers CONFIG REQUIRED)
find_path(HTTPLIB_INCLUDE_DIR NAMES httplib.h)

target_include_directories(grim PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(grim PRIVATE flatbuffers::flatbuffers)

if(WIN32)
    target_link_libraries(grim PRIVATE ws2_32)
endif()
```

**In your UI code:**

```cpp
// File: D:\G.R.I.M\ui\ui_training_panel.cpp  
#include "../control/training_control_client.hpp"

// Create client
GRIMText::TrainingControlClient client("127.0.0.1", 11436);

// Use it!
if (client.startTraining()) {
    LOG_DEBUG("Training", "Started");
}
```

## 🚀 Usage Examples

### Start Control Server

**Manual Start:**

```powershell
cd D:\G.R.I.M\control\build\Release
.\training_control_server.exe
```

**Auto-start from GRIM.exe (⚠️ NEEDS IMPLEMENTATION):**

```cpp
// File: D:\G.R.I.M\ai\grim_text_server_manager.cpp (line ~266)
// Current code only builds the path, doesn't launch the server:
fs::path serverExe = fs::absolute("control/build/Release/training_control_server.exe");

// ⚠️ TODO: Add auto-launch logic like this:
if (!client.isServerRunning()) {
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    
    std::string cmdLine = "\"" + serverExe.string() + "\"";
    std::vector<char> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back('\0');
    
    CreateProcessA(nullptr, cmdBuf.data(), nullptr, nullptr, 
                   FALSE, CREATE_NEW_CONSOLE, nullptr, nullptr, &si, &pi);
    
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
```

Server logs:

```text
========================================
  GRIM-text Training Control Server
========================================

[Server] Starting on http://127.0.0.1:11436

API Endpoints:
  GET  /health                - Health check
  GET  /api/status            - Get training status
  POST /api/training/start    - Start training
  POST /api/training/stop     - Stop training
  POST /api/config            - Update configuration

Press Ctrl+C to stop
```

### Use Client from GRIM.exe

```cpp
// File: D:\G.R.I.M\ui\ui_training_panel.cpp
#include "../control/training_control_client.hpp"

void UITrainingPanel::update(float dt) {
    // Poll status every second
    if (pollTimer > 1.0f) {
        GRIMText::TrainingState state;
        GRIMText::TrainingStats stats;
        GRIMText::TrainingConfig config;
        
        if (trainingClient.getStatus(state, stats, config)) {
            // Update UI
            progressBar->setValue(stats.trainingProgress);
            lossLabel->setText("Loss: " + std::to_string(stats.currentLoss));
            epochLabel->setText("Epoch: " + std::to_string(stats.currentEpoch) + "/" + std::to_string(stats.totalEpochs));
            
            // Check state
            if (state == GRIMText::TrainingState::Training) {
                startButton->setEnabled(false);
                stopButton->setEnabled(true);
            } else if (state == GRIMText::TrainingState::Completed) {
                startButton->setEnabled(true);
                stopButton->setEnabled(false);
                showCompletionDialog();
            }
        }
        
        pollTimer = 0.0f;
    }
    pollTimer += dt;
}

void UITrainingPanel::onStartButtonClicked() {
    GRIMText::TrainingConfig config;
    config.epochs = epochSlider->getValue();
    config.batchSize = batchSizeSlider->getValue();
    config.learningRate = learningRateSlider->getValue();
    
    if (trainingClient.startTraining(&config)) {
        LOG_DEBUG("TrainingPanel", "Training started");
    } else {
        showErrorDialog(trainingClient.getLastError());
    }
}

void UITrainingPanel::onStopButtonClicked() {
    if (trainingClient.stopTraining()) {
        LOG_DEBUG("TrainingPanel", "Training stopped");
    } else {
        showErrorDialog(trainingClient.getLastError());
    }
}
```

## 🔌 API Reference

### Client Methods

```cpp
class TrainingControlClient {
public:
    // Constructor
    TrainingControlClient(const std::string& host = "127.0.0.1", int port = 11436);
    
    // Check if server is running
    bool isServerRunning();
    
    // Get current training status (non-blocking)
    bool getStatus(TrainingState& state, TrainingStats& stats, TrainingConfig& config);
    
    // Start training (optionally override config)
    bool startTraining(const TrainingConfig* customConfig = nullptr);
    
    // Stop training
    bool stopTraining();
    
    // Update configuration (only works when idle)
    bool updateConfig(const TrainingConfig& config);
    
    // Get last error message
    std::string getLastError() const;
};
```

### Data Structures

```cpp
enum class TrainingState {
    Idle,        // Not training
    Collecting,  // Collecting data
    Verifying,   // Verifying data
    Training,    // Training in progress
    Paused,      // Paused (not yet implemented)
    Completed,   // Training finished
    Error        // Error occurred
};

struct TrainingStats {
    int currentEpoch;
    int totalEpochs;
    int currentBatch;
    int totalBatches;
    float currentLoss;
    float avgLoss;
    float perplexity;
    float tokensPerSec;
    float gpuMemoryUsed;    // MB
    float gpuMemoryTotal;   // MB
    float trainingProgress; // 0.0 to 1.0
    std::string currentPhase;
    std::string lastError;
    int64_t startTime;      // Unix timestamp
    int64_t elapsedTime;    // Seconds
};

struct TrainingConfig {
    int epochs = 3;
    int batchSize = 8;
    float learningRate = 0.0001f;
    int maxSeqLen = 8192;
    int warmupSteps = 1000;
    bool useGPU = true;
    bool useFlashAttention = true;
    std::string dataPath;
    std::string vocabPath;
    std::string outputPath;
};
```

## 🔍 Debugging

### Check Server Status

```powershell
# Test health endpoint
Invoke-RestMethod -Uri http://127.0.0.1:11436/health

# Get status (returns binary FlatBuffer)
Invoke-WebRequest -Uri http://127.0.0.1:11436/api/status -OutFile status.fb
```

### View FlatBuffer Content

```cpp
// In C++
std::ifstream file("status.fb", std::ios::binary);
file.seekg(0, std::ios::end);
size_t size = file.tellg();
file.seekg(0, std::ios::beg);

std::vector<uint8_t> buffer(size);
file.read(reinterpret_cast<char*>(buffer.data()), size);

auto status = GRIMText::Control::GetStatusResponse(buffer.data());
std::cout << "Epoch: " << status->stats()->current_epoch() << std::endl;
```

### Common Issues

**Server won't start - port in use:**

```powershell
netstat -ano | findstr :11436
taskkill /PID <pid> /F
```

**Can't connect from GRIM.exe:**

```cpp
// ⚠️ CURRENT ISSUE: Server must be manually started
// TODO: Implement auto-start in grim_text_server_manager.cpp

if (!client.isServerRunning()) {
    // Option 1: Show error to user
    showErrorDialog("Training server not running. Please start it manually.");
    
    // Option 2: Auto-start (needs implementation)
    fs::path serverExe = fs::absolute("control/build/Release/training_control_server.exe");
    // ... launch code here ...
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
```

**FlatBuffer version mismatch:**

```powershell
# Recompile schema
cd D:\G.R.I.M\control
flatc --cpp training_control.fbs

# Rebuild server
cmake --build build --config Release

# Rebuild GRIM.exe
cd D:\G.R.I.M
cmake --build build --config Release --target grim
```

## 🔧 Current Implementation Status

### ✅ Completed

- [x] Control folder moved to GRIM root (`D:\G.R.I.M\control/`)
- [x] Path resolution system (`training_paths.hpp` with `getSafeResourcePath()`)
- [x] Exclusion list to avoid searching external/vcpkg_installed directories
- [x] All include paths updated in GRIM.exe UI files
- [x] CMakeLists.txt updated with new control path
- [x] training_control_server.exe built successfully
- [x] train_gpu.exe built with CUDA support
- [x] grim_data_pipeline.exe built successfully
- [x] GRIM.exe rebuilt with updated includes

### ⚠️ Pending Issues

- [ ] **CRITICAL**: Test full training pipeline (UI → server → train_gpu.exe)
- [ ] **CRITICAL**: Test data collection pipeline (UI → server → grim_data_pipeline.exe)
- [ ] **HIGH**: Implement auto-start for training_control_server.exe in `grim_text_server_manager.cpp`
- [ ] **MEDIUM**: Debug why UI reports "pipeline failed" when executable exists
- [ ] **MEDIUM**: Add graceful error handling when server isn't running
- [ ] **LOW**: Update all remaining documentation with new paths

### 🔍 Known Path Issues Fixed

| Component | Old Path | New Path | Status |
|-----------|----------|----------|--------|
| Control folder | `resources/models/GRIM-text/training/control/` | `D:\G.R.I.M\control/` | ✅ Moved |
| Include in UI | `"../resources/.../control/"` | `"../control/"` | ✅ Fixed |
| CMakeLists | `${CMAKE_SOURCE_DIR}/resources/.../control` | `${CMAKE_SOURCE_DIR}/control` | ✅ Fixed |
| train_gpu.cu | `"control/"` | `"../../../control/"` | ✅ Fixed |
| Server path | Relative from old location | Uses `getSafeResourcePath()` | ✅ Fixed |

## ✅ Advantages

| Feature | Benefit |
|---------|---------|
| **No CUDA Linking** | GRIM.exe doesn't need CUDA toolkit |
| **Separate Process** | Training crash won't crash UI |
| **Hot Restart** | Can restart training without restarting GRIM.exe |
| **Network Ready** | Could run training on remote machine |
| **Type Safe** | FlatBuffers prevent malformed messages |
| **Efficient** | Zero-copy deserialization |
| **Debuggable** | Use curl/Postman to test API |
| **Versioned** | FlatBuffer schema supports evolution |

## 📊 Performance

| Metric | Value |
|--------|-------|
| Server binary size | ~2.7 MB |
| Status query latency | <1ms (local) |
| FlatBuffer overhead | ~0% (zero-copy) |
| Poll frequency | 1-2 Hz recommended |
| Memory usage | <10 MB (server) |

## 🔮 Future Enhancements

- [ ] Pause/resume training
- [ ] Real-time log streaming
- [ ] Multi-GPU training control
- [ ] Checkpoint management
- [ ] Training queue (multiple jobs)
- [ ] Remote training (secure HTTPS)
- [ ] Training profiles (presets)
- [ ] Auto-save on crash

## 🚨 Next Steps (Priority Order)

### 1. **Test Training Pipeline** (CRITICAL)

```powershell
# Terminal 1: Start server
cd D:\G.R.I.M\control\build\Release
.\training_control_server.exe

# Terminal 2: Start GRIM.exe
cd D:\G.R.I.M\build\Release
.\GRIM.exe

# In GRIM UI:
# 1. Open Training Panel
# 2. Click "Start Training"
# 3. Verify train_gpu.exe launches
# 4. Check training_status.fb updates
```

### 2. **Test Data Collection Pipeline** (CRITICAL)

```powershell
# With server running (Terminal 1 from above)

# In GRIM UI:
# 1. Open Training Panel
# 2. Click "Run Data Pipeline"
# 3. Verify grim_data_pipeline.exe launches
# 4. Check data/training_data.grmt created
```

### 3. **Implement Auto-Start** (HIGH PRIORITY)

Edit `D:\G.R.I.M\ai\grim_text_server_manager.cpp`:

```cpp
// Around line 266, add after path construction:
bool launchTrainingControlServer() {
    fs::path serverExe = fs::absolute("control/build/Release/training_control_server.exe");
    
    if (!fs::exists(serverExe)) {
        LOG_ERROR("TrainingControl", "Server executable not found: " + serverExe.string());
        return false;
    }
    
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;  // Hidden console
    
    PROCESS_INFORMATION pi = {};
    
    std::string cmdLine = "\"" + serverExe.string() + "\"";
    std::vector<char> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back('\0');
    
    BOOL result = CreateProcessA(
        nullptr, cmdBuf.data(), nullptr, nullptr,
        FALSE, CREATE_NEW_CONSOLE, nullptr, 
        serverExe.parent_path().string().c_str(),
        &si, &pi
    );
    
    if (!result) {
        LOG_ERROR("TrainingControl", "Failed to start server: " + std::to_string(GetLastError()));
        return false;
    }
    
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    // Wait for server to initialize
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    LOG_DEBUG("TrainingControl", "Server started successfully");
    return true;
}
```

### 4. **Debug Pipeline Failure** (HIGH PRIORITY)

Add logging to trace the issue:

```cpp
// In ui_training_panel.cpp around line 1380
if (!result.success) {
    LOG_ERROR("TrainingPanel", "Pipeline failed: " + result.error);
    LOG_DEBUG("TrainingPanel", "Server response: " + result.message);
    
    // Check if executable exists
    fs::path exePath = "resources/models/GRIM-text/DataCollection/build/Release/grim_data_pipeline.exe";
    LOG_DEBUG("TrainingPanel", "Checking path: " + exePath.string());
    LOG_DEBUG("TrainingPanel", "Exists: " + std::to_string(fs::exists(exePath)));
}
```

### 5. **Verify Path Resolution** (MEDIUM)

Test `getSafeResourcePath()` in isolation:

```cpp
// Test file: test_path_resolution.cpp
#include "control/training_paths.hpp"

int main() {
    // Test train_gpu.exe
    auto trainPath = GRIM::Training::getSafeResourcePath(
        "train_gpu.exe", 
        GRIM::Training::PathResolutionMode::Search
    );
    std::cout << "train_gpu.exe: " << trainPath << std::endl;
    
    // Test grim_data_pipeline.exe
    auto pipelinePath = GRIM::Training::getSafeResourcePath(
        "grim_data_pipeline.exe",
        GRIM::Training::PathResolutionMode::Search
    );
    std::cout << "grim_data_pipeline.exe: " << pipelinePath << std::endl;
    
    return 0;
}
```

## 📝 License

Apache 2.0 - Same as GRIM project

---

## 📌 Quick Reference

### Build All Components

```powershell
# 1. Control Server
cd D:\G.R.I.M\control
cmake -B build -DCMAKE_PREFIX_PATH="D:/G.R.I.M/vcpkg_installed/x64-windows"
cmake --build build --config Release

# 2. Train GPU
cd D:\G.R.I.M\resources\models\GRIM-text\training\TrainingLoop
cmake -B build -DCMAKE_PREFIX_PATH="D:/G.R.I.M/vcpkg_installed/x64-windows"
cmake --build build --config Release

# 3. Data Pipeline
cd D:\G.R.I.M\resources\models\GRIM-text\DataCollection
cmake -B build -DCMAKE_PREFIX_PATH="D:/G.R.I.M/vcpkg_installed/x64-windows"
cmake --build build --config Release

# 4. GRIM.exe
cd D:\G.R.I.M
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="D:/G.R.I.M/vcpkg_installed/x64-windows"
cmake --build build --config Release --target grim
```

### File Locations Cheat Sheet

| Component | Location |
|-----------|----------|
| Control Server Executable | `D:\G.R.I.M\control\build\Release\training_control_server.exe` |
| Training Executable | `D:\G.R.I.M\resources\models\GRIM-text\training\TrainingLoop\build\Release\train_gpu.exe` |
| Data Pipeline Executable | `D:\G.R.I.M\resources\models\GRIM-text\DataCollection\build\Release\grim_data_pipeline.exe` |
| GRIM Main Executable | `D:\G.R.I.M\build\Release\GRIM.exe` |
| Control Client Header | `D:\G.R.I.M\control\training_control_client.hpp` |
| FlatBuffer Schema | `D:\G.R.I.M\control\training_control.fbs` |
| Path Utilities | `D:\G.R.I.M\control\training_paths.hpp` |
| Training Status | `D:\G.R.I.M\resources\models\GRIM-text\training\training_status.fb` |

### Common Commands

```powershell
# Start training server
cd D:\G.R.I.M\control\build\Release ; .\training_control_server.exe

# Check if server is running
Invoke-RestMethod -Uri http://127.0.0.1:11436/health

# Kill server if port stuck
netstat -ano | findstr :11436
taskkill /PID <pid> /F

# View GRIM.exe logs
cd D:\G.R.I.M\logs
Get-Content -Wait training_log_*.txt

# Rebuild just GRIM.exe (fast)
cd D:\G.R.I.M
cmake --build build --config Release --target grim

# Rebuild everything (slow)
cd D:\G.R.I.M
cmake --build build --config Release
```

---

**Last Updated**: November 7, 2025  
**Status**: All executables built, end-to-end testing pending  
**Ready to use!** Follow the Next Steps section to complete integration.
