# GRIM-text Training Control Integration - Complete Roadmap

**Last Updated:** November 6, 2025  
**Status:** Control Server Built & Tested ✅  
**Next Phase:** UI Integration into GRIM.exe

---

## 📊 Current Status

### ✅ Completed Components

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| FlatBuffer Schema | ✅ Complete | `training_control.fbs` | Defines all message types |
| Generated Headers | ✅ Complete | `training_control_generated.h` | Auto-generated from schema |
| Control Server | ✅ Built & Tested | `training_control_server.exe` | HTTP server on port 11436 |
| Client Library | ✅ Ready | `training_control_client.hpp` | Header-only, no linking needed |
| Health Endpoint | ✅ Working | `GET /health` | Returns 200 OK with FlatBuffer |
| Status Endpoint | ✅ Working | `GET /api/status` | Returns training state & stats |
| Start Endpoint | ✅ Ready | `POST /api/training/start` | Accepts FlatBuffer config |
| Stop Endpoint | ✅ Ready | `POST /api/training/stop` | Stops training gracefully |
| Config Endpoint | ✅ Ready | `POST /api/config` | Updates training config |

### 🔄 In Progress

- None currently

### ⏳ Not Started

- UI Panel implementation in GRIM.exe
- Process lifecycle management (spawn/monitor `train_gpu.exe`)
- Real-time status monitoring
- Training data pipeline UI

---

## 🎯 Phase 1: UI Panel Foundation (Priority: HIGH)

### 1.1 Create UI Training Panel Files

**Files to Create:**
```
d:\G.R.I.M\ui\ui_training_panel.hpp
d:\G.R.I.M\ui\ui_training_panel.cpp
```

**Requirements:**
- Inherit from `UIPanel` base class
- Include `training_control_client.hpp`
- Include `training_control_generated.h`
- Follow patterns from `ui_settings_menu.cpp` and `console_panel.cpp`

**Key Features:**
- Connection status indicator
- Training state display (Idle/Training/Paused/Completed/Error)
- Configuration sliders (epochs, batch size, learning rate, etc.)
- Start/Stop/Pause buttons
- Progress bars (overall, epoch, batch)
- Real-time statistics (loss, perplexity, tokens/sec, GPU usage)
- ETA display
- Log viewer (scrolling console)

**Estimated Time:** 4-6 hours

---

### 1.2 Add Include Paths to GRIM.exe CMakeLists.txt

**File to Modify:**
```
d:\G.R.I.M\CMakeLists.txt
```

**Changes:**
```cmake
# Add FlatBuffers include
target_include_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/include
    ${CMAKE_SOURCE_DIR}/resources/models/GRIM-text/training/control
)

# Link cpp-httplib (header-only, already available via vcpkg)
```

**Estimated Time:** 15 minutes

---

### 1.3 Register Panel with UI System

**Files to Modify:**
```
d:\G.R.I.M\ui\ui_root.hpp
d:\G.R.I.M\ui\ui_root.cpp
```

**Changes:**
- Add `#include "ui_training_panel.hpp"` to ui_root.hpp
- Add `std::unique_ptr<UITrainingPanel> trainingPanel;` member variable
- Create panel in constructor: `trainingPanel = std::make_unique<UITrainingPanel>();`
- Add to panel list for rendering/updating
- Add keybind (e.g., F4) to toggle panel visibility
- Add menu command (like Settings command)

**Estimated Time:** 30 minutes

---

### 1.4 Create Widget Layout

**UI Layout Structure:**

```
┌─────────────────────────────────────────────────────────┐
│ GRIM-text Training Control                    [X]       │
├─────────────────────────────────────────────────────────┤
│ Server: ● Connected (127.0.0.1:11436)                   │
│ State:  ⚡ Training                                      │
├─────────────────────────────────────────────────────────┤
│ Configuration                                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Epochs:          [====|=====] 5                      │ │
│ │ Batch Size:      [===|======] 16                     │ │
│ │ Learning Rate:   [==|=======] 0.0002                 │ │
│ │ Max Seq Length:  [=====|====] 4096                   │ │
│ │ Warmup Steps:    [===|======] 500                    │ │
│ │ ☑ Use GPU        ☑ Flash Attention                  │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                           │
│ Controls                                                  │
│ [▶ Start Training]  [⏸ Pause]  [⏹ Stop Training]        │
│                                                           │
│ Progress                                                  │
│ Overall:  [████████████████░░░░░░░░] 67% - Epoch 4/5    │
│ Epoch:    [██████████████████████░░] 89% - Batch 89/100 │
│ Batch:    [████████████████████████] 100%               │
│                                                           │
│ Statistics                                                │
│ Loss: 2.347        Perplexity: 10.45                    │
│ Tokens/sec: 2,456  GPU Memory: 8,192 / 12,288 MB        │
│ ETA: 1h 23m        Elapsed: 2h 14m                      │
│                                                           │
│ Training Logs                              [Clear Logs]  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ [12:34:56] Starting training...                      │ │
│ │ [12:34:57] Loaded 1,234,567 training samples        │ │
│ │ [12:35:00] Epoch 1/5, Batch 1/100, Loss: 4.234      │ │
│ │ [12:35:02] Epoch 1/5, Batch 2/100, Loss: 4.123      │ │
│ │ ... (scrollable)                                     │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Estimated Time:** 2-3 hours

---

## 🎯 Phase 2: Server Integration (Priority: HIGH)

### 2.1 Implement Polling System

**Location:** `ui_training_panel.cpp` - `update()` method

**Implementation:**
```cpp
void UITrainingPanel::update(const InputState& input, float dt) {
    pollTimer += dt;
    
    if (pollTimer >= pollInterval) {
        pollTimer = 0.0f;
        
        // Check server connection
        if (!client.isServerRunning()) {
            serverConnected = false;
            // Show "Server Offline" indicator
            return;
        }
        
        serverConnected = true;
        
        // Get current status
        if (client.getStatus(currentState, currentStats, currentConfig)) {
            updateStatusDisplay();
            updateProgressBars();
        }
    }
    
    // Update UI widgets
    UIPanel::update(input, dt);
}
```

**Estimated Time:** 1 hour

---

### 2.2 Implement Control Button Handlers

**Start Training Handler:**
```cpp
void UITrainingPanel::handleStartTraining() {
    if (!serverConnected) {
        showError("Server not connected");
        return;
    }
    
    // Get config from UI sliders
    GRIMText::TrainingConfig config;
    config.epochs = static_cast<int>(epochsSlider->getValue());
    config.batchSize = static_cast<int>(batchSizeSlider->getValue());
    config.learningRate = learningRateSlider->getValue();
    config.maxSeqLen = static_cast<int>(maxSeqLenSlider->getValue());
    config.warmupSteps = static_cast<int>(warmupStepsSlider->getValue());
    config.useGPU = useGPUToggle->isEnabled();
    config.useFlashAttention = useFlashAttentionToggle->isEnabled();
    config.dataPath = "data/training_data.grmt";
    config.vocabPath = "models/vocab.bin";
    config.outputPath = "models/grim_text_trained.bin";
    
    if (client.startTraining(&config)) {
        LOG_DEBUG("Training", "Training started successfully");
    } else {
        showError("Failed to start: " + client.getLastError());
    }
}
```

**Stop Training Handler:**
```cpp
void UITrainingPanel::handleStopTraining() {
    if (!serverConnected) return;
    
    if (client.stopTraining()) {
        LOG_DEBUG("Training", "Training stopped");
    } else {
        showError("Failed to stop: " + client.getLastError());
    }
}
```

**Estimated Time:** 1 hour

---

### 2.3 Auto-Start Server Feature

**Implementation:**
```cpp
void UITrainingPanel::checkServerConnection() {
    if (!client.isServerRunning()) {
        // Try to start server
        std::string serverPath = "resources/models/GRIM-text/training/build_vs_cuda/control/Release/training_control_server.exe";
        
        if (std::filesystem::exists(serverPath)) {
            #ifdef _WIN32
            STARTUPINFO si = {sizeof(si)};
            PROCESS_INFORMATION pi;
            
            if (CreateProcess(
                serverPath.c_str(),
                nullptr, nullptr, nullptr, FALSE,
                CREATE_NEW_CONSOLE, nullptr, nullptr,
                &si, &pi
            )) {
                CloseHandle(pi.hProcess);
                CloseHandle(pi.hThread);
                
                // Wait a moment for server to start
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            #endif
        }
    }
}
```

**Estimated Time:** 1 hour

---

## 🎯 Phase 3: Enhanced Features (Priority: MEDIUM)

### 3.1 Real-time Loss Graph

**Feature:** Plot loss values over time

**Widget:** Custom `UIGraph` or use existing charting library

**Data Structure:**
```cpp
struct LossDataPoint {
    float timestamp;
    float loss;
};

std::vector<LossDataPoint> lossHistory;
```

**Estimated Time:** 3-4 hours

---

### 3.2 Log Viewer with Filtering

**Features:**
- Scrollable log window
- Auto-scroll to bottom
- Filter by log level (INFO/DEBUG/WARNING/ERROR)
- Clear logs button
- Export logs to file

**Estimated Time:** 2 hours

---

### 3.3 Configuration Presets

**Features:**
- Save current config as preset
- Load preset configurations
- Delete presets
- Default presets (Fast Training, Quality Training, Test Run)

**Storage:** JSON files in `data/training_presets/`

**Estimated Time:** 2-3 hours

---

### 3.4 Checkpoint Management

**Features:**
- List available checkpoints
- Load checkpoint to resume training
- Delete old checkpoints
- View checkpoint metadata (epoch, loss, timestamp)

**Estimated Time:** 2-3 hours

---

## 🎯 Phase 4: Training Process Integration (Priority: HIGH)

### 4.1 Process Spawning in Control Server

**File to Modify:**
```
d:\G.R.I.M\resources\models\GRIM-text\training\control\training_control_server.cpp
```

**Current Status:** Server has `TrainingProcessController` class with placeholder implementation

**Required Changes:**
```cpp
bool TrainingProcessController::start(const TrainingConfig& config) {
    std::string command = "train_gpu.exe";
    command += " --data " + config.dataPath;
    command += " --vocab " + config.vocabPath;
    command += " --output " + config.outputPath;
    command += " --epochs " + std::to_string(config.epochs);
    command += " --batch-size " + std::to_string(config.batchSize);
    command += " --lr " + std::to_string(config.learningRate);
    
    #ifdef _WIN32
    STARTUPINFO si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    
    if (CreateProcess(nullptr, (LPSTR)command.c_str(), 
        nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        processHandle = pi.hProcess;
        processId = pi.dwProcessId;
        CloseHandle(pi.hThread);
        return true;
    }
    #endif
    
    return false;
}
```

**Estimated Time:** 2-3 hours

---

### 4.2 Status File Monitoring

**Current Implementation:** `StatusFileMonitor` polls `training_status.fb` file

**Required:** `train_gpu.exe` must write status to this file

**File to Modify:**
```
d:\G.R.I.M\resources\models\GRIM-text\training\train_gpu.cu
```

**Add Status Writing:**
```cpp
void writeTrainingStatus(const TrainingStats& stats) {
    flatbuffers::FlatBufferBuilder builder(1024);
    
    auto fbStats = CreateTrainingStats(builder,
        stats.currentEpoch,
        stats.totalEpochs,
        stats.currentBatch,
        // ... all stats fields
    );
    
    auto response = CreateStatusResponse(builder,
        TrainingState_Training,
        fbStats,
        0, 0
    );
    
    builder.Finish(response);
    
    std::ofstream file("training_status.fb", std::ios::binary);
    file.write((char*)builder.GetBufferPointer(), builder.GetSize());
}

// Call in training loop
if (batch % 10 == 0) {
    writeTrainingStatus(currentStats);
}
```

**Estimated Time:** 2-3 hours

---

## 🎯 Phase 5: Testing & Polish (Priority: MEDIUM)

### 5.1 Error Handling

**Test Cases:**
- Server not running
- Server crashes during training
- Invalid configuration values
- Out of GPU memory
- Training data not found
- Vocab file not found
- Process spawn failures

**Estimated Time:** 2-3 hours

---

### 5.2 UI Polish

**Features:**
- Smooth animations for progress bars
- Color coding (green=running, red=error, yellow=paused)
- Icons for buttons
- Tooltips for sliders
- Responsive layout
- Dark/light theme support

**Estimated Time:** 3-4 hours

---

### 5.3 Performance Optimization

**Optimizations:**
- Reduce polling frequency when idle
- Batch FlatBuffer parsing
- Efficient log entry storage (circular buffer)
- Debounce slider changes

**Estimated Time:** 2 hours

---

### 5.4 Documentation

**Documents to Create:**
- User guide for training panel
- Troubleshooting guide
- Architecture diagram
- API documentation for control protocol

**Estimated Time:** 2-3 hours

---

## 📅 Estimated Timeline

| Phase | Tasks | Time | Priority |
|-------|-------|------|----------|
| Phase 1 | UI Panel Foundation | 8-10 hours | HIGH |
| Phase 2 | Server Integration | 3-4 hours | HIGH |
| Phase 3 | Enhanced Features | 9-12 hours | MEDIUM |
| Phase 4 | Process Integration | 4-6 hours | HIGH |
| Phase 5 | Testing & Polish | 9-12 hours | MEDIUM |
| **TOTAL** | | **33-44 hours** | |

---

## 🚀 Quick Start Implementation Order

1. **Create `ui_training_panel.hpp` and `.cpp`** (4-6 hours)
2. **Add to CMakeLists.txt and ui_root** (45 minutes)
3. **Implement basic polling** (1 hour)
4. **Add Start/Stop button handlers** (1 hour)
5. **Test with running control server** (30 minutes)
6. **Implement process spawning in control server** (2-3 hours)
7. **Add status file writing to train_gpu.cu** (2-3 hours)
8. **Polish UI and add error handling** (3-4 hours)

**Minimum Viable Product:** ~12-16 hours

---

## 📝 Notes

- Control server is fully functional and tested
- FlatBuffer communication protocol is working
- Header-only client library requires no additional linking
- All CUDA code stays in GRIM-text (no linking issues)
- HTTP communication is localhost-only (no network security concerns)
- Server can be auto-started by UI if not running
- Real-time updates via polling (1-2 second intervals)

---

## 🔗 Key Files Reference

| File | Purpose |
|------|---------|
| `training_control_server.cpp` | HTTP control server |
| `training_control_client.hpp` | Client library for GRIM.exe |
| `training_control.fbs` | FlatBuffer schema |
| `training_control_generated.h` | Generated FlatBuffer C++ code |
| `example_usage.cpp` | Integration examples |
| `train_gpu.cu` | Actual GPU training code |
| `grim_text_server.cpp` | Reference HTTP server implementation |

---

**END OF ROADMAP**
