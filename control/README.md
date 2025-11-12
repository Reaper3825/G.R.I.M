# GRIM-text Training Control System

Decoupled training control for GRIM-text via HTTP + FlatBuffers.

## Architecture

```
┌─────────────────┐                    ┌──────────────────────────┐
│   GRIM.exe      │                    │  GRIM-text Training      │
│                 │                    │                          │
│  ┌───────────┐  │   HTTP + FB       │  ┌───────────────────┐   │
│  │ Training  │──┼───────────────────┼─→│ Control Server    │   │
│  │ UI Panel  │  │  (localhost:11436) │  │ (training_control │   │
│  └───────────┘  │                    │  │  _server.exe)     │   │
│                 │                    │  └─────────┬─────────┘   │
│  Uses:          │                    │            │             │
│  - training_    │                    │            ↓             │
│    control_     │                    │  ┌─────────────────────┐ │
│    client.hpp   │                    │  │ train_gpu.exe       │ │
│                 │                    │  │ (CUDA training)     │ │
└─────────────────┘                    │  └─────────────────────┘ │
                                       │                          │
                                       │  Status: training_status.fb
                                       └──────────────────────────┘
```

## Components

### 1. Control Server (`training_control_server.cpp`)
- HTTP server on port 11436
- Manages training process lifecycle
- Monitors `training_status.fb` for real-time updates
- Communicates via FlatBuffers

**Endpoints:**
- `GET /health` - Health check
- `GET /api/status` - Get current training status
- `POST /api/training/start` - Start training
- `POST /api/training/stop` - Stop training
- `POST /api/config` - Update configuration
- `POST /api/server/shutdown` - Shutdown server gracefully

### 2. Control Client (`training_control_client.hpp`)
- Header-only client library
- Include in GRIM.exe (no linking required!)
- Simple C++ API wrapping HTTP + FlatBuffers

**Usage:**
```cpp
#include "training_control_client.hpp"

GRIMText::TrainingControlClient client("127.0.0.1", 11436);

// Check if server is running
if (!client.isServerRunning()) {
    // Start server first...
}

// Get status
GRIMText::TrainingState state;
GRIMText::TrainingStats stats;
GRIMText::TrainingConfig config;

if (client.getStatus(state, stats, config)) {
    std::cout << "Epoch: " << stats.currentEpoch << "/" << stats.totalEpochs << std::endl;
    std::cout << "Loss: " << stats.currentLoss << std::endl;
    std::cout << "Progress: " << (stats.trainingProgress * 100.0f) << "%" << std::endl;
}

// Start training
GRIMText::TrainingConfig customConfig;
customConfig.epochs = 5;
customConfig.batchSize = 16;
customConfig.learningRate = 0.0002f;

if (client.startTraining(&customConfig)) {
    std::cout << "Training started!" << std::endl;
}

// Stop training
if (client.stopTraining()) {
    std::cout << "Training stopped" << std::endl;
}

// Shutdown server
if (client.shutdownServer()) {
    std::cout << "Server shutdown initiated" << std::endl;
}
```

### 3. FlatBuffer Schema (`training_control.fbs`)
- Schema for all communication
- Zero-copy deserialization
- Type-safe, versioned protocol

**Compile schema:**
```bash
flatc --cpp training_control.fbs
# Generates: training_control_generated.h
```

## Building

### Prerequisites
- CMake 3.22+
- FlatBuffers
- cpp-httplib
- vcpkg (recommended)

### Compile FlatBuffer Schema
```bash
cd D:\G.R.I.M\resources\models\GRIM-text\training\control
flatc --cpp training_control.fbs
```

### Build Control Server
```bash
cd D:\G.R.I.M\resources\models\GRIM-text\training
cmake --preset vs-cuda-release
cmake --build build_vs_cuda --config Release --target training_control_server
```

Output: `build_vs_cuda\Release\training_control_server.exe`

## Running

### Start Control Server
```powershell
cd D:\G.R.I.M\resources\models\GRIM-text\training
.\build_vs_cuda\Release\training_control_server.exe
```

Server starts on `http://127.0.0.1:11436`

### Test with curl
```bash
# Get status
curl -X GET http://127.0.0.1:11436/api/status --output status.fb
xxd status.fb

# Health check
curl http://127.0.0.1:11436/health
```

## Integration with GRIM.exe

### Include in UI Panel
```cpp
// In your ui_training_panel.cpp
#include "resources/models/GRIM-text/training/control/training_control_client.hpp"

class UITrainingPanel : public UIPanel {
private:
    GRIMText::TrainingControlClient trainingClient{"127.0.0.1", 11436};
    
    void update(const InputState& input, float dt) override {
        // Poll status every second
        if (pollTimer > 1.0f) {
            GRIMText::TrainingState state;
            GRIMText::TrainingStats stats;
            GRIMText::TrainingConfig config;
            
            if (trainingClient.getStatus(state, stats, config)) {
                // Update UI with stats
                updateProgressBar(stats.trainingProgress);
                updateLossDisplay(stats.currentLoss);
                // etc...
            }
            
            pollTimer = 0.0f;
        }
        pollTimer += dt;
    }
    
    void onStartTrainingButton() {
        if (trainingClient.startTraining()) {
            LOG_DEBUG("TrainingPanel", "Training started");
        } else {
            LOG_ERROR("TrainingPanel", trainingClient.getLastError());
        }
    }
};
```

### Add to CMakeLists.txt
```cmake
# In D:\G.R.I.M\CMakeLists.txt (main GRIM.exe build)

# Just include the header directory - no linking needed!
target_include_directories(grim PRIVATE
    ${CMAKE_SOURCE_DIR}/resources/models/GRIM-text/training/control
)

# FlatBuffers and httplib (already in vcpkg)
find_package(flatbuffers CONFIG REQUIRED)
find_path(HTTPLIB_INCLUDE_DIR NAMES httplib.h)

target_include_directories(grim PRIVATE
    ${HTTPLIB_INCLUDE_DIR}
)

target_link_libraries(grim PRIVATE
    flatbuffers::flatbuffers
)
```

**That's it!** No CUDA linking, no .cu files, just pure HTTP communication.

## Status File Format

The training process writes `training_status.fb` (FlatBuffer format):

```cpp
// Read status file directly
std::ifstream file("training_status.fb", std::ios::binary);
std::vector<uint8_t> buffer(...);
file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

auto status = GRIMText::Control::GetStatusResponse(buffer.data());
std::cout << "Epoch: " << status->stats()->current_epoch() << std::endl;
```

## Protocol Details

### Message Format
All HTTP requests/responses use FlatBuffer binary format:
- Content-Type: `application/octet-stream`
- Body: Raw FlatBuffer bytes
- Zero parsing overhead
- Type-safe schema validation

### State Machine
```
Idle → Collecting → Verifying → Training → Completed
  ↑                                ↓
  └────────────────────────────────┘
              (Stop)
```

### Error Handling
- Server returns error messages in FlatBuffer responses
- Client provides `getLastError()` method
- All operations return `bool` success/failure

## Advantages

✅ **No Linking** - GRIM.exe doesn't link to CUDA code  
✅ **Decoupled** - Training runs in separate process  
✅ **Efficient** - FlatBuffers for zero-copy communication  
✅ **Type-Safe** - Schema-validated messages  
✅ **Debuggable** - HTTP requests visible in network tools  
✅ **Testable** - Can test with curl or other HTTP clients  
✅ **Scalable** - Could run training on remote machine  

## Troubleshooting

**Server won't start:**
```bash
# Check if port is in use
netstat -ano | findstr :11436

# Kill existing process
taskkill /PID <pid> /F
```

**Can't connect from GRIM.exe:**
```cpp
if (!client.isServerRunning()) {
    // Start server programmatically
    system("start training_control_server.exe");
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
```

**FlatBuffer parse errors:**
```bash
# Recompile schema
flatc --cpp training_control.fbs

# Rebuild server
cmake --build build_vs_cuda --config Release --target training_control_server
```

## License

Apache 2.0 - Same as GRIM project
