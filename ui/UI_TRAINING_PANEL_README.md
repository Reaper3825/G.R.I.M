# GRIM-text Training Control Panel

## Overview

The Training Control Panel provides a comprehensive interface for managing the GRIM-text model training pipeline directly from GRIM.exe. It integrates server management, configuration, data collection, verification, and real-time training monitoring in a single unified interface.

---

## Panel Layout

The panel is divided into two main columns:

### Left Column (35%) - Configuration & Data Management
- **Training Configuration** - Adjustable hyperparameters
- **Data Source Management** - Add custom data sources
- **Data Verification** - Quality control for training data
- **Control Buttons** - Start/Stop/Pause/Close

### Right Column (65%) - Monitoring & Logs
- **Server Status** - Connection state and training state
- **Hardware Information** - GPU/CPU/RAM details
- **Training Progress** - Real-time progress bar and statistics
- **Verbose Output** - Reserved for detailed metrics
- **Training Logs** - Real-time log messages with timestamps

---

## Configuration Section

### Training Hyperparameters

All sliders feature **dynamic precision** based on their value range:

#### **Epochs** (Range: 1 - 100)
- **Purpose**: Number of complete passes through the training dataset
- **Default**: 10
- **Precision**: 2 decimal places
- **Impact**: More epochs = longer training, better convergence (risk of overfitting)
- **Time Effect**: Linear - doubles epochs = doubles training time

#### **Batch Size** (Range: 1 - 64)
- **Purpose**: Number of samples processed simultaneously
- **Default**: 8
- **Precision**: 2 decimal places
- **GPU Impact**: 
  - Larger batches = better GPU utilization
  - Too large = out of memory errors
- **Time Effect**: 
  - Larger batches = fewer batches per epoch = faster
  - Batch size 16+ gets 10% efficiency gain
  - Batch size <4 gets 20% slowdown

#### **Learning Rate** (Range: 0.00001 - 0.01)
- **Purpose**: Step size for gradient descent optimization
- **Default**: 0.0001
- **Precision**: 6 decimal places (e.g., 0.000100)
- **Guidelines**:
  - Too high = unstable training, divergence
  - Too low = very slow convergence
  - Typical range: 0.0001 - 0.001
- **Time Effect**: Does not affect training time

#### **Max Seq Length** (Range: 512 - 16384)
- **Purpose**: Maximum token sequence length for context window
- **Default**: 512
- **Precision**: 0 decimal places (integers)
- **Memory Impact**: 
  - Longer sequences = quadratic memory cost
  - 1024 tokens uses 4x memory of 512 tokens
- **Time Effect**: Linear - sequence length multiplier applied to time per batch

#### **Warmup Steps** (Range: 0 - 5000)
- **Purpose**: Number of initial batches with gradually increasing learning rate
- **Default**: 1000
- **Precision**: 0 decimal places (integers)
- **Benefits**: 
  - Prevents early training instability
  - Smooths initial convergence
- **Time Effect**: Adds `warmupSteps × timePerBatch` to total time

### Save Config Button
- **Function**: Saves current configuration to `ai_config.json`
- **Color**: Blue border (0xFF00AAFF)
- **Persistence**: Configuration survives GRIM restarts
- **Auto-save**: Also triggered when starting training

---

## Data Source Management

### Add Data Source Section

#### **URL Input Box**
- **Placeholder**: "Enter data source URL..."
- **Validation**: Must start with `http://` or `https://`
- **Format**: Plain text input
- **Color**: Cyan border (0xFF00AAFF)

#### **Add Source Button**
- **Function**: Adds URL to `source_data.json`
- **Color**: Green border (0xFF00FF00)
- **Effect**: 
  - Creates new source entry with default settings
  - Sets `enabled: true`, `priority: 5`, `source_type: custom`
  - Updates training time estimate based on new source count

### Source Data Structure
When you add a source, it's stored as:
```json
{
  "name": "Custom Source",
  "url": "https://example.com",
  "source_type": "custom",
  "enabled": true,
  "priority": 5,
  "requires_auth": false
}
```

---

## Data Verification

### Run Verification Button
- **Function**: Executes `verifier.exe` to validate collected data
- **Color**: Purple border (0xFFAA00FF)
- **Process Management**:
  - Spawns verifier.exe using Windows CreateProcess API
  - Runs hidden (no console window)
  - 60-second timeout
  - Tracked by Process ID (logged)
  - Proper handle cleanup

### Verification Process
1. **Input**: Reads unverified entries from `data/collected/`
2. **Checks**:
   - Domain whitelist approval (40% weight)
   - Content quality validation (40% weight)
     - Min length: 100 characters
     - Max length: 50,000 characters
   - Source type reliability (20% weight)
   - Duplicate detection (Jaccard similarity)
3. **Output**: Verified entries saved to `data/verified/`
4. **Stats**: Written to `verification_stats.json`

### Verification Stats Display
Shows after verification completes:
```
Processed: 150
Passed: 90
Failed: 60
```
- **Processed**: Total entries checked
- **Passed**: Entries meeting reliability threshold (0.8)
- **Failed**: Rejected entries (domain/quality/duplicate)

---

## Training Control Buttons

All control buttons are stacked vertically at the bottom left of the panel using a VBox layout.

### Start Training Button
- **Label**: "▶ Start"
- **Color**: Green when enabled (0xFF00FF00), Gray when disabled
- **Enabled When**: Server connected AND state is Idle
- **Function**:
  1. Auto-starts GRIM-text server if not running (500ms delay for initialization)
  2. Validates server connection
  3. Sends training start command via FlatBuffer protocol
  4. Updates configuration from sliders
  5. Begins training session
- **Hotkey**: None (use mouse click)

### Stop Training Button
- **Label**: "⏹ Stop"
- **Color**: Red when enabled (0xFFFF0000), Gray when disabled
- **Enabled When**: Server connected AND (Training OR Paused)
- **Function**:
  1. Validates current state
  2. Sends stop command to server
  3. Resets progress bar to 0%
  4. Transitions state to Idle
- **Safety**: Prevents accidental stops when not training

### Pause/Resume Button
- **Label**: "⏸ Pause" or "▶ Resume"
- **Color**: Orange when enabled (0xFFFFAA00), Gray when disabled
- **Enabled When**: Server connected AND (Training OR Paused)
- **Function**:
  - **When Training**: Sends pause command, label changes to "▶ Resume"
  - **When Paused**: Sends resume command, label changes to "⏸ Pause"
- **State Aware**: Button text updates automatically

### Close Panel Button
- **Label**: "✖ Close"
- **Color**: Blue border (0xFF00AAFF)
- **Always Enabled**: Can always close the panel
- **Function**: Hides the panel (`setVisible(false)`)
- **Server Behavior**: Does NOT stop the training server
- **Use Case**: Close UI while training continues in background

---

## Right Panel - Monitoring

### Server Status (Top)
**Connection Status**:
- 🟢 **"Server Online"** (Green): Connected to training server on port 11436
- 🔴 **"Server Offline"** (Red): No connection, server not running

**Disconnected State**:
- Red text: ">>> Disconnected"
- Indicates FlatBuffer client cannot reach server

**Training State**:
- **Idle**: Ready to start training
- **Training**: Active training in progress
- **Paused**: Training paused, can resume
- **Completed**: Training finished successfully
- **Error**: Training encountered an error

### Hardware Information

Displays real-time system capabilities from `g_systemInfo`:

**GPU Information**:
```
GPU: NVIDIA GeForce RTX 3080 Ti (12287 MB VRAM)
CUDA: Available
```
- Shows GPU model name and VRAM capacity
- CUDA status indicates GPU acceleration availability

**CPU Information**:
```
CPU: 20 cores
```
- Total logical processor count

**RAM Information**:
```
RAM: 130861 MB
```
- Total system memory in megabytes (~127 GB)

### Estimated Training Time

**Dynamic Calculation** based on:
1. **Hardware**:
   - CUDA available: Base time 0.15s per batch
   - CUDA + High VRAM (>10GB): 20% faster
   - CPU only: 8x slower (1.2s per batch)
   - CPU cores: 16+ cores = 40% faster on CPU

2. **Configuration**:
   - Sequence length multiplier: `maxSeqLength / 512`
   - Batch size efficiency:
     - 16+ batches: 10% faster
     - <4 batches: 20% slower
   - Warmup overhead: `warmupSteps × timePerBatch`

3. **Dataset Size**:
   - Reads `source_data.json`
   - Counts enabled sources
   - Sums `fetch_limit` values
   - Applies 60% verification pass rate
   - Example: 10 sources × 500 limit = 5000 raw → 3000 verified samples

**Formula**:
```
totalBatches = (datasetSize / batchSize) × epochs
trainingTime = totalBatches × timePerBatch + warmupSteps × timePerBatch
```

**Display Format**:
- Hours: "2h 15m 30s"
- Minutes: "45m 12s"
- Seconds: "28s"

### Training Progress Bar

**Visual Indicator**:
- Width: Full right panel width - 20px
- Height: 30px
- Color: Cyan (0xFF00FFFF)
- Background: Dark gray (0xFF202020)

**Progress Calculation**:
```
epochProgress = (currentEpoch - 1) / totalEpochs
batchProgress = currentBatch / totalBatches / totalEpochs
totalProgress = epochProgress + batchProgress
```

**Real-time Updates**:
- Polls server every 1.5 seconds
- Updates based on `TrainingStats` from server
- Shows percentage: "0.0%" to "100.0%"

### Training Statistics (When Connected)

**Epoch Progress**:
```
Epoch: 3/10
```
- Current epoch / Total epochs

**Batch Progress**:
```
Batch: 45/62
```
- Current batch within epoch / Total batches per epoch

**Loss**:
```
Loss: 2.3456
```
- Current training loss (lower is better)
- 4 decimal places precision

**Perplexity**:
```
Perplexity: 12.34
```
- Model confusion metric (lower is better)
- 2 decimal places precision
- Exponential of loss: `exp(loss)`

### Verbose Output Area

**Reserved Section** for future features:
- GPU utilization graphs
- Memory usage tracking
- Token throughput metrics (tokens/second)
- Gradient statistics
- Learning rate schedule visualization

Currently displays placeholder text in gray.

### Training Logs

**Scrollable Log Area**:
- Auto-scroll: Automatically scrolls to newest messages
- Max entries: 1000 logs (oldest removed)
- Timestamp format: `[HH:MM:SS]`

**Log Levels**:
- **Level 0 (Cyan)**: Info messages
  - Configuration loaded
  - Server started
  - Training progress milestones
- **Level 1 (Yellow)**: Warnings
  - Validation issues
  - Non-critical errors
- **Level 2 (Red)**: Errors
  - Connection failures
  - Training errors
  - File I/O errors

**Example Log Output**:
```
[19:17:12] Configuration loaded from ai_config.json
[19:17:12] Training panel initialized
[19:18:56] Configuration saved to ai_config.json
[19:19:02] Starting GRIM-text server...
[19:19:03] Server connection established
```

---

## File Integration

### Configuration Files

#### `ai_config.json` (Root directory)
```json
{
  "training": {
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "max_seq_len": 512,
    "warmup_steps": 1000,
    "optimizer": "adam",
    "gradient_clip": 1.0
  }
}
```
- Loaded on panel initialization
- Saved when "Save Config" clicked or training starts
- Persists between GRIM sessions

#### `source_data.json` (resources/models/GRIM-text/training/)
```json
{
  "version": "1.0.0",
  "description": "GRIM Web Data Collection Configuration",
  "collection_settings": {
    "max_entries_per_source": 500,
    "timeout_seconds": 30,
    "rate_limit_delay_ms": 1000
  },
  "data_sources": [
    {
      "name": "Project Gutenberg",
      "url": "https://www.gutenberg.org/...",
      "source_type": "gutenberg",
      "enabled": true,
      "priority": 10,
      "fetch_limit": 1000
    }
  ]
}
```
- Read during time estimation
- Modified when adding custom sources
- Used by web collector for data gathering

#### `verification_stats.json` (resources/models/GRIM-text/training/data/)
```json
{
  "total_processed": 150,
  "passed_verification": 90,
  "failed_verification": 35,
  "domain_rejected": 10,
  "quality_rejected": 15,
  "duplicate_rejected": 10
}
```
- Written by verifier.exe
- Read by panel after verification completes
- Displayed in verification stats area

---

## Server Communication

### FlatBuffer Protocol

The panel communicates with the GRIM-text training server using FlatBuffers:

**Port**: 11436 (HTTP)

**Message Types**:
- `StartTraining` - Begin training session
- `StopTraining` - Halt training
- `PauseTraining` - Pause training
- `ResumeTraining` - Resume from pause
- `GetStatus` - Request current state and stats
- `UpdateConfig` - Send new configuration

**Polling**:
- Interval: 1.5 seconds
- Automatic when panel visible
- Gets `TrainingState`, `TrainingStats`, `TrainingConfig`

### Training Control Client

**Implementation**: Header-only C++ client
**Location**: `training/control/training_control_client.hpp`

**Key Methods**:
- `isServerRunning()` - Check connection
- `getStatus(state, stats, config)` - Retrieve all data
- `startTraining(config)` - Begin session
- `stopTraining()` - End session
- `pauseTraining()` / `resumeTraining()` - Pause control

---

## Global Access

### Command Integration

The training panel is accessible via the global pointer:

```cpp
extern std::shared_ptr<UITrainingPanel> g_trainingPanel;
```

**Defined in**: `main.cpp` (line ~45)

**Command Functions** (callable from commands system):
```cpp
void UITrainingPanel::startTrainingSession();
void UITrainingPanel::stopTrainingSession();
void UITrainingPanel::pauseTrainingSession();
void UITrainingPanel::resumeTrainingSession();
void UITrainingPanel::shutdownTrainingServer();
```

**Example Command Usage**:
```cpp
// From commands/commands_training.cpp
if (g_trainingPanel) {
    g_trainingPanel->startTrainingSession();
}
```

---

## Keyboard & Mouse Interaction

### Mouse Input

**Sliders**:
- Click and drag handle to adjust value
- Real-time value display updates
- Triggers `calculateTrainingEstimate()` for relevant parameters

**Buttons**:
- Click to activate
- Visual feedback (color changes on hover/click)
- State-aware enable/disable

**Input Box**:
- Click to focus
- Type URL directly
- Enter key not supported (use Add Source button)

**Panel Dragging**:
- Click and drag title bar to move panel
- Position persists during session

**Panel Resizing**:
- Drag bottom-right corner handle
- Minimum size: 300×200 pixels
- Content scrolls if overflows

### Scrolling

**Left Panel (Configuration)**:
- Scrollable when content exceeds height
- Scroll bar on right edge (cyan)
- Mouse wheel support (if implemented)

**Logs Area**:
- Auto-scroll to newest entries
- Manual scroll locks auto-scroll temporarily

---

## Performance Considerations

### Update Frequency
- **Server Polling**: Every 1.5 seconds (configurable)
- **UI Refresh**: Every frame (~60 FPS)
- **Log Limit**: 1000 entries to prevent memory bloat

### Resource Usage
- **Minimal CPU**: Polling only when visible
- **No GPU**: UI rendering is CPU-based
- **Memory**: ~1-2 MB for panel state and logs

### Thread Safety
- Verification runs on detached thread
- Server start uses async thread (500ms delay)
- Proper mutex locking for shared state

---

## Troubleshooting

### Server Won't Start
**Symptom**: "Server Offline" persists after clicking Start
**Solutions**:
1. Check if port 11436 is already in use
2. Verify GRIM-text server executable exists
3. Check logs for startup errors
4. Manually start server via `start_grim_text_server.ps1`

### Time Estimate Shows "Invalid config"
**Cause**: Batch size or epochs set to 0 or negative
**Solution**: Adjust sliders to valid ranges (epochs ≥1, batch size ≥1)

### Verification Button Does Nothing
**Symptom**: Clicking "Run Verification" has no effect
**Solutions**:
1. Check if `verifier.exe` exists in `resources/models/GRIM-text/training/build/`
2. Build verification tool: `cd training && cmake --build build --target verifier`
3. Check logs for error messages

### Learning Rate Shows 0.01 Instead of 0.0001
**Fixed**: Dynamic precision now shows 6 decimals (0.000100)
**If Still Broken**: Verify `ui_slider.cpp` has precision logic based on value range

### Buttons Not Clickable
**Fixed**: Buttons now use `drawOverlay()` instead of manual rendering
**If Still Broken**: Check that button `update()` is called in `UITrainingPanel::update()`

### Progress Bar Stuck at 0%
**Cause**: Server not sending stats updates
**Solutions**:
1. Verify server is actually training (check server console)
2. Ensure server writes to stats file
3. Check FlatBuffer protocol compatibility

---

## Development Notes

### Adding New Configuration Parameters

1. **Add to TrainingConfig struct** (`ui_training_config.hpp`)
2. **Create slider** in `UITrainingPanel` constructor
3. **Add to JSON save/load** in `saveConfigToJSON()` / `loadConfigFromJSON()`
4. **Update estimation** in `calculateTrainingEstimate()` if time-related
5. **Add to UI rendering** in `drawOverlay()` scrollable area

### Extending Data Sources

1. **Define new source_type** in `source_data.json`
2. **Add validation** in `addDataSource()` if special rules needed
3. **Update verifier** to handle new source types
4. **Adjust reliability weights** in verifier configuration

### Custom Verification Metrics

1. **Modify verifier.hpp** to add new check functions
2. **Update Stats struct** to track new rejection reasons
3. **Update UI display** in `updateVerificationStats()` to show new metrics

---

## Architecture Summary

```
UI Training Panel
├── Left Column (Scrollable)
│   ├── Configuration Sliders (5)
│   │   ├── Epochs
│   │   ├── Batch Size
│   │   ├── Learning Rate
│   │   ├── Max Seq Length
│   │   └── Warmup Steps
│   ├── Save Config Button
│   ├── Data Source Section
│   │   ├── URL Input Box
│   │   └── Add Source Button
│   ├── Verification Section
│   │   ├── Run Verification Button
│   │   └── Stats Display
│   └── (Scroll Bar)
├── Right Column
│   ├── Server Status
│   ├── Hardware Info
│   ├── Time Estimate
│   ├── Progress Bar
│   ├── Training Stats
│   ├── Verbose Area (Reserved)
│   └── Logs (Scrollable)
└── Bottom Left (VBox)
    ├── Start Training
    ├── Stop Training
    ├── Pause/Resume
    └── Close Panel
```

---

## Version History

- **v1.0.0** - Initial training panel implementation
  - Basic sliders and configuration
  - Server start/stop controls
- **v1.1.0** - Enhanced UI redesign
  - Moved buttons to bottom left VBox
  - Added progress bar
  - Hardware info display
  - Training time estimation
- **v1.2.0** - Data pipeline integration
  - Source URL input
  - Verification system UI
  - Source-based time calculation
- **v1.3.0** - Precision & calculation fixes
  - Dynamic slider precision
  - Warmup steps affect time
  - Improved process management for verifier

---

## Related Files

- **Header**: `ui/ui_training_panel.hpp`
- **Implementation**: `ui/ui_training_panel.cpp`
- **Training Config**: `ui/ui_training_config.hpp`
- **Server Manager**: `ai/grim_text_server_manager.hpp`
- **Control Client**: `resources/models/GRIM-text/training/control/training_control_client.hpp`
- **Verifier**: `resources/models/GRIM-text/training/verifier.hpp`
- **Global Access**: `main.cpp` (`g_trainingPanel`)

---

## Future Enhancements

### Planned Features
- [ ] **Verbose Metrics**
  - Real-time GPU utilization graph
  - Memory usage charts
  - Token throughput display
  - Gradient norm tracking
  
- [ ] **Enhanced Data Management**
  - Source enable/disable toggles
  - Source priority adjustment
  - Fetch limit editing
  - Collection progress tracking
  
- [ ] **Training History**
  - Loss curve plotting
  - Checkpoint management
  - Model comparison
  - Training session history
  
- [ ] **Advanced Controls**
  - Learning rate schedule editor
  - Custom optimizer settings
  - Gradient clipping controls
  - Mixed precision training toggle
  
- [ ] **Distributed Training**
  - Multi-GPU support
  - Gradient accumulation
  - Pipeline parallelism options

### Community Requests
- Export training logs to file
- Import configuration presets
- Training templates for common tasks
- Notification system for completion

---

**Last Updated**: November 6, 2025  
**GRIM Version**: Development Build  
**Author**: GRIM Development Team
