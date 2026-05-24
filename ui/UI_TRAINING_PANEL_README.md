# GRIM-text Training Control Panel

## Overview

The Training Control Panel provides a comprehensive interface for managing the GRIM-text model training pipeline directly from GRIM.exe. It integrates server management, configuration, data collection, verification, and real-time training monitoring in a single unified interface.

---

## Panel Layout

The panel is divided into two main columns:

### Left Column (35%) - Configuration & Data Management
- **Training Configuration** - Adjustable hyperparameters (5 sliders)
- **Data Source Management** - Add custom data sources
- **Data Pipeline** - Unified collect/verify/merge process
- **Training Data Path** - Specify dataset for training (only path exposed in UI)

### Right Column (65%) - Monitoring & Logs
- **Server Status** - Connection state and data collection state
- **Training State** - Current training phase
- **Training Statistics** - Epoch, batch, loss, perplexity
- **System Resource Monitoring** - Real-time CPU/Memory/GPU graph
- **Training Progress Bars** - Dual progress bars (training & data collection)
- **Verbose Output** - Reserved for future detailed metrics
- **Training Logs** - Real-time scrollable log messages with timestamps
- **Control Buttons** - Start/Stop/Pause/Reset/Close (bottom left)

---

## Configuration Section

### Training Hyperparameters

All sliders feature **dynamic precision** based on their value range:

#### **Epochs** (Range: 1 - 50)
- **Purpose**: Number of complete passes through the training dataset
- **Default**: 10
- **Precision**: Integer (0 decimal places)
- **Impact**: More epochs = longer training, better convergence (risk of overfitting)
- **Time Effect**: Linear - doubles epochs = doubles training time

#### **Batch Size** (Range: 1 - 128)
- **Purpose**: Number of samples processed simultaneously
- **Default**: 8
- **Precision**: Integer (0 decimal places)
- **GPU Impact**: 
  - Larger batches = better GPU utilization
  - Too large = out of memory errors
- **Time Effect**: 
  - Larger batches = fewer batches per epoch = faster
  - Batch size 32+ gets 50% efficiency gain
  - Batch size 16+ gets 20% efficiency gain
  - Batch size 8+ gets standard speed
  - Batch size 4+ gets 30% slowdown
  - Batch size <4 gets 50% slowdown

#### **Learning Rate** (Range: 0.000001 - 0.01)
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
- **Function**: Saves current configuration through the canonical GRIM runtime config accessor
- **Color**: Cyan border (0xFF00AAFF)
- **Persistence**: Configuration survives GRIM restarts
- **Auto-save**: Also triggered when starting training
- **Config Source**: Reads merged `ai_config.json` + `ai_config.local.json`; writes only the main config keys it owns
- **Location**: Below configuration sliders in left panel scrollable area

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

## Data Collection & Verification

### Data Collection Status Indicator
- **Display**: Shows real-time status above the data pipeline button
- **States**:
  - **"🔵 [ACTIVE] Data Collection"** (Cyan): Pipeline is currently running (collecting/verifying)
  - **"🟢 [COMPLETED] Data Collection"** (Green): Pipeline finished successfully (≥99% progress)
  - **"⚪ [IDLE] Data Collection"** (Gray): No active pipeline operations
- **Update Frequency**: Polled every 200ms from server state
- **Progress Display**: Shows percentage when active or completed (e.g., "(85%)")

### Run Data Pipeline Button
- **Label**: "Run Data Pipeline"
- **Function**: Executes unified data collection pipeline (collect → verify → merge)
- **Color**: Cyan border (0xFF00AAFF)
- **Location**: Left panel scrollable area, below data source input
- **Process**:
  1. **In-Process Execution**: Runs via DataCollectionManager (no HTTP, no external server)
  2. **Background Execution**: Runs asynchronously in detached thread (non-blocking UI)
  3. **State Tracking**: Sets `dataCollectionActive` flag and updates UI indicator
  4. **Progress Updates**: Manager reports progress via getStatus() (0-100%)
  5. **Phase Logging**: Logs each pipeline phase (Collecting → Verifying → Merging)
  6. **Auto-Completion Detection**: Automatically detects when pipeline finishes (progress ≥99%)

### Data Pipeline Process

1. **Collection Phase**:
   - Reads enabled sources from `source_data.json`
   - Fetches data from configured URLs
   - Saves raw entries to `data/collected/`
   - Respects `fetch_limit` and rate limiting

2. **Verification Phase**:
   - Domain whitelist approval (40% weight)
   - Content quality validation (40% weight)
     - Min length: 100 characters
     - Max length: 50,000 characters
   - Source type reliability (20% weight)
   - Duplicate detection (Jaccard similarity)
   - Output: Verified entries saved to `data/verified/`

3. **Checkpoint Merge Phase**:
   - Detects existing checkpoint files (`.checkpoint`)
   - Merges with verified data
   - Removes duplicates
   - Outputs final `training_data.grmt`

**Status**: ✅ **WORKING END-TO-END** (as of November 8, 2025)
- All pipeline phases execute successfully
- Data collection → verification → training flow validated
- Model training completes and saves output file

### Data Collection Progress Bar
- **Display**: Cyan-colored progress bar in right panel
- **Range**: 0-100%
- **Updates**: Smooth animated progress (30% interpolation to prevent visual jumps)
- **Resets**: Automatically resets to 0% when new collection starts
- **Completion**: Holds at final progress when collection completes
- **Current Phase Display**: Shows pipeline phase text below bar (e.g., "Phase: Collecting from sources...")

### Pipeline Status Tracking
- **Real-time Indicator**: Status changes based on actual pipeline state
- **Phase Logging**: 
  ```
  [19:25:12] Starting data collection pipeline...
  [19:25:13]   → Collecting from sources...
  [19:26:45]   → Verifying data quality...
  [19:27:30]   → Merging with existing data...
  [19:28:01] ✓ Data collection completed successfully!
  ```
- **Automatic Reset**: Status indicator updates when pipeline completes or errors
- **Progress Tracking**: UI polls collectionManager->getStatus() every frame during update cycle

### Dataset Information Display
Shows in left panel scrollable area:
- **Dataset Size Info**: Displays number of processed/verified samples in training-ready dataset
  - Color: Bright cyan (0xFF00FFAA)
  - Example: "Dataset: 1,234 samples ready"
- **Checkpoint Stats Info**: Shows raw collected data before merging
  - Color: Gray (0xFF888888)
  - Example: "Checkpoints: 567 raw entries"
  - Displayed in right panel above dataset info

### Verification Stats Display
Shows in left panel after pipeline completes (if verification_stats.json exists):
```
Processed: 150
Passed: 90
Failed: 60
```
- **Processed**: Total entries checked
- **Passed**: Entries meeting reliability threshold
- **Failed**: Rejected entries (domain/quality/duplicate)
- **Color**: Gray (0xFF808080)

---

## Training Control Buttons

All control buttons are stacked vertically at the bottom left of the panel using a VBox layout.

### Start Training Button
- **Label**: "▶ Start Training"
- **Color**: Green border (0xFF00FF00)
- **Size**: 140px wide × 35px tall
- **Enabled When**: Always clickable (internal validation handles state checks)
- **Function**:
  1. Checks server connection, attempts reconnect if disconnected
  2. Validates training is not already in progress (checks server state)
  3. Updates configuration from sliders
  4. Validates training data exists (train.bin or .grmt file)
  5. Loads paths from the canonical GRIM runtime config
  6. Sends training start command via TrainingControlClient
  7. Resets progress bar and statistics
- **Smart State Handling**: Detects and corrects UI/server state desync
- **Data Validation**: Warns if no training data found, suggests running data pipeline

### Stop Training Button
- **Label**: "⏹ Stop Training"
- **Color**: Red border (0xFFFF0000)
- **Size**: 140px wide × 35px tall
- **Enabled When**: Always clickable (internal validation handles state checks)
- **Function**:
  1. Validates server connection
  2. Checks if training is actually running
  3. Sends stop command to server via TrainingControlClient
  4. Logs stop request
- **Safety**: Internal checks prevent stopping when not training

### Pause/Resume Button
- **Label**: "⏸ Pause Training" or "▶ Resume Training"
- **Color**: Orange border (0xFFFFAA00)
- **Size**: 140px wide × 35px tall
- **Enabled When**: Always clickable (internal validation handles state checks)
- **Function**:
  - **When Training**: Sends pause command via TrainingControlClient
  - **When Paused**: Sends resume command via TrainingControlClient
- **State Aware**: Button text updates automatically based on currentState

### Reset Status Button
- **Label**: "Reset Status"
- **Color**: Yellow border (0xFFFFAA00)
- **Size**: 140px wide × 35px tall
- **Always Enabled**: Can always reset status
- **Function**: 
  1. Calls resetState() to clear all training state flags
  2. Clears checkpoint merge status
  3. Removes stale training_status.fb file if it exists
  4. Logs reset action
- **Use Case**: Clear stale state when UI desynchronizes from server

### Close Panel Button
- **Label**: "Close Panel"
- **Color**: Cyan border (0xFF00AAFF)
- **Size**: 140px wide × 35px tall
- **Always Enabled**: Can always close the panel
- **Function**: Hides the panel (`setVisible(false)`)
- **Server Behavior**: Does NOT stop the training server
- **Use Case**: Close UI while training continues in background

---

## Right Panel - Monitoring

### Server Status (Top Right)

**Training Server Connection Status**:
- 🟢 **"[ONLINE] Training"** (Green): Connected to training server
- 🔴 **"[OFFLINE] Training"** (Red): No connection, server not running
- **Location**: Fixed at top of right panel (not scrollable)
- **Update Frequency**: Polled every 200ms via client->isServerRunning()

**Data Collection Status** (Real-time):
- 🔵 **"[ACTIVE] Data Collection"** (Cyan): Pipeline is currently running
- 🟢 **"[COMPLETED] Data Collection"** (Green): Pipeline finished (progress ≥99%)
- ⚪ **"[IDLE] Data Collection"** (Gray): No active pipeline operations
- **Location**: Below server status in right panel
- **Update Frequency**: Updated every frame from dataCollectionActive flag
- **Accuracy**: Flag set by startDataCollection(), cleared when pipeline completes

**Training State**:
- **Display**: "State: [STATE]" with color coding
- **Possible States**:
  - **Idle**: Ready to start training (White)
  - **Training**: Active training in progress (Green)
  - **Collecting**: Gathering data from sources (Cyan)
  - **Verifying**: Running quality checks on collected data (Yellow)
  - **Paused**: Training paused, can resume (Orange)
  - **Completed**: Training finished successfully (Green)
  - **Error**: Training encountered an error (Red)
- **Location**: Below data collection status

### System Resource Monitoring

**Resource Monitoring Graph**:
- **Type**: Area graph with 3 data series
- **Series**:
  - **CPU Usage** (Cyan - 0xFF00AAFF): Percentage of CPU utilization
  - **Memory Usage** (Green - 0xFF00FF00): Percentage of RAM used
  - **GPU Usage** (Orange - 0xFFFF6600): Percentage of GPU utilization
- **Y-Axis Range**: 0-100%
- **Sample Rate**: 500ms (2 samples per second)
- **History**: 60 samples (30 seconds of data)
- **Features**:
  - Auto-scaling disabled (fixed 0-100% range)
  - Grid lines enabled (5 horizontal lines)
  - Legend enabled (bottom of graph)
  - Axis labels enabled
  - Line thickness: 2px
- **Location**: Right panel, below training state
- **Size**: Full right panel width - 20px × 200px height

**Resource Sampling**:
- Samples CPU, memory, and GPU every 500ms during training
- Uses system_detect functions for accurate readings
- Clamps values to 0-100% range to prevent display issues

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
- **Label**: "Training Progress"
- **Width**: Full right panel width - 20px
- **Height**: 30px
- **Fill Color**: Dark green (0xFF00AA00)
- **Background**: Very dark gray (0xFF1A1A1A)
- **Location**: Right panel, below resource graph

**Progress Source**:
- Uses `currentStats.trainingProgress` directly from server (0-100%)
- Converted to 0.0-1.0 range for UIProgressBar
- Server calculates progress on every batch update in train_gpu.exe

**Real-time Updates**:
- Polls server every 200ms (0.2 seconds) for fast training visibility
- Updates based on `TrainingStats` from server status checks
- Shows percentage: "0.0%" to "100.0%"
- Optimized for rapid training sessions (completes in seconds on small datasets)

### Data Collection Progress Bar

**Visual Indicator**:
- **Label**: "Data Collection Progress"
- **Width**: Full right panel width - 20px
- **Height**: 30px
- **Fill Color**: Cyan (0xFF00AAFF)
- **Background**: Very dark gray (0xFF1A1A1A)
- **Location**: Right panel, below training progress bar

**Progress Source**:
- Uses `currentStats.collectionProgress` directly from server (0-100%)
- Converted to 0.0-1.0 range for UIProgressBar
- Updated by DataCollectionManager every second during collection

**Smooth Animation**:
- Implements 70/30 interpolation to prevent visual jumps
- Resets to 0% when new collection starts
- Progress never decreases during active collection
- Holds final value when collection completes

**Current Phase Display**:
- Shows below progress bar when active
- Example: "Phase: Collecting from sources..."
- Color: Cyan (0xFF00AAFF)
- Source: `currentStats.currentPhase` from server

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
- GPU utilization graphs (currently shown in resource monitor)
- Memory usage tracking (currently shown in resource monitor)
- Token throughput metrics (tokens/second)
- Gradient statistics
- Learning rate schedule visualization
- Per-layer activation statistics

**Current Display**:
- Placeholder text in gray
- Located between control buttons and logs section
- ~110px height reserved

### Training Logs

**Scrollable Log Area**:
- **Auto-scroll**: Shows newest messages at bottom
- **Max entries**: 1000 logs (oldest removed when exceeded)
- **Timestamp format**: `[HH:MM:SS]` (24-hour format)
- **Location**: Bottom of right panel
- **Background**: Very dark gray (0xFF0A0A0A)
- **Border**: Dark gray (0xFF303030)
- **Thread-safe**: Uses mutex for concurrent log access
- **Line height**: 18px per entry
- **Visible logs**: Calculated based on available height

**Log Levels & Colors**:
- **Level 0 (Green - 0xFF00FF00)**: Info messages
  - Configuration loaded/saved
  - Training panel initialized
  - Training progress milestones
  - Data collection phases
- **Level 1 (Yellow/Orange - 0xFFFFAA00)**: Warnings
  - Validation issues
  - State desynchronization
  - Non-critical errors
  - Retry attempts
- **Level 2 (Red - 0xFFFF0000)**: Errors
  - Connection failures
  - Training errors
  - File I/O errors
  - Missing data files

**Example Log Output**:
```
[19:17:12] Training panel initialized
[19:17:12] Configuration loaded from GRIM runtime config
[19:18:56] Configuration saved to GRIM runtime config
[19:19:02] Server connection established
[19:20:15] Training data found, starting training...
[19:20:15]   Using tokenized binary: train.bin
```

---

## File Integration

### Configuration Files

#### `ai_config.json` (canonical main config at GRIM root)

Runtime reads merge `ai_config.local.json` on top for machine-local overrides; UI saves write only the main config keys they own.

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
  },
  "paths": {
    "grim_text": {
      "vocab": "resources/models/GRIM-text/training/vocab.bin",
      "model": "resources/models/GRIM-text/grim_text.bin",
      "training_data": "resources/models/GRIM-text/training/data/training_data.grmt",
      "checkpoints": "resources/models/GRIM-text/training/checkpoints",
      "logs": "resources/models/GRIM-text/training/logs",
      "training_status": "resources/models/GRIM-text/training/training_status.fb"
    }
  }
}
```
- Loaded on panel initialization via loadPathsFromConfig()
- Saved when "Save Config" clicked or training data path changes
- Persists between GRIM sessions
- All paths loaded in background, only training_data path exposed in UI

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

### Training Control Client

The panel communicates with the training server via `TrainingControlClient`:

**Communication Method**: Direct client connection (not FlatBuffer)

**Client Operations**:
- `isServerRunning()` - Check if server is responsive
- `getStatus(state, stats, config)` - Retrieve current training state and statistics
- `startTraining(config)` - Begin training session with configuration
- `stopTraining()` - Halt training
- `pauseTraining()` - Pause training
- `resumeTraining()` - Resume from pause

**Polling**:
- **Interval**: 200ms (0.2 seconds) - optimized for fast training visibility
- **Automatic**: Runs continuously when panel visible
- **Thread-safe**: Uses atomic flag to prevent concurrent polling
- **Async**: Runs on detached thread to avoid blocking UI
- **Gets**: `TrainingState`, `TrainingStats`, `TrainingConfig` from server
- **Crash Detection**: Monitors for stalled progress (75 polls = ~15 seconds)

**Data Collection**:
- **Method**: In-process via DataCollectionManager (no server communication)
- **Polling**: Checks status every frame via collectionManager->getStatus()
- **Progress**: Updates UI with progress, phase, and running state
- **Completion**: Auto-detects when progress reaches ≥99%

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
- **Scrollable**: When content exceeds height
- **Scroll bar**: Right edge (cyan - 0xFF00FFFF when scrollbar, 0xFF202020 track)
- **Track**: Dark gray background
- **Handle Size**: Proportional to content/viewport ratio
- **Handle Position**: Follows scroll position
- **Mouse Support**: Click and drag scrollbar (mouse wheel may be implemented)

**Logs Area**:
- **Auto-scroll**: Always shows newest entries at bottom
- **Display**: Shows only entries that fit in viewport (calculated by height / 18px)
- **Rendering**: Skips entries outside visible area for performance

---

## Performance Considerations

### Update Frequency

- **Server Polling**: Every 200ms (0.2 seconds) for real-time progress visibility
- **UI Refresh**: Every frame (~60 FPS)
- **Log Limit**: 1000 entries to prevent memory bloat
- **Resource Sampling**: Every 500ms (0.5 seconds) for CPU/Memory/GPU monitoring
- **Resource History**: 60 samples (30 seconds of data retained)
- **Data Collection Polling**: Every frame via getStatus() when dataCollectionActive flag set

### Resource Usage
- **Minimal CPU**: Polling only when visible
- **No GPU**: UI rendering is CPU-based
- **Memory**: ~1-2 MB for panel state and logs

### Thread Safety
- **Data collection**: Runs on detached thread via DataCollectionManager
- **Server polling**: Runs on detached thread (async, non-blocking)
- **Atomic polling flag**: Prevents concurrent polling requests
- **Log mutex**: Thread-safe log access via std::mutex
- **Status polling**: Released immediately after completion to allow next poll

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

### Data Pipeline Returns Immediately
**Symptom**: Pipeline completes instantly or doesn't start
**Causes**: 
1. No enabled sources in source_data.json
2. DataCollectionManager already has an active pipeline
3. Missing data collection configuration

**Solutions**:
1. Add data sources via "Add Source" input box
2. Check source_data.json for enabled sources
3. Click "Reset Status" button to clear state
4. Verify DataCollectionManager status via logs

**Technical Details**:
- UI maintains `dataCollectionActive` flag to prevent duplicate calls
- Pipeline runs in-process via DataCollectionManager (no server/HTTP)
- `pipelineRequestPending` flag prevents concurrent requests
- Status polled every frame via collectionManager->getStatus()
- Auto-completion detection when progress ≥99% and isRunning = false

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
├── Left Column (35% width, Scrollable)
│   ├── Configuration Sliders (5)
│   │   ├── Epochs (1-50)
│   │   ├── Batch Size (1-128)
│   │   ├── Learning Rate (0.000001-0.01)
│   │   ├── Max Seq Length (512-16384)
│   │   └── Warmup Steps (0-5000)
│   ├── Save Config Button
│   ├── Data Source Section
│   │   ├── URL Input Box
│   │   └── Add Source Button
│   ├── Data Collection Section
│   │   ├── Collection Status Indicator
│   │   └── Run Data Pipeline Button
│   ├── Training Data Path Section
│   │   └── Training Data Input Box (only path shown in UI)
│   ├── Verification Stats Display (gray text)
│   ├── Dataset Size Info (cyan text)
│   └── (Scroll Bar - cyan)
├── Right Column (65% width)
│   ├── Training Server Status (fixed top)
│   ├── Data Collection Status (with progress %)
│   ├── Checkpoint Stats Info (gray)
│   ├── Dataset Size Info (cyan)
│   ├── Training State (colored by state)
│   ├── Training Statistics (Epoch/Batch/Loss/Perplexity)
│   ├── System Resource Graph (CPU/Memory/GPU - 200px height)
│   ├── Training Progress Bar (green fill)
│   ├── Data Collection Progress Bar (cyan fill)
│   │   └── Current Phase Display
│   ├── Verbose Output Area (reserved - ~110px)
│   └── Training Logs (scrollable, color-coded by level)
└── Bottom Left (VBox - 140×35px buttons)
    ├── Start Training (green)
    ├── Stop Training (red)
    ├── Pause/Resume Training (orange)
    ├── Reset Status (yellow)
    └── Close Panel (cyan)
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
- **v1.4.0** - Real-time progress optimization (November 6, 2025)
  - Reduced polling interval from 1.5s to 200ms for fast training visibility
  - Fixed working directory for train_gpu.exe to enable proper log/status file creation
  - Added server auto-detection to prevent duplicate server instances
  - Updated verifier path to correct build location (build_vs_cuda/Release/)
  - Server now properly monitors training_status.fb via StatusFileMonitor (500ms interval)
  - Progress bar now updates smoothly even for training sessions completing in seconds
- **v1.5.0** - Unified data pipeline & status tracking (November 7, 2025)
  - Replaced separate collect/verify buttons with unified "Run Data Pipeline" button
  - Added real-time Data Collection Status indicator (🔵 Active / ⚪ Idle)
  - Status indicator automatically syncs with server state every 200ms
  - Fixed race condition preventing duplicate pipeline execution
  - `dataCollectionActive` flag now accurately reflects server state
  - Improved state management for collection/verification operations
  - Enhanced guard checks with server state verification
  - Added Reset Status button for clearing stale states
- **v1.6.0** - Training system fully operational (November 8, 2025)
  - ✅ **END-TO-END TRAINING VALIDATED**: Complete training pipeline working
  - Fixed stale status file detection - server now clears old state on startup
  - Fixed GRIM root path detection - eliminated path doubling bug
  - Fixed FlatBuffer config parsing - now reads dataPath/vocabPath/outputPath
  - Added comprehensive file-based debug logging (training_control_debug.log)
  - Thread-safe timestamp logging with localtime_s()
  - Training completes successfully: 50 epochs, model saved (347 KB)
  - Best validation loss: 0.526225, final train perplexity: 5.091650
- **v1.7.0** - UI refinement and path management (November 12, 2025)
  - Removed 4 path input boxes from UI (vocab, model, checkpoints, logs)
  - Kept only training data path input exposed to user
  - All paths still loaded/saved in background from ai_config.json
  - Simplified left panel configuration area
  - Reduced file size from 2,100 lines to 2,017 lines (~83 lines removed)
  - Improved UI clarity - users only see what they need to configure
  - Background path management remains fully functional

---

## Current Status: ✅ FULLY OPERATIONAL

**Validated November 8, 2025:**

- Data collection pipeline: ✅ Working
- Data verification: ✅ Working
- Training initiation: ✅ Working
- Training execution: ✅ Working (50 epochs completed)
- Model output: ✅ Working (grim_text_trained.bin saved successfully)
- Real-time progress tracking: ✅ Working (200ms polling)
- Server state management: ✅ Working (no more stale states)

**Known Working Configuration:**

- Epochs: 50
- Batch Size: 8
- Learning Rate: 0.0001
- Max Seq Length: 512
- Training Time: ~3 seconds (small dataset)
- Output: 347 KB trained model

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

**Last Updated**: November 12, 2025  
**GRIM Version**: Development Build  
**File Size**: 2,017 lines (reduced from 2,100)  
**Training System Status**: ✅ Fully Operational  
**Author**: GRIM Development Team
