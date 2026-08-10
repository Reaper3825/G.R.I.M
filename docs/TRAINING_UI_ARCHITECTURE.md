# GRIM-text Training UI - Real-Time Progress Tracking

> Historical architecture note. The GRIM UI no longer launches or monitors
> local training. The former Training tab is now the per-model `.grimcfg`
> creator documented in `ui/UI_TRAINING_PANEL_README.md`; HPC training is
> managed through the SSH workflow.

## Overview
Complete real-time training monitoring system with **zero coupling** between GRIM.exe and GRIM-text training backend.

## Architecture

```
┌─────────────────┐         HTTP          ┌──────────────────────────┐
│   GRIM.exe      │◄──────────────────────►│ training_control_server  │
│                 │   Port 11436           │      (localhost)         │
│  UI Panel       │                        │                          │
│  - Progress Bar │                        │  - Monitors status file  │
│  - Metrics      │                        │  - Launches train_gpu    │
│  - Logs         │                        │  - Serves HTTP API       │
└─────────────────┘                        └────────────┬─────────────┘
                                                        │
                                                        │ Process launch
                                                        ▼
                                            ┌──────────────────────────┐
                                            │   train_gpu.exe          │
                                            │                          │
                                            │  - Runs training loop    │
                                            │  - Writes status file    │
                                            │    EVERY batch           │
                                            └──────────────────────────┘
                                                        │
                                                        │ Writes
                                                        ▼
                                            ┌──────────────────────────┐
                                            │  training_status.fb      │
                                            │  (FlatBuffer file)       │
                                            │                          │
                                            │  Updated every batch!    │
                                            └──────────────────────────┘
```

## Components

### 1. train_gpu.exe (GRIM-text backend)
**File:** `resources/models/GRIM-text/training/train_gpu.cu`

**What it does:**
- Runs GPU-accelerated training loop
- Writes `training_status.fb` on **EVERY batch** with:
  - Current epoch/batch progress
  - Loss, perplexity, tokens/sec
  - GPU memory usage
  - Training phase
  - Error states

**Key feature:** Status updates on every batch (not just logging intervals) for smooth UI updates regardless of epoch length.

### 2. training_control_server.exe (HTTP bridge)
**File:** `resources/models/GRIM-text/training/control/training_control_server.cpp`

**What it does:**
- Runs independent HTTP server on `localhost:11436`
- Monitors `training_status.fb` file (500ms polling)
- Launches/manages `train_gpu.exe` process
- Serves status via HTTP endpoints

**API Endpoints:**
- `GET /health` - Server health check
- `GET /api/status` - Get current training status (FlatBuffer)
- `POST /api/training/start` - Start training with config
- `POST /api/training/stop` - Stop training
- `POST /api/config` - Update configuration

**Key feature:** Server stays alive between training sessions - no need to restart!

### 3. GRIM.exe UI Panel
**Files:** 
- `ui/ui_training_panel.cpp`
- `ui/ui_training_panel.hpp`
- `ui/ui_training_config.hpp`

**What it does:**
- Polls server every 1.5 seconds
- Updates progress bar in real-time
- Shows all metrics (loss, perplexity, GPU memory, etc.)
- Manages training lifecycle (start/stop/pause)

**Key feature:** Server connection is persistent - doesn't shut down when training completes.

## Data Flow

1. **User clicks "Start Training" in GRIM.exe**
   - UI sends config to training control server

2. **Server receives start command**
   - Launches `train_gpu.exe` with absolute paths
   - Returns immediately (non-blocking)

3. **train_gpu.exe starts training**
   - Loads data and model
   - Enters training loop
   - **On every batch:**
     - Calculates metrics
     - Writes `training_status.fb` atomically
     - Logs to console (at intervals)

4. **Server monitors status file**
   - Reads `training_status.fb` every 500ms
   - Detects if training process is still alive
   - Serves latest status via HTTP

5. **UI polls server**
   - Requests status every 1.5 seconds
   - Updates progress bar
   - Updates metrics display
   - Updates logs

6. **Training completes**
   - `train_gpu.exe` writes final status with `Completed` state
   - Server detects completion
   - **Server STAYS ALIVE** for next training session
   - UI shows completion message

## File Communication Protocol

### training_status.fb (FlatBuffer)
```
StatusResponse {
    state: TrainingState (Idle/Training/Paused/Completed/Error)
    stats: TrainingStats {
        current_epoch: int
        total_epochs: int
        current_batch: int       ← Updated every batch!
        total_batches: int       ← Updated every batch!
        current_loss: float
        avg_loss: float
        perplexity: float
        tokens_per_sec: float
        gpu_memory_used: float
        gpu_memory_total: float
        training_progress: float  ← 0-100%
        current_phase: string
        last_error: string
        start_time: int64
        elapsed_time: int64
    }
    config: TrainingConfig { ... }
    process_running: bool
    timestamp: int64
}
```

## Progress Tracking

The UI can determine training progress using:

1. **Batch Progress:** `(current_batch / total_batches) * 100%`
2. **Epoch Progress:** `(current_epoch / total_epochs) * 100%`
3. **Overall Progress:** `((current_epoch - 1) * batches_per_epoch + current_batch) / (total_epochs * batches_per_epoch) * 100%`

**Key:** Batch count increments on every batch, so UI can:
- Show smooth progress bar updates
- Detect if training is stalled (batch not changing)
- Calculate accurate time remaining

## Error Handling

### Errors are captured at multiple levels:

1. **Data Loading Errors**
   - Status written with `Error` state
   - `last_error` contains error message
   - UI displays error in panel

2. **Training Crashes**
   - Server detects process exit
   - `process_running` becomes false
   - UI shows "Training process stopped"

3. **Server Connection Lost**
   - UI polls fail
   - `serverConnected` becomes false
   - UI shows "Server offline"

## Benefits

✅ **Zero Coupling:** GRIM.exe and GRIM-text are completely independent
✅ **Real-Time Updates:** Status written every batch for smooth progress
✅ **Persistent Server:** No need to restart between training sessions
✅ **Works for Any Epoch Length:** Batch-level granularity
✅ **Robust Error Handling:** Errors captured and displayed
✅ **File-Based Communication:** Simple, reliable, no complex networking
✅ **Atomic Writes:** Temp file + rename prevents corruption

## Testing

Run the integration test:
```powershell
./test_training_ui_integration.ps1
```

This will:
1. Verify all executables exist
2. Check training data is present
3. Start training control server
4. Test HTTP endpoints
5. Leave server running for manual testing

## Manual Testing Flow

1. Start GRIM.exe: `./out/build/Release/GRIM.exe`
2. Press `T` to open training panel
3. (Optional) Adjust hyperparameters with sliders
4. Click "Start Training"
5. Watch real-time updates:
   - Progress bar advances every batch
   - Metrics update (loss, perplexity, tokens/sec)
   - GPU memory shown
   - Logs scroll
6. Training completes automatically
7. Server stays alive - can start another session!

## Configuration

Training config stored in `ai_config.json`:
```json
{
  "training": {
    "server_host": "127.0.0.1",
    "server_port": 11436,
    "config": {
      "epochs": 3,
      "batch_size": 8,
      "learning_rate": 0.0001,
      "max_seq_len": 8192,
      "warmup_steps": 1000,
      "use_gpu": true,
      "use_flash_attention": true,
      "data_path": "data/training_data.grmt",
      "vocab_path": "models/vocab.bin",
      "output_path": "models/grim_text_trained.bin"
    }
  }
}
```

## Future Enhancements

- [ ] Pause/Resume functionality
- [ ] Checkpoint browsing
- [ ] Training history graph
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] Live training data preview
- [ ] Model comparison tools
