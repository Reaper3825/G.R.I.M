# G.R.I.M Project — Weekly Progress Report

**Period:** April 3–9, 2026  
**Author:** Austin Wadkins  
**Prepared for:** PI Review Meeting

---

## Executive Summary

This week represented substantial progress across both the GRIM-text language model training infrastructure and the broader G.R.I.M assistant platform. The work fell into five major feature areas, each motivated by specific research or engineering needs:

| # | Feature Area | Motivation |
|---|-------------|-----------|
| 1 | Training Observability & Optimizer Diagnostics | Move from sampled log-line diagnostics to continuous, per-step telemetry with causal analysis tooling |
| 2 | Tokenizer Quality & Vocabulary Management | Stabilize Unigram pruning, add runtime vocabulary tuning, harden byte fallback guarantees |
| 3 | Cross-Platform macOS Port | G.R.I.M assistant UI, TTS, and process management running natively on macOS |
| 4 | Multi-Device Communication & Shared Storage | Full device pairing, hub-spoke communication, content-addressable file sharing — new subsystem from scratch |
| 5 | Checkpoint Reliability & Serialization | Post-write verification, zero-fill detection, CUDA state dumps — eliminate silent checkpoint corruption |
| 6 | Diagnostic Inference Sampling | Live text generation during training for qualitative monitoring of model behavior |

**By the numbers:** 45 commits, 302 files touched, ~20,200 lines added, ~4,000 removed across 6 working days.

---

## Feature 1: Training Observability & Optimizer Diagnostics

### Problem

Our diagnostics up to now relied on periodic log-line output (gradient norms every 10 batches, loss every batch). This sampling approach made it impossible to see *exactly when* training dynamics shifted during warmup, or to correlate optimizer internal state with loss trajectory at full resolution.

### What Was Built

**1. TelemetryLattice CSV Export Pipeline**

The GRIM-text training system already had a hierarchical telemetry system called TelemetryLattice — a GPU-resident streaming statistics engine with 8 aggregation levels and 5 metric streams. However, this data was only accessible through log output.

This week, a new `TelemetryCsvLogger` component was built that:

- Dumps **every (stream × level) pair** to CSV at **every training step**
- Records raw observations alongside running aggregates (mean, variance, min, max) at each hierarchy level
- Produces machine-readable data suitable for pandas/matplotlib analysis

The hierarchy matters: level 0 is raw per-step, level 1 aggregates every 2 steps, level 2 every 4, etc. up to level 7 (128-step windows). This lets us zoom in on transient events or zoom out for trend analysis without re-running training.

**2. Telemetry Viewer Tool (`view_telemetry.py`)**

A 580-line interactive Python visualization tool that:

- Auto-discovers the latest telemetry CSV in the training logs directory
- Pivots the multi-stream data into time-aligned columns
- Generates multi-panel plots: loss trajectory, learning rate schedule, gradient norms per component, and custom metric overlays
- Supports filtering by stream name, level, and step range

This tool is designed for rapid iteration — run a training session on Anvil, pull the CSV, and immediately visualize any metric of interest.

**3. Adam Warmup Causation Analysis (`_warmup_causation.py`)**

This is a theoretical analysis tool that computes, from first principles, the causal chain behind loss behavior during Adam warmup. The key insight it quantifies:

- **Adam's second moment (v) has a half-life of ~693 steps** (from β₂=0.999). Before the v-estimates converge, per-parameter step sizes are essentially random — some parameters get 10–100× larger updates than appropriate.
- **Linear warmup ramps lr from 0 → 6e-4 over 1,000 steps**, so each step's displacement is larger than the last. Cumulative parameter displacement from Xavier initialization grows quadratically: $\sum_{t=1}^{T} \text{lr}(t) \propto T^2$.
- **By step ~700**, total displacement is approximately 6× the Xavier embedding scale. The initial uniform-logit equilibrium is destroyed.
- **The loss peak around step 700 coincides with v reaching 50% convergence** — Adam finally "knows" which parameters need big vs. small steps. The signal-to-noise ratio of updates jumps above 1.0 and recovery begins.

The script computes all these quantities and writes them to CSV for overlay with actual telemetry data. This gives us a predictive model for warmup dynamics that we can validate against observed training curves.

**4. Execution Block Health Metrics**

Extended the telemetry system to monitor the ScratchBlock (structured reasoning) layer:

- Per-step entropy of argument selection, operation selection, and write-gate distributions
- Collapse detection: alarms when any distribution's entropy drops below threshold (indicating the block is stuck selecting the same operation/arguments)
- Slot utilization tracking: what fraction of the scratch memory slots are actively being read/written

### What This Enables

We can now do in minutes what previously took multiple training runs:

1. Compare the theoretical warmup causation model against actual per-step telemetry
2. Pinpoint the exact step where dynamics shift and which metrics change first
3. Test schedule changes (e.g., longer warmup, β₂ annealing) and immediately visualize the effect without guesswork
4. Monitor ScratchBlock health continuously — catch entropy collapse or write-gate saturation the moment it happens

---

## Feature 2: Tokenizer Quality & Vocabulary Management

### Problem

The Unigram + Byte Fallback tokenizer is the only path from text to model input. Vocabulary quality has a direct, measurable impact on model performance:

- **Over-fragmented vocabulary**: Common words split into too many subwords → longer sequences → more attention computation → less effective context window
- **Under-pruned vocabulary**: Rare subwords consuming embedding table rows → wasted model capacity
- **Unstable pruning**: Removing low-frequency subwords during training causing vocabulary oscillation

### What Was Improved

**Pruning Stability (April 3–4)**

The Unigram language model trainer uses an EM algorithm that iteratively scores subwords and prunes the lowest-scoring ones. Two iterations of refinement this week:

1. **Removed backfill during pruning** — Previously, when a subword was pruned, the system tried to backfill the vocabulary slot. This created cascading re-scoring issues. The new approach simply prunes and allows the vocabulary to shrink, with re-mining only in designated phases.

2. **Frequency threshold tuning** — Enabled pruning during the mining phase (previously pruning only happened post-mining). This prevents the vocabulary from growing unbounded during initial subword discovery, keeping memory usage predictable.

**Vocabulary Score Multiplier (April 9)**

Added a `vocab_score_multiplier` parameter that adjusts subword log-probabilities when saving the vocabulary to disk. This allows:

- Boosting scores of morphologically meaningful subwords (e.g., "-ing", "-tion") without re-training
- Attenuating noise subwords that survived pruning due to corpus bias
- A/B testing vocabulary variants by adjusting the multiplier rather than re-running the full trainer

**Byte Limit Handling (April 8)**

Improved robustness of the byte token range [0–255]. The tokenizer must guarantee 100% UTF-8 coverage — any byte sequence that can't be tokenized as a learned subword falls back to individual byte tokens. The refinement ensures that byte tokens are never accidentally pruned or their scores corrupted during vocabulary updates.

### Relevance to Model Performance

Token layout in GRIM-text: `[0-255] = bytes`, `[256-511] = atom placeholders`, `[512+] = unigram vocab`. The target vocabulary size is 10,000 tokens. Every subword token that doesn't carry meaningful semantic information is a wasted row in the 10,000 × 768 embedding matrix — that's 768 parameters not contributing to model quality. Good vocabulary management is essentially free model capacity.

---

## Feature 3: Cross-Platform macOS Port

### Motivation

Development and debugging happen on macOS (local machine), while training runs on Linux GPU clusters (Anvil). The G.R.I.M assistant — the user-facing application that will eventually use GRIM-text for language understanding — needs to run on the development machine for UI iteration and testing.

### What Was Ported

**Popup 3D Renderer**

G.R.I.M presents responses through an animated popup window with a 3D character avatar. This week the rendering pipeline was ported to macOS:

- **3D Object Renderer**: Built a new OpenGL-based renderer that loads Wavefront .obj meshes, applies textures, and renders with custom shaders. This is the visual centerpiece of G.R.I.M's UI — the character that "speaks" responses.
- **OBJ Mesh Loader**: Parser for .obj format with vertex positions, normals, texture coordinates, and face index support.
- **Texture Loading**: Integrated stb_image for cross-platform texture loading (PNG, JPG).
- **Shader Pipeline**: Vertex and fragment shaders for basic Phong lighting with texture mapping.

The previous Windows implementation used platform-specific rendering APIs. The new implementation uses portable OpenGL, making it the foundation for both macOS and Linux desktop builds.

**Apple TTS Integration**

G.R.I.M has a voice output system. On Windows, this uses SAPI (Speech API). This week:

- Integrated Apple's `AVSpeechSynthesizer` framework for native macOS text-to-speech
- Connected it to the existing voice output pipeline so the same API (`voice_speak()`) works cross-platform
- No quality compromise — Apple's neural TTS voices are comparable or better than SAPI

**Process Management**

G.R.I.M orchestrates multiple processes (training server, inference server, etc.). The `ProcessManager` was extended with:

- POSIX `fork()`/`exec()` for macOS (replacing Windows `CreateProcess`)
- Signal-based process management (`SIGTERM`, `SIGKILL` for cleanup)
- Proper file descriptor inheritance handling

**Input Handling**

macOS keyboard layouts differ from Windows (⌘ vs Ctrl, different key codes). Updated the key mapping layer so shortcuts work correctly on macOS.

### Current State

The G.R.I.M assistant UI now launches and renders on macOS with:
- ✅ 3D character avatar with shader-based rendering
- ✅ Text-to-speech via Apple's neural voices
- ✅ Process management for background services
- ✅ Keyboard input handling
- 🔲 Full UI panel integration (in progress)

---

## Feature 4: Multi-Device Communication & Shared Storage

### Vision

G.R.I.M is designed as a **multi-device personal assistant**. The long-term vision is that a user's phone, tablet, laptop, and desktop all participate in a shared ecosystem — G.R.I.M running on the desktop acts as the "hub" that coordinates context, files, and capabilities across devices.

This week, the foundational communication layer was built from scratch.

### Architecture: Hub-Spoke Model

```
  ┌─────────────┐     WebSocket      ┌──────────────┐
  │  Phone App  │ ◄──────────────► │              │
  └─────────────┘                    │   G.R.I.M    │
                                     │   Desktop    │     ┌──────────────────┐
  ┌─────────────┐     WebSocket      │   (Hub)      │ ──► │  Shared Storage  │
  │  Tablet App │ ◄──────────────► │              │     │  (Local Disk)    │
  └─────────────┘                    └──────────────┘     └──────────────────┘
```

The hub runs on the desktop where G.R.I.M is installed. Satellite devices connect over the local network via WebSocket.

### Feature: Device Registration & Pairing

The pairing flow is designed to be simple and secure:

1. **Hub generates a pairing code** — An 8-character code (XXXX-XXXX format) displayed in the G.R.I.M UI
2. **User enters code on remote device** — The phone/tablet app sends a `RegisterMessage` with the code over WebSocket
3. **Hub validates and creates device entry** — The device is registered in a persistent `DeviceRegistry` with its name, type (phone/tablet/desktop), platform (iOS/Android/Windows/macOS), and declared capabilities
4. **Mutual confirmation** — Both sides acknowledge pairing; device enters "paired" state

The registry persists across restarts — once paired, a device reconnects automatically.

### Feature: Shared Storage with Content-Addressable Index

Once devices are paired, they can share files through a centralized storage system:

- **StorageManager** handles file I/O on the hub's local disk
- **StorageIndex** maintains a content-addressable index keyed by file hash — if two devices upload the same file, it's stored once
- **FileTransferManager** handles chunked transfer over WebSocket — large files are split into pieces for reliable transfer over potentially unstable connections

### Feature: UI Integration

A new "Shared Storage" panel was added to G.R.I.M's console UI:

- Displays connection status for each paired device (online/offline/pending)
- Shows the local pairing code for new device linking
- Provides a device code input field for reverse-pairing (entering a code from a remote device)
- File browser for shared storage contents

### Subsystem Scale

The entire device communication subsystem was built this week — 2,611 lines of new code across 8 files. This is a new capability area for G.R.I.M that didn't exist before April 5.

### Open Questions for Discussion

- What's the right trust model for device pairing? Currently code-based; should we consider certificate pinning?
- Should shared storage support conflict resolution (simultaneous edits from multiple devices)?
- How does this integrate with future GRIM-text capabilities (e.g., the model summarizing shared content)?

---

## Feature 5: Checkpoint Reliability & Serialization

### Problem

GRIM-text training runs on Purdue's Anvil cluster, where jobs can run for hours or days. Model checkpoints are saved periodically (every N batches) using FlatBuffers serialization. If a checkpoint is silently corrupted — bad tensor data, incomplete write, buffer underflow — the problem isn't discovered until someone tries to resume training, potentially days later. All compute since the last good checkpoint is wasted.

We encountered exactly this scenario and invested significant effort this week in making checkpoint save/load robust.

### What Was Built

**Post-Write Verification**

After writing a checkpoint file, the system now:

1. Re-opens the file and reads it back
2. Verifies FlatBuffer structural integrity (magic bytes, table offsets, vector lengths)
3. Computes per-tensor statistics (min, max, checksum) and compares against the in-memory tensors
4. Scans for zero-filled regions that shouldn't be zero (catches buffer underflows or incomplete writes)
5. If any check fails, the save is reported as corrupt *immediately* — before the training job continues

**CUDA State Dumps on Failure**

When serialization fails, the system now dumps comprehensive CUDA device state:

- GPU memory usage (allocated vs. total)
- Active CUDA stream status
- Last CUDA error flag
- Device properties and driver version

This is critical for remote debugging — when a job fails on Anvil at 3 AM, the log file now contains enough information to diagnose whether the failure was OOM, device error, or filesystem issue.

**FlatBuffers Version Compatibility**

Updated the version compatibility check to handle FlatBuffers library version differences between development (macOS, likely newer headers) and cluster (Linux, potentially older). Previously, a minor FlatBuffers version mismatch could prevent loading a valid checkpoint.

### Impact

These changes turn checkpoint serialization from a "hope it works" operation into a verified, forensic-quality pipeline. The cost is a few extra seconds per checkpoint (for the re-read and verification), which is negligible compared to minutes or hours of training between checkpoints.

---

## Feature 6: Diagnostic Inference Sampling

### What It Does

During training, the model periodically generates sample text from a prompt. This runs at a configurable interval (e.g., every 100 optimizer steps) and logs the generated text to the training output.

### Why It Matters

Loss curves tell you *how wrong* the model is, but not *what kind of wrong*. Diagnostic inference lets you see:

- **Mode collapse**: If the model generates the same token repeatedly (e.g., all spaces), you see it immediately as "                    " in the log
- **Repetition loops**: If the model gets stuck in cycles ("the the the the"), it's visible before the loss curve shows anything abnormal
- **Qualitative progress**: You can watch the model go from generating random bytes → recognizable words → coherent phrases across training

### Safety Design

The inference sampling system operates on **detached copies** of model weights. This is a critical design constraint — the model's `requires_grad` flags, gradient buffers, and optimizer state must never be touched by the inference path. The implementation creates ephemeral `Tensor::detach()` views so inference runs in read-only mode with respect to the training graph.

---

## Weekly Output Summary

| Metric | Value |
|--------|-------|
| Total commits | 45 |
| Files changed | 302 |
| Lines added | ~20,200 |
| Lines removed | ~4,000 |
| Net new code | ~16,200 lines |
| New subsystems | 2 (Device Comm, Telemetry CSV) |
| New analysis tools | 3 Python scripts |
| Days active | 6 (Apr 3–9) |

### Daily Breakdown

| Day | Commits | Primary Focus |
|-----|---------|---------------|
| Thu Apr 3 | 10 | Tokenizer pruning refinement, diagnostic inference sampling, 3D popup renderer, macOS popup UI |
| Fri Apr 4 | 8 | macOS platform port (ProcessManager, TTS, input handling), execution block cleanup |
| Sat Apr 5 | 15 | Device communication subsystem (full build-out), serialization hardening, shared storage |
| Sun Apr 6 | 3 | 3D renderer textures/shaders, serialization post-write verification |
| Mon Apr 7 | 2 | Tokenizer mining refactor, autograd loss validation |
| Wed Apr 9 | 7 | Telemetry CSV pipeline, warmup causation analysis, vocabulary score multiplier |

---

## Agenda Items for Discussion

1. **Warmup schedule experiments** — The causation analysis predicts the v-estimate half-life (~693 steps) drives early instability. Should we test longer warmup (2,000 steps), exponential warmup, or β₂ annealing on the next Anvil run?
2. **Telemetry-enabled training run** — Ready to launch a full run with per-step CSV telemetry. Discuss priorities: which metrics to focus on, how many epochs, and whether to try alternative optimizers (Lion, Sophia) in parallel
3. **Tokenizer vocabulary size** — Is 10,000 tokens optimal for our corpus size? Literature suggests 32k–64k for larger models, but we're constrained by embedding table memory at 768 dims
4. **Device communication next steps** — The subsystem is functional; discuss what the demo target looks like and whether we need end-to-end testing with actual phone/tablet clients this semester
5. **Anvil compute budget** — How many GPU-hours remain? Plan allocation between telemetry runs, schedule experiments, and potential hyperparameter sweeps

---

*Report generated from git history: commits `195eb88a1` through `8ee21e18c` (April 3–9, 2026)*
