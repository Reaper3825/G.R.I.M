# Build and Runtime

This guide combines model separation and generator troubleshooting. It does
not add a model DLL/static-library alternative or another build/dependency
system.

## Build and runtime

| Owner | Current definition |
|---|---|
| G.R.I.M host/services | [CMakeLists.txt](../../CMakeLists.txt), [Sources.cmake](../../cmake/Sources.cmake) |
| CUDA trainer/model/tokenizer | [TrainingLoop/CMakeLists.txt](../../resources/models/GRIM-text/training/TrainingLoop/CMakeLists.txt) |
| HTTP bridge | [GRIM/CMakeLists.txt](../../resources/models/GRIM-text/GRIM/CMakeLists.txt) |

The host's `GRIM_GPU_SOURCES` is empty. It still supports GPU-enabled
dependencies/services. The separate TrainingLoop explicitly enables CUDA,
requires its supported toolchain, and builds pinned FlashAttention; it is not
a CPU-fallback model trainer.

[TrainingLoop presets](../../resources/models/GRIM-text/training/TrainingLoop/CMakePresets.json)
and dependency manifest/install directory are separate from
[host presets](../../CMakePresets.json). The trainer source directory is
`resources\models\GRIM-text\training\TrainingLoop`, not `training` itself.

```text
G.R.I.M / MMO process manager
  -> public HTTP bridge
    -> train_gpu --inference worker
      -> Phase 1 config/model/tokenizer handoff
      -> Phase 2 generation and session state
```

- [ProcessManager.cpp](../../MMO/Core/ProcessManager.cpp) manages the bridge.
- [grim_text_server.cpp](../../resources/models/GRIM-text/GRIM/grim_text_server.cpp)
  manages its worker child and forwards requests, without owning model tensors,
  TrainingContext, tokenizer loading, or token generation.
- [train_gpu.cu](../../resources/models/GRIM-text/training/train_gpu.cu) finalizes
  the authoritative config snapshot and owns startup and the inference worker.
- [Phase2_InferenceLoop.cu](../../resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.cu)
  orchestrates generation over that handoff.

Process options are `--public-port`, `--worker-port`, and optional `--worker-exe`.
Default bridge/worker ports are 11435/11436. The bridge locates the GRIM root
through the config file and launches its worker there. These options do not
form a competing model/vocabulary config. Bridge liveness is distinct from
worker/model readiness.

The current bridge shares the host HTTP contract header. Model/training
implementation and durable runtime stay outside the host; do not invent an
embedded model or `/v1/completions` API from old sketches.

## Historical generator issue

The November 5, 2025 investigation reported this NVCC error while model CUDA
translation units were embedded in the host:

```text
nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified
```

It explored generated Ninja/MSBuild PDB flags and temporary patches.
Those experiments, machine-specific target counts, and old compatibility
conclusions are not universal present-day toolchain facts or instructions
to patch installed CMake modules.

For a new failure:

1. Identify the failing target, its actual CMake owner, and cache.
2. Compare its selected preset/toolchain, CUDA/architecture, and dependency root.
3. Inspect the real compiler command/generated rule before identifying a cause.
4. Scope any requested configure/build to that target; do not revive deleted
   patch scripts or blindly downgrade tools from the historical investigation.

Model separation did not remove NVCC from the entire repository.
No host, trainer, tokenizer, server, or GPU target was built or run for this
documentation organization.
