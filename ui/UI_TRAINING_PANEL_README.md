# GRIM model and configuration hub

`UITrainingPanel` is the model-management panel for GRIM-text. It follows the
same core-plus-tab layout as DataHub.

## Tabs

- **Home** — browse registered models, inspect residency/resource usage, and
  register model metadata.
- **Knowledge Gaps** — review router misses and create model entries from them.
- **Tool Gaps** — review proposed missing tools.
- **Model Config** — edit an in-memory model configuration snapshot and compile
  it into an immutable `.grimcfg` artifact.
- **Tokenizer** — validate the tokenizer and inspect text encoding through the
  tokenizer control service.

## Model Config workflow

The Model Config tab replaces the former local training dashboard. GRIM does
not start, stop, pause, resume, or monitor a local training process. Training is
expected to run remotely through the HPC/SSH workflow.

The tab:

1. Uses the runtime model values as in-memory defaults for the authored inputs
   to `grim_compiled_hyperparameters.fbs`.
2. Exposes only model-semantic fields, grouped by FlatBuffer table. Training
   procedure, optimizer, loss, batching, logging, diagnostics, and telemetry
   settings are excluded.
3. Applies edits only to the in-memory preset. It does not mutate
   `ai_config.json`.
4. Selects an existing text-model directory or accepts a new model ID.
5. Invokes the standalone `compile_model_config` executable with the selected
   KTMG v4 `vocab.bin`.
6. Writes the artifact atomically to:

   ```text
   resources/models/model_store/<model-id>/model.grimcfg
   ```

The compiler derives vocabulary geometry, validates cross-field invariants,
and embeds the artifact integrity hashes. A failed compilation leaves an
existing `model.grimcfg` intact.

## Config compiler

From the repository root:

```powershell
cmake -S resources/models/GRIM-text/training/ConfigCompiler `
  -B build/grim-config-standalone `
  -DCMAKE_PREFIX_PATH="resources/models/GRIM-text/training/vcpkg_installed/x64-windows"

cmake --build build/grim-config-standalone `
  --target compile_model_config `
  --config Release
```

The UI searches the standard standalone and TrainingLoop build locations. The
compiler path remains editable when a different build location is used.

## Implementation layout

```text
ui/
  ui_training_panel.hpp
  ui_training_panel.cpp                         shared lifecycle and dispatch
  training_panel/
    ui_training_panel_internal.hpp              shared tab helpers
    ui_training_panel_home_tab.cpp
    ui_training_panel_knowledge_gaps_tab.cpp
    ui_training_panel_tool_gaps_tab.cpp
    ui_training_panel_training_tab.cpp          Model Config implementation
    ui_training_panel_tokenizer_tab.cpp
```

The historical filename `ui_training_panel_training_tab.cpp` is retained for
source continuity, but it no longer contains training execution logic.
