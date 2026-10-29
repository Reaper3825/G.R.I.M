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
- **Tokenizer** — inspect the loaded model's tokenizer through `grim_text_server`.
- **Observatory** — inspect the loaded model through the same public bridge.
- **World Lab** — author rooms, nested objects, passages, and player starts;
  add doors and challenges, compile/play through the TextWorld worker, and record
  verified solutions. Save/load includes undo/redo and recovery.
  See [World Lab](../docs/WORLD_LAB.md).
- **Assets** — create, edit, save, and insert independent room/object templates.
  See the [feature reference and terminology index](../docs/WORLD_LAB_REFERENCE.md).
- **Tools** — browse registered tools and inspect their metadata without execution.

## Registered tools viewer

Open **Tools** in Training Control to see the current MMO `ToolRegistry` snapshot.
The list shows canonical tool IDs, categories, providers, and registration status.
Search matches IDs, display names, descriptions, categories, providers, aliases,
and capability tags, ignoring case. The count distinguishes matching entries from
the total registry size. Scroll the list and select a row to inspect its details.

The selectable, copyable detail pane shows documentation, usage, parameters,
examples, provider/version, confirmation and permission metadata, context
requirements, aliases, tags, and usage statistics. Missing fields display
**Not documented**; an empty parameter list does not imply that no arguments are
needed. Registration status does not verify that a tool is ready to execute.

The viewer refreshes every two seconds while active and preserves selection by
tool ID. Removed tools clear their stale details. This view displays the current
registry coverage; legacy commands that bypass the registry are not added or
inferred. It does not edit registrations, dispatch commands, or populate World
Lab objects. The World Lab TextWorld Actions catalog remains separate.

Built-in commands (including input controls)
publish handlers and metadata into the MMO registry. They appear alongside plugin
tools, with provider **builtin**. Filesystem commands (`pwd`, `cd`, `ls`, `mkdir`,
and `rm`) are supplied by the separately deployed **Filesystem Plugin**. The
`clean` and `help` commands are supplied by the **Utility Plugin**, retaining
their original responses. `open` and `search` come from the **Application Plugin**
using the host's API v3.1 error catalog and OS opener.
`list_tools`, `tool_info`, and `tool_stats` come from the **Tool Inspection Plugin**
using API v3.2 read-only metadata snapshots. `alias list`, `alias info`, and
`alias refresh` come from the **Aliases Plugin**, using API v3.3 to access the
host's existing application-alias store and refresh service. `system_info` comes
from the **System Information Plugin**, using API v3.4 for fresh host hardware
snapshots. `set_timer` comes from the **Timers Plugin**; `remember`, `recall`, and
`forget` come from the **Memory Plugin**, using API v3.5 host queue/rotation/storage
services. Existing timer-delivery and non-deleting-forget limitations remain
documented in the plugin README. The three `perception_*` observation commands
come from the **Perception Plugin**, using API v3.6 and existing host perception
services. The eleven voice/TTS-cache tools come from the **Voice Plugin** using
API v3.7, with capture, streaming, playback, configuration, and cache lifecycle
remaining in the host. Existing cancellation/embedding placeholders are documented
in the plugin README. Input control remains in the host. The `core_plugin.cpp` filename
contains the host's built-in registration list; it does not mean those commands
require a separately loaded plugin.

Focused host-only validation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -Tools
```

## Tokenizer diagnostics

`core/grim_text_server_api.hpp` defines the shared public and worker endpoint
paths, default ports and public URL, loopback host, service identity, and JSON
content type. The HTTP bridge, worker, server/process managers, startup code,
native generation backend, tokenizer client, and Observatory use this contract. It is separate
from the DLL plugin function table in `core/plugin_api.hpp`.

Load a model through the model loader before validating or encoding text. The
panel uses the server URL from `GRIMTextServerManager` and posts to
`/api/tokenizer/run` and `/api/tokenizer/encode`. The bridge forwards to the
model worker, which owns the loaded vocabulary and tokenizer. There is no
dependency on `training_control_server` or the retired `tokenizer_runner`.

Validation runs six smoke checks for metadata alignment, token ID ranges, and
decoding across empty, ordinary, whitespace, numeric, structural, and Unicode
inputs. It is not the offline tokenizer self-test suite. Encoding displays token
pieces, IDs, timing, and the decoded text; structural boundary tokens can make
the decoded text differ from the original input.

All requests run in background futures with values applied on the UI thread.
The online indicator checks the bridge's `/health` endpoint every two seconds,
with at most one check in flight. Bridge liveness is independent of model
readiness. Tokenizer requests return a busy error rather than waiting behind
generation or inspection. Closing or switching tabs is safe while work runs;
background tasks do not capture the panel.

## Model Config workflow

The Model Config tab replaces the former local training dashboard. GRIM does
not start, stop, pause, resume, or monitor a local training process. Training is
expected to run remotely through the HPC/SSH workflow.

The tab:

1. Loads root `model_config.json` as the in-memory authored input for
   `grim_compiled_hyperparameters.fbs`.
2. Exposes only model-semantic fields, grouped by FlatBuffer table. Training
   procedure, optimizer, loss, batching, logging, diagnostics, and telemetry
   settings are excluded.
3. Applies edits only to the in-memory preset. It does not mutate
   `ai_config.json`.
4. Selects an existing text-model directory or accepts a new model ID.
5. Invokes the standalone `compile_model_config` executable with the authored
   model configuration.
6. Writes the artifact atomically to:

   ```text
   resources/models/model_store/<model-id>/model.grimcfg
   ```

The compiler derives model geometry, validates cross-field invariants, and
embeds the artifact integrity hashes. Vocabulary size remains late-bound by
training or inference startup. A failed compilation leaves an existing
`model.grimcfg` intact.

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
