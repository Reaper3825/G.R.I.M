# Training Panel: Current Connections and Legacy Control API

Source-verified November 15, 2026. Distinguish implemented server routes from
actual UI callers. The current panel does not launch/monitor a local training
session through the old training-control API.

## Current source: connected UI

[UITrainingPanel::setView / update / pollServer](../../plugins/model_workspace_plugin/ui_training_panel.cpp#L276)
dispatch the panel tabs. `pollServer` calls
`TokenizerClient(modelWorkspaceServerURL()).isServerRunning()`.
Polling occurs on entering Tokenizer and periodically while that tab is active;
`serverConnected` is tokenizer bridge health, not training progress.

[beginConfigCompile / compileConfigPreset](../../plugins/model_workspace_plugin/training_panel/ui_training_panel_training_tab.cpp#L556)
run the host-only `compile_model_config` utility asynchronously to create
`<model-store>\<model-id>\model.grimcfg` from edited model-schema JSON.
They do not POST a training-start request or launch the trainer.

[Tokenizer tab callers](../../plugins/model_workspace_plugin/training_panel/ui_training_panel_tokenizer_tab.cpp#L21)
invoke `TokenizerClient::runTokenizer` and `encodeText` in background work;
`applyTokenizerResults` consumes their futures on the UI thread.
[TokenizerClient](../../control/tokenizer_client.hpp#L15) actually calls:

| Request | Contract |
|---|---|
| `GET /health` | JSON liveness with service `grim_text_server`. |
| `POST /api/tokenizer/run` | JSON `{}`; validate the loaded tokenizer. |
| `POST /api/tokenizer/encode` | JSON `{"text":"..."}`; encode supplied text. |

These calls target the current public GRIM-text bridge, default
**127.0.0.1:11435**, not the legacy control server.
[Public route handlers](../../resources/models/GRIM-text/GRIM/grim_text_server.cpp#L261)
implement health and forward tokenizer requests to the worker's internal routes.
[Worker route handlers](../../resources/models/GRIM-text/training/train_gpu.cu#L278)
implement tokenizer diagnostics. The public health route establishes bridge
liveness, not successful tokenizer/model execution.

## Retained legacy control server: implemented, not connected here

[setupAPI](../../control/training_control_server.cpp#L521) still registers:

| Route | Implemented server behavior |
|---|---|
| `GET /health` | FlatBuffer `HealthCheckResponse`. |
| `GET /api/status` | FlatBuffer status/stats plus process-state reconciliation. |
| `POST /api/training/start` | Optional FlatBuffer config; calls legacy `ProcessController::startTraining`. |
| `POST /api/training/stop` | Calls legacy process stop; FlatBuffer response. |
| `POST /api/config` | FlatBuffer config update, rejected while Training. |
| `POST /api/server/shutdown` | Stops control server/process management. |
| `POST /api/tokenizer/run` | JSON request; runs a standalone tokenizer executable. |
| `POST /api/tokenizer/encode` | JSON text request; runs a standalone tokenizer executable. |

[StatusFileMonitor::monitorLoop](../../control/training_control_server.cpp#L172)
polls a status FlatBuffer file every 500 ms.
The uncalled host `TrainingServerManager` and `commands_training` adapters were
removed on November 8, 2026, together with the manager's teardown hook. This does
not remove the standalone control server or connect it to `UITrainingPanel`.

The legacy server and current worker share **11436 as a default** but are
different protocols/services; do not launch the legacy server alongside the
default worker or point the JSON tokenizer client at its FlatBuffer health route.
No current TrainingPanel call to the legacy start/stop/config/status routes was
found. Local start/stop/pause/progress UI integration is an explicit gap, not an
implemented feature. The old "every batch → progress bar" and pause instructions
therefore do not describe the current panel.

## Verification boundary

This update checked actual request expressions, their URL owner, route handlers,
and config-compiler call sites. It did not launch a server/trainer, perform SSH
training, certify status-file freshness, or validate the retained control
server's ability to run today's trainer. No runtime/build instructions from the
obsolete local-training tutorial remain.

## Model Workspace provider ownership

`plugins/model_workspace_plugin` owns the `UITrainingPanel` implementation and
instance. Its provider name is **Model Workspace Plugin**, and its stable panel
key remains `training_control`. Home, Knowledge Gaps, Tool Gaps, Model Config,
Tokenizer, Observatory, World Lab, Assets, and Tools remain in this one panel.

Initialization constructs the panel; the base constructor stages registration
under the provider generation. Commit attaches the same instance and publishes
its `ui.panel` resource. Host retirement detaches it before plugin shutdown.
The bootstrap factory and UI Plugin's `ui.training_control` request are removed.
Generic panel operations remain in UI Plugin.

The panel owns its composed view instances. Observatory, World Lab, Assets, and
Tools reuse host UI component implementations through explicit native exports.
Their callbacks are released before unload. Model registry/loading, resource
sampling, configuration persistence, and dataset services remain host services.
`ui/model_workspace_services.*` supplies borrowed services and copied configuration
and server URL data; it also supplies host-owned World Lab goal-access callbacks.
This migration does not transfer model/server/dataset service ownership.

Shutdown destroys composed views and waits for tokenizer-health, tokenizer-run,
encode, and configuration-compile futures before unloading provider code. These
operations are not detached or canceled; an in-flight request/compiler can delay
unload. Unsaved panel-local edits and queues are not preserved across reload.

Native builds require matching host/compiler/CRT/UI ABI and a matching Windows
host import library, as with Console and Settings.

Validation:

- `tests/check_model_workspace_ui.ps1`: syntax-checks all provider sources and the
  host service adapter, without building or running model/training targets.
- `tests/run_ui_input_foundation.ps1 -ModelWorkspace`: loads the real provider DLL
  against a focused test host, tests registration/replacement/unload and eight
  non-tokenizer tabs. Model/server operations are disabled and Observatory's
  native viewport is stubbed. World Lab, Assets, and Tools use real UI components.
- `tests/test_plugin_world_assets.py`: checks the authored Model Workspace room
  and `training_control` resource mapping.
