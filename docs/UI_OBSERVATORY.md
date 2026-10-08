# Native Observatory

Open **Model + Training Hub → Observatory**. Select a model from the **Model
store** menu. The menu uses `paths.grim_text.model_store` from the runtime AI
configuration, resolving relative paths against the GRIM root just like Model
Config. It lists model directories containing `model.grimcfg`, and selecting one
automatically loads that artifact. **Refresh store** rescans the store; **Reload
model** rereads the selected model after recompilation. The list also refreshes
when opening the tab. An empty store prompts you to compile a model in Model
Config. The existing config reader verifies the artifact;
its architecture supplies the layer count, width, attention head counts, and
sequence limit. Loading another model clears the previous capture. File reads
run in an owned asynchronous job; results are installed on the UI thread.

Choose **Synthetic preview** to exercise the controls without running a model.
This uses the configured number of layers with an explicitly labeled six-token
toy vocabulary and fabricated probabilities. It is not a Jacobian calculation
or a capture from the checkpoint. Alternatively enter a saved JSON capture path
and choose **Load capture**. Capture mode and attention visibility are recorded
metadata, not switches that reinterpret an existing report.

## Views and controls

- 3D coordinates: X = candidate probability `[0,1]`; Y = full-vocabulary entropy
  `[0,ln(V)]` in nats; Z = encoder layer. Coordinates are scaled into a common
  display cube without altering the inspector's recorded numbers.
- Paths connect the same vocabulary ID only at consecutive captured layers.
  Gaps remain gaps. Paths are prediction trajectories, not attention edges or
  a graph of architectural connections.
- Left drag orbits, right/middle drag pans, wheel zooms, and clicking a point
  selects its layer and candidate. Reset camera restores the overview.
- Token position and candidate menus allow exact selection, including points
  which overlap. Candidate menu percentages retain their full-vocabulary
  denominator; the inspector reports the total shown probability mass.
- The layer slider covers the model's entire layer count. Missing readouts are
  labeled. **Play depth** advances through captured layers every 650 ms.
- **Flat view** shows the selected layer's candidates on the same probability /
  entropy axes. All candidates at one layer share that layer's entropy.
- Changing tabs, minimizing, or hiding the hub hides the native window and
  pauses playback. Open menus temporarily show the flat view to keep native
  input from intercepting menu clicks.

The inspector displays capture mode, attention visibility, checkpoint identity,
input token, layer, full-vocabulary entropy, and candidate IDs/probabilities.
The loaded `.grimcfg` summary remains above both views.

## Saved capture schema v1

The panel is a consumer. A model-side direct-capture producer is **not connected
yet**. It must apply the actual model readout (including required normalization
and LM-head transformations), compute entropy across the full vocabulary, and
write the report below. The same schema accepts inference captures and
post-training replays of training examples. This UI does not start training,
inference, or a server, and does not add capture work to the training loop.

```json
{
  "schema_version": 1,
  "config_sha256": "<64 lowercase hex characters: grimcfg semantic_sha256>",
  "checkpoint": "<checkpoint identifier or digest>",
  "layer_count": 7,
  "d_model": 128,
  "vocab_size": 100,
  "mode": "inference",
  "attention_visibility": "causal",
  "positions": [
    {
      "position": 3,
      "input_text": "France",
      "layers": [
        {
          "layer": 6,
          "entropy_nats": 2.0,
          "candidates": [
            {"token_id": 2, "text": "is", "probability": 0.4}
          ]
        }
      ]
    }
  ]
}
```

The geometry above is illustrative, not a default. `layer_count`, `d_model`, and
`config_sha256` must match the selected compiled config. The checkpoint identity
is displayed as supplied; the panel does not open or authenticate a checkpoint.
`vocab_size` comes from the capture because the compiled architecture snapshot
does not contain the realized vocabulary size.

Positions and layer indices in the file are zero-based. Layer labels in the UI
are one-based. `mode` is `inference` or `training_example`;
`attention_visibility` is `causal` or `full_sequence`. `synthetic` optionally
marks externally generated previews (default false).

The loader rejects mismatched configuration fingerprints/geometry, repeated
positions/layers/candidate IDs, out-of-range indices, nonfinite or out-of-range
metrics, and candidate probability sums above one (with rounding tolerance).
Reports are capped at 64 MiB, 4096 positions, 64 candidates per readout, and
250,000 candidate entries in total. Missing layers and changing top-k membership
are supported. Entropy must be recorded from the whole distribution; no top-k
entropy estimate or top-k renormalization occurs in the panel.

## Build integration and verification

The desktop CMake target includes the standalone compiled-config reader and
generates its FlatBuffer header. The existing shader pipeline includes the new
colored Observatory vertex/fragment shaders. This does not link the model
runtime. A missing/unsupported shader program leaves an explicit status message
and the flat view remains available.

Host checks with the existing Windows MSVC toolchain:

```powershell
./tests/run_ui_observatory_prerequisites.ps1
./tests/run_ui_observatory_report.ps1 -CheckPanelSyntax
```

These run focused host tests and optionally C++ syntax diagnostics (`/Zs`),
without building or launching GRIM. Native window composition, actual shader
compilation/rendering, device scaling, and interactive input still require an
explicitly authorized desktop build and visual smoke test.
