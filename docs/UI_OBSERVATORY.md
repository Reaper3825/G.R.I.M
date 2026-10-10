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
or a capture from the checkpoint. **Run inspection** saves the complete validated
response to `observatory_capture.json` beside the selected model's `model.grimcfg`.
Selecting a model creates this file if missing, with a `capture_pending` placeholder
until the first successful inspection. **Load capture** uses that file automatically;
there is no manual path field. A pending file prompts you to run an inspection.
Existing files are preserved on selection; successful inspections replace the previous
capture after writing a complete temporary file. Save errors are displayed, while
the live report remains available in memory. Synthetic previews do not overwrite
saved captures. Capture mode and attention visibility are recorded
metadata, not switches that reinterpret an existing report.

Inspection HTTP failures are reported before interpreting the body as capture JSON.
An HTTP 404 indicates missing inspection support in the running bridge or worker;
rebuild/restart those components with the current endpoint code. An unavailable
worker requires loading the model through the existing model loader. Empty or
malformed HTTP 200 responses are identified as server response errors, separately
from saved-file loading errors.

## Views and controls

- 3D coordinates default to X = candidate probability `[0,1]`, Y = shown
  candidate rank (rank 1 at the top), Z = encoder layer. **3D: rank / 3D: entropy**
  toggles Y between rank and full-vocabulary entropy `[0,ln(V)]` in nats.
  Rank spacing uses the maximum displayed candidate count across the selected
  position's layers; changing top-k membership does not stretch individual planes.
  Rank is the order of the recorded probability-sorted candidates, including the
  stored order for ties. Coordinates are scaled without changing recorded values.
- Color-coded X/Y/Z axes have arrowheads, tick marks and camera-facing title
  billboards anchored to their endpoints. X shows probability percentages, Y
  shows candidate rank numbers or entropy in nats, and Z shows one-based encoder
  layer numbers. Axes stay at the first layer as a fixed reference while the
  active plane moves. Tick labels thin out when they overlap on screen; axis
  titles have priority and retain leader lines when repositioned to fit.
- Every encoder layer has a frame; unavailable readouts have dim frames. The
  active layer has a translucent surface, grid, probability ticks and point count.
  Circular points have stable token colors; their size highlights selection and
  does not encode probability or hidden-state dimensions. `d_model` remains model
  metadata; points represent vocabulary candidates, not individual neurons.
- Paths connect the same vocabulary ID only at consecutive captured layers.
  Gaps remain gaps. Paths are prediction trajectories, not attention edges or
  a graph of architectural connections.
- Paths use shape-preserving cubic Hermite splines through the captured points.
  Each segment stays within its endpoint probability and rank/entropy ranges;
  depth advances linearly. Two-point paths remain straight. Curves are visual
  interpolation, not additional measurements. Hovered or pinned token paths
  use a brighter, approximately three-pixel-wide camera-facing strip.
- Left drag orbits, right/middle drag pans, wheel zooms, and clicking a point
  selects its layer and candidate. Reset camera restores the overview.
- Hovering shows a camera-facing billboard anchored to the point, with token
  text/ID, shown rank, probability, entropy and changes from the immediately
  preceding layer. Probability changes are in percentage points; positive rank
  changes mean a move toward rank 1. Missing candidates/readouts produce explicit
  unavailable labels, never zero probabilities or deltas across missing layers.
  Clicking pins the token path. Layer scrubbing and playback preserve that token
  even outside top-k; the candidate menu indicates when it is not shown. Choose
  **Select / unpin** to release it. Hover temporarily highlights another path.
- Token position and candidate menus allow exact selection, including points
  which overlap. Candidate menu percentages retain their full-vocabulary
  denominator; the inspector reports the total shown probability mass.
- The layer slider covers the model's entire layer count. Missing readouts are
  labeled. **Play depth** advances through captured layers every 650 ms.
- **Flat view** retains the selected layer's probability / entropy axes,
  independent of the 3D rank toggle. All candidates at one layer share its entropy.
- Changing tabs, minimizing, or hiding the hub hides the native window and
  pauses playback. Open menus temporarily show the flat view to keep native
  input from intercepting menu clicks.

The inspector displays capture mode, attention visibility, checkpoint identity,
input token, layer, full-vocabulary entropy, and candidate IDs/probabilities.
The loaded `.grimcfg` summary remains above both views.

## Selected hidden-state Jacobian row

Enter a prompt, set **Target pos** (`-1`: last real input position), set
**Output dim** (zero-based; defaults to `0`), and choose **Jacobian row**.
This computes one final post-encoder-block hidden dimension with respect to
every source hidden dimension, token position and encoder layer:

`J[layer, source_position][out, in] = d h_final[target_position, out] / d h_layer[source_position, in]`.

Each source position/layer owns a `1 x d_model` row. The target is a hidden
activation before the LM head. The final-layer row is checked against the
corresponding identity row at the target position and zero at other positions.

The **Hidden Jacobian** view opens automatically. Position and layer controls
select the source row. X is the source hidden dimension; Y labels the selected
output dimension. Blue is negative, orange positive, and color intensity uses
a square-root scale relative to the largest visible absolute derivative.
Overview bins show the signed entry with largest absolute magnitude. Hover
shows its exact coordinates and value. Wheel zoom reaches individual entries;
**Reset matrix** restores all input dimensions. Save/load retains every entry.
Previously saved full matrices still load and display as two-dimensional heatmaps.

Execution performs **one forward/backward pair** on an uncached, dropout-free
activation graph with detached parameters and LoRA views. No full matrix is
allocated or assembled. Ordinary generation and inspection retain their paths;
attention visibility follows the effective model configuration. Execution stays
serialized with generation. For the 12-layer, 768-wide router and five source
positions, retained derivatives occupy **180 KiB**, compared with 135 MiB for
full matrices. This does not imply a measured end-to-end speedup: forward,
readout, serialization and UI overhead remain.

Scratch MiB covers captured derivatives and root-gradient buffers; saved
activations, autograd workspace and serialization require additional memory.
The raw derivative response limit remains 192 MiB, with a 512 MiB JSON/file
limit. Oversized captures fail without truncation.

Request `jacobian` fields are `kind: final_hidden`, `target_position` (default
`-1`) and `output_dimension` (default `0`). Capture schema v1 adds those fields
plus `rows: 1`, `columns: d_model`, `layout: output_by_input`,
`encoding: ieee754_f32_hex`, and `boundary: post_encoder_block`. Each layer
stores `hidden_jacobian_f32_hex`: eight most-significant-first hexadecimal
digits per finite IEEE-754 FP32 entry. Explicit `output_dimension` metadata
identifies a selected row. Full saved matrices omit it and require
`rows == d_model`; incomplete matrices remain invalid. The loader validates
shape, target bounds, complete source coverage and finite values. Compact
`.grimlens` storage does not carry this optional JSON extension.
