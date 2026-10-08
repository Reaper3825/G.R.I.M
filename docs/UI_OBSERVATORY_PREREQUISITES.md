# Observatory UI prerequisites

These primitives support the native Observatory tab, now integrated in the
Model + Training Hub. See [UI_OBSERVATORY.md](UI_OBSERVATORY.md) for panel usage
and the saved-capture contract. Model capture and an inspection endpoint remain
separate work.

## Model-dependent controls

Read the selected model with `GRIM::Config::loadCompiledModelConfig(path)`.
Use `snapshot.architecture.num_layers` for layer bounds and validate the capture
against the configuration identity and geometry. Do not use a fixed layer count
or the currently selected training model when inspecting a different checkpoint.

`UISlider::setRange(min, max)` clamps the current value and cancels an in-progress
edit/drag. Equal bounds support a single-layer model. A step of 1 snaps values
relative to the minimum, including text entry and programmatic updates. Neither
`setRange` nor `setValue` invokes the user-interaction callback.

## Graph coordinates and selection

Existing `DataPoint(value, label)` calls keep indexed X coordinates.
`DataPoint::xy(x, y, label)` supplies explicit coordinates for Line, Scatter,
MultiLine, and Area. Points remain in supplied order, including backtracking
trajectories. In a plot mixing explicit and implicit X, implicit X is its index
on the common numeric axis. Other graph types retain their categorical behavior.

Use `setXAxisRange(0, 1)` for probabilities and `setAxisRange(0, entropyMaximum)`
for entropy in nats. Automatic X scaling includes visible series and expands
single-value ranges. Store and display full-vocabulary entropy from capture;
do not recompute it from top-k or treat all other tokens as one category.

`hitTest` returns the nearest visible point within the hit radius, including its
series index (`-1` for single-series data). `setOnSeriesPointHover` and
`setOnSeriesPointClick` expose this identity. Existing callbacks still receive
the correct selected point. Indices refer to displayed data after downsampling;
keep a stable identity in the data label or disable downsampling for exact
layer-by-layer inspection. Downsampling preserves endpoints when the limit is
at least two. Existing widgets use screen coordinates for drawing and input.

## Cartesian camera

`UIOrbitCamera` is independent of Cesium and the globe-oriented `UICamera`.
It provides Y-up orbit/pan/zoom, bounds fitting, projection, and pick rays using
the same perspective geometry. Supply viewport-local pixel coordinates.
`projectPoint` returns top-left pixel coordinates and depth in `[0,1]`, or no
result for points outside the frustum. Configure the clip convention using
`bgfx::getCaps()->homogeneousDepth` at the renderer boundary.

For the prediction landscape, map candidate probability to world X, entropy to
world Y, and layer to world Z. Retain the original units for axis labels. A
projection is a display transform; it does not alter the recorded values.
Connections between layer samples are guides, not attention edges.

Reuse `UI3DViewport` and `UINative3DViewportAttachment` for the native surface.
The scene owner must register/release its render pass and view IDs, route
input, and hide its viewport when the Observatory tab is inactive or minimized.
Use an owned asynchronous inspection job and publish immutable reports to the
UI thread; do not copy the tokenizer tab's detached-thread mutation pattern.

## Focused verification

On Windows with the existing MSVC toolchain and GLM installation:

```powershell
./tests/run_ui_observatory_prerequisites.ps1
```

This compiles and runs only a host unit-test executable. It links the real graph,
slider, camera, widget, and focus implementations with drawing/input test sinks.
It does not configure or build GRIM, a trainer, or a model server. Actual native
window/bgfx rendering still needs integration testing in an authorized UI build.
