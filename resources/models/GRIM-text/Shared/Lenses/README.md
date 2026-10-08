# Layer/position lens capture

Opt-in capture uses the existing shared model forward and the current LM head.
It supports the selected token or every real position in one batch row, at the
final encoder block or every block. Atom-insertion gap models remain unsupported.
The Observatory panel can inspect the model already loaded through the existing
inference lifecycle. There is no separate checkpoint loader or inspection executable.

## Observatory interface

Select the matching model-store entry, enter inference input, set top-k and the
capture scratch budget in MiB, and choose **Run inspection**. The panel sends
`/api/inspect` through the existing HTTP bridge to `/internal/inspect` on the
Phase 1-owned inference worker. The worker checks the selected artifact path
and semantic digest against its loaded configuration. It uses the loaded
checkpoint and tokenizer identities, captures all layers and real positions,
and returns before sampling any token. Generation and inspection share a mutex
because they borrow the same model, tokenizer, caches and workspaces.

`executePhase2Inspection` uses the same prefill adapters, upload, cached forward
and cleanup as ordinary generation. Graph connections and dropout remain disabled.
The ordinary generation entrypoints retain their existing behavior.
`LensInspectionReport.hpp` adapts the immutable capture to the existing
Observatory HTTP presentation contract; it does not compute new probabilities.
All readout kinds, raw logits, full-vocabulary probabilities, entropy and final
maximum absolute logit error survive transport. The panel can switch between
direct, actual final logits and identity control; unavailable layer/kind pairs
are displayed as unavailable. Presentation snapshots stay immutable while the
native viewport reads them.

Position validation follows the selected `.grimcfg` maximum sequence length;
top-k follows the actual vocabulary. The former fixed position/candidate/entry
limits are removed. Host files and responses retain a 512 MiB size guard, which
is independent of the capture scratch budget.

This connects inference inspection. Training-example replay still requires the
prepared-window reference and payload orchestration; the inference endpoint
rejects training replay instead of reinterpreting it as prompt text. Compact
`.grimlens` save/load remains separate from this presentation adapter.

Host-only validation (no runtime build or execution):

```powershell
./tests/run_ui_observatory_report.ps1 -CheckPanelSyntax -CheckRuntimeSyntax
```

## Ownership

- `LensMetadata.hpp` is the global host contract; `LensMetadata.cpp` resolves
  existing payload annotations and summarizes logits.
- `LensCapture_GPU.cu` owns one capture session per shared forward. Both encoder
  loops call it before intermediate output storage can be reused. Final-layer
  capture runs after the ordinary head, while final hidden states remain live.
- `ModelForwardRequest::lens_capture` borrows a caller-owned request only for the
  duration of the call. A null pointer disables captures and head replays.
- `ModelForwardOutputs::lens_capture_result` owns the immutable host collection,
  ordered by layer then position. `lens_snapshot` aliases its final entry for
  existing single-token consumers. Both survive `clear()` when retained by callers;
  neither retains GPU tensors, autograd nodes, or model weights.
- Phase 2 carries `prefill_lens_capture_result` and the compatible snapshot through
  `GeneratedSequence` into `Phase2TextInferenceResult`. Decode calls are not
  captured by this entrypoint. Realized supervision comes from `BatchPayload`.

## Usage

Pass an optional fifth argument to `executePhase2PayloadInference`:

```cpp
GRIM::Lenses::LensCaptureRequest lens;
lens.identity = {session_id, checkpoint_fingerprint, config_fingerprint,
                 tokenizer_fingerprint, parameter_revision};
lens.all_layers = true;
lens.all_positions = true; // seq_lengths[batch_row], excluding padding
lens.temporary_memory_budget_bytes = 64ULL * 1024 * 1024;
lens.replay_identity_control = true;
// prefill owns the supplied token IDs, GoalTokenSpan tree, and atom side channels.
auto result = executePhase2PayloadInference(ctx, tokenizer, prefill, generation_hp, &lens);
auto captures = result.prefill_lens_capture_result;
```

Identities are authored by the orchestration owner, must be nonempty, and must
describe the loaded model. The caller must keep parameters stable throughout the
forward/replay, just as for ordinary inference. Defaults select the final token
of the named `prompt` span, final encoder block, and top 10 vocabulary entries.
Set `prompt_span_name` for another configured prompt name, and
`prompt_span_entry_index` to disambiguate repeated entries. Lower `top_k` for a
vocabulary smaller than 10. No new architectural model configuration is required.
The text result returns the first generated sequence's snapshot, matching its
existing first-sequence text behavior.

Training owners can supply the same request directly on `ModelForwardRequest`.
The default position comes from the selected span, not the pinned-prefix end or
sequence length. Rows without that span require an explicit local token position.
Inference has no
ground-truth target: target/supervision optionals remain empty. Span annotations
are only exposed when the existing payload carries them; the lens does not infer
span trees by parsing generated text. The payload inference entrypoint preserves the caller-authored tree. The
compatibility wire adapter uses shared span encoding without constructing a
training ConceptBlock; the plain-string adapter has no structured tree.

## Coordinates and supervision

Layer indices are zero-based post-block outputs. Token positions are row-local;
absolute positions include the KV prefix for cached calls. For a training window,
the sequence coordinate system is that realized window, not the unsliced source
document. Sequence IDs are optional when the payload does not supply them.

Span entry indices reference the existing immutable row tree. Resolved configured
policies are stored separately from realized LM targets. A context/prompt token
may predict the first supervised output token. Target span membership describes
the next input position only for an active LM target. Atom auxiliary masks can
suppress LM targets even inside a supervised span. Missing prefix metadata is
represented as unknown rather than inferred from the LM mask.

Attention ranges are half-open intervals of permitted real keys, not weights.
The current token paths have one contiguous range shared across layers/heads.
Full-attention visibility follows the effective causal config plus its ablation;
cached attention is always causal, matching EncoderSelfAttention_GPU.cu.
Padding is excluded. Future windowed/sparse attention requires extending this
contract in its attention owner before using this capture path.

Forward IDs and capture sequences are process-local monotonically increasing
counters, qualified by the caller's session ID. A failed capture may consume an
ID. Timestamps are taken immediately before the host copies; snapshots are only
published after stream completion. Monotonic time is not comparable across hosts
or restarts.

## Memory and readout

Snapshots store top-k results, scalar metrics, and shared host span metadata.
Set `retain_readout_input=true` to also retain the FP32 hidden rows; the compact
default does not. Probabilities and entropy normalize over the complete vocabulary
at temperature 1. Ties sort by token ID; nonfinite logits are rejected.

`forwardLmHead` retains its complete payload rectangle and prompt-boundary checks,
including sequence-level mean pooling. It and `forwardLmHeadReadoutChunk` share
one implementation of normalization, optional residual MLP, token-type gating,
projection, and bias. Batch and sequence geometry come from `BatchPayload`.
Capture passes that payload, the full hidden rectangle, detached parameters,
and a selected batch row/position range. The chunk entry validates that range
against real payload positions and creates its detached view. Tensor shapes are
checked against the payload/selection, not used to author batch geometry.
Sequence pooling stays in the ordinary payload-aware forward wrapper.
There are no centering or PC1 paths.

`temporary_memory_budget_bytes` bounds capture-owned device readout buffers plus
host vocabulary reduction scratch, excluding model/ordinary-forward allocations,
existing cuBLAS workspace, allocator overhead, metadata, and persistent results.
The conservative plan includes simultaneous projection/bias outputs, normalization,
MLP intermediates, host logits/index arrays, and one shared gated weight matrix
when gating is enabled. A budget that cannot fit one row fails before encoder
execution. `max_replay_rows` is now an additional chunk-size cap, not a limit on
the full padded payload. Each chunk completes before its scratch is released for
the next chunk; no layer-specific vocabulary matrices are retained.

Intermediate layers always produce direct readouts. Expanded capture also replays
the final layer and compares it to actual logits. Single-token final-only capture
replays only when `replay_identity_control=true`. Identity-J is an implicit control
at the final layer and allocates no matrix. Calls synchronize deliberately.
The published collection records chunk size, budget, and planned temporary bytes.
Full requested layer/position coverage is checked before publication.

`identity_max_abs_logit_error` compares every vocabulary entry. The consumer
chooses its numerical tolerance. The code does not claim a successful identity
check merely from matching top-k. `JacobianIdentity` explicitly identifies the
control and must not be displayed as a fitted J-lens.

## Validation

### Existing inference diagnostic

Set `training.config.generation_lens_enabled` to `true` in `ai_config.json`
to enable capture in `logDiagnosticSample` at its existing
diagnostic interval. Normal diagnostic sampling settings remain unchanged. Each
sample adds a bounded head replay and emits a single-line JSON `[Lens]` record
containing decoded top tokens, all three readouts, visibility, available span and
target annotations, provenance, and the full-vocabulary maximum absolute logit
error. `jacobian_identity_control` is explicitly a control, not a fitted lens.

Set `training.config.generation_lens_validate` to `true` for the stronger test
(it implies lens capture). These settings flow through `GenerationHP::lens_*`;
the earlier `GRIM_SAMPLE_LENS*` environment switches are no longer read.
At that interval the diagnostic first generates a capture-disabled baseline,
then performs repeated captured generations with the same settings and weights.
This mode forces greedy decoding and a short continuation. `[LensValidation]`
reports exact token-ID and decoded-text agreement plus identity parity. The
snapshot is read after the inference entrypoint returns and its forward cleanup
has run, exercising its host lifetime. GPU memory observations are collected
after warming both paths and after repeated result cleanup. They are device-wide
used-byte readings, not a leak verdict or peak-workspace measurement. Persistent
session caches and unrelated device activity may affect them.

| training.config field | Default | Meaning |
| --- | --- | --- |
| `generation_lens_enabled` | `false` | Routine capture and reporting |
| `generation_lens_validate` | `false` | Paired/repeated validation mode |
| `generation_lens_top_k` | `10` | Display count, capped to vocabulary size |
| `generation_lens_absolute_tolerance` | `0.0001` | Maximum absolute identity-logit error to pass |
| `generation_lens_validation_rounds` | `3` | Captured validation runs, range 2–20, plus one baseline |
| `generation_lens_validation_tokens` | `16` | Validation continuation limit, range 1–256 |
| `generation_lens_max_replay_rows` | `1024` | Maximum rows per readout chunk |

The normal `GRIM_SAMPLE_TOKENS` and `GRIM_SAMPLE_MAX_CHARS` must remain positive
for the diagnostic to run. New controls are strictly parsed. Lens reports use
the training logger's session ID plus optimizer step to identify live weights,
the compiled configuration's semantic SHA-256, and the existing vocabulary-file
hash routine. The live-weight identity is provenance, not a digest of parameter
bytes or a claim that the current weights equal a saved checkpoint. Unsupported
gap models and missing captures are logged through the existing diagnostic error
path. A numerical mismatch emits `fail` without changing training state.

Inference payloads do not contain supervised targets, so those fields are null.
The diagnostic now calls the structured overload and selects the prompt name from
the configured root definition with source path `/prompt`. Its capture can precede
the end of the full prefix when goal/state spans follow the user prompt. Span
metadata is explicitly marked unavailable for plain-string payloads. Validation
compares generated IDs/text, not capture-disabled full logits (which are not
exported by that baseline). Identity parity compares the captured original logits
against the head replay across the full vocabulary. Parameter immutability is
not independently checksummed by this diagnostic.

### Host-only tests

`Tests/lens_metadata_test.cpp` is a standalone host-only test with
`Shared/Lenses/LensMetadata.cpp`, `training/Diagnostics/DiagnosticLens.cpp`, and the existing nlohmann JSON include directory.
It covers causal target alignment, inherited span policy, realized masking,
padding rejection, prefill/decode coordinates, retained span lifetime, required
identities, replay row limits, stable full-vocabulary probabilities, ties, and
nonfinite rejection. It needs neither CUDA execution nor model weights.

From an x64 MSVC developer terminal in a separate scratch output directory
(adjust the repository path if needed):

```bat
cl /nologo /std:c++17 /EHsc /W4 /I"D:\G.R.I.M\resources\models\llama.cpp\vendor" "D:\G.R.I.M\resources\models\GRIM-text\Tests\lens_metadata_test.cpp" "D:\G.R.I.M\resources\models\GRIM-text\Shared\Lenses\LensMetadata.cpp" "D:\G.R.I.M\resources\models\GRIM-text\training\Diagnostics\DiagnosticLens.cpp" /Fe:lens_metadata_test.exe
lens_metadata_test.exe
```

GPU identity parity and training-gradient noninterference still require an
explicitly authorized runtime test. No trainer/model runtime build is required
to run the metadata test.
