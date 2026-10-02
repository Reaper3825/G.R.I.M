# Final-layer lens capture

This is the first, opt-in token-model lens implementation. It captures the final
encoder output before the real LM head and the actual logits before sampling.
Optional identity control replays `forwardLmHead` with detached parameter/input
views and a separate forward sink. Direct and identity-J readouts share that
replay because both have exactly the same input at the final boundary. No fitted
Jacobian, matrix file loader, intermediate-layer transport, UI, or gap-model lens
is implemented here.

## Ownership

- `LensMetadata.hpp` is the global host contract; `LensMetadata.cpp` resolves
  existing payload annotations and summarizes logits.
- `LensCapture_GPU.cu` owns the capture/replay implementation. The shared forward
  owner calls it after the real head completes, before forward cleanup/loss.
- `ModelForwardRequest::lens_capture` borrows a caller-owned request only for the
  duration of the call. A null pointer disables captures and head replays.
- `ModelForwardOutputs::lens_snapshot` owns an immutable host snapshot. Copying
  its shared pointer before `clear()` preserves it without retaining GPU tensors,
  autograd nodes, or model weights.
- Phase 2 carries a prefill snapshot through `GeneratedSequence` into
  `Phase2TextInferenceResult`. Decode calls are not captured by this entrypoint.
- Loss and encoder owners need no new hooks for this stage: the final encoder
  output is already retained, and realized supervision lives on `BatchPayload`.

## Usage

Pass an optional fifth argument to either `executePhase2TextInference` overload:

```cpp
GRIM::Lenses::LensCaptureRequest lens;
lens.identity = {session_id, checkpoint_fingerprint, config_fingerprint,
                 tokenizer_fingerprint, parameter_revision};
lens.replay_identity_control = true;
auto result = executePhase2TextInference(ctx, tokenizer, supplied_state, generation_hp, &lens);
auto snapshot = result.prefill_lens_snapshot;
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
span trees by parsing generated text. The structured inference overload carries
the canonical projected tree; the plain-string overload has no structured tree.

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

Snapshots store one FP32 hidden row, top-k results, scalar metrics, and shared host
span metadata. Temporary logits use O(V) host memory. Probabilities normalize over
the complete vocabulary at temperature 1, never over just the displayed top-k.
Ties sort by token ID; nonfinite logits are rejected.

Identity control borrows the complete detached encoder rectangle and replays the
complete head with the original payload. This preserves centering/PC1 context and
batch geometry. It adds full-rectangle head scratch and logits, but no parameter
copies or parameter gradients. `max_replay_rows` defaults to 1024 and is checked
before model execution; it bounds rows rather than claiming an exact byte limit.
At 768 dimensions, the persistent captured hidden row costs 3 KiB. Identity J is
implicit and allocates no matrix. Calls synchronize deliberately in this first
diagnostic version; this is not an asynchronous streaming telemetry path.

`identity_max_abs_logit_error` compares every vocabulary entry. The consumer
chooses its numerical tolerance. The code does not claim a successful identity
check merely from matching top-k. `JacobianIdentity` explicitly identifies the
control and must not be displayed as a fitted J-lens.

## Validation

### Existing inference diagnostic

Set `training.config.generation_lens_enabled` to `true` in `ai_config.json`
to enable capture in `logDiagnosticSample` at its existing
diagnostic interval. Normal diagnostic sampling settings remain unchanged. Each
sample adds one complete head replay and emits a single-line JSON `[Lens]` record
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
| `generation_lens_max_replay_rows` | `1024` | Full-head replay row limit |

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
`Shared/Lenses/LensMetadata.cpp` and the existing nlohmann JSON include directory.
It covers causal target alignment, inherited span policy, realized masking,
padding rejection, prefill/decode coordinates, retained span lifetime, required
identities, replay row limits, stable full-vocabulary probabilities, ties, and
nonfinite rejection. It needs neither CUDA execution nor model weights.

From an x64 MSVC developer terminal in a separate scratch output directory
(adjust the repository path if needed):

```bat
cl /nologo /std:c++17 /EHsc /W4 /I"D:\G.R.I.M\resources\models\llama.cpp\vendor" "D:\G.R.I.M\resources\models\GRIM-text\Tests\lens_metadata_test.cpp" "D:\G.R.I.M\resources\models\GRIM-text\Shared\Lenses\LensMetadata.cpp" /Fe:lens_metadata_test.exe
lens_metadata_test.exe
```

GPU identity parity and training-gradient noninterference still require an
explicitly authorized runtime test. No trainer/model runtime build is required
to run the metadata test.
