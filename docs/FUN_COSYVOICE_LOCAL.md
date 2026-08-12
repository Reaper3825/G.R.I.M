# Fun-CosyVoice 3 local provider

GRIM's CosyVoice provider runs as a persistent local Python process. It does
not download code, models, or packages during `grim.exe` startup.

## Provision the local runtime

From the repository root, run the setup script explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_cosyvoice.ps1
```

The script creates these paths, which match the dormant `ai_config.json`
configuration:

- `.venv-cosyvoice/Scripts/python.exe`
- `external/CosyVoice`
- `resources/models/Fun-CosyVoice3-0.5B`
- `resources/models/wetext`

The setup script does not select the provider or run GRIM.

## Provider configuration

The `voice.cosyvoice` object owns local runtime paths and timeouts. Relative
paths resolve from the GRIM repository root. `reference_audio` may be used by
itself for cross-lingual cloning. When `reference_text` contains an exact
transcript of the reference clip, the bridge uses zero-shot cloning instead.

The active configuration uses `cosyvoice` for `voice.engine` and its neural
entries in `voice.rules`. Missing local dependencies are treated as provider
initialization errors and never trigger an automatic download or Coqui
fallback.

Text-normalization FSTs are provisioned under `resources/models/wetext` and
injected into the bridge directly. Normal GRIM startup therefore does not call
ModelScope or depend on a user-profile cache.

The configured GRIM reference is a compact 4.15-second clip derived from the existing
voice material. Its exact local Whisper transcript is stored alongside the
reference path in `voice.cosyvoice`, immediately after CosyVoice 3's required
`<|endofprompt|>` delimiter. This uses the supported zero-shot cloning path
without an unnecessary instruction prefix.

`voice.cosyvoice.fp16` enables half-precision model inference on supported CUDA
hardware. GRIM voice-rate requests are forwarded to CosyVoice and bounded to
the provider's supported `0.5` through `2.0` range.

At bridge startup, the configured reference is registered as an in-memory
zero-shot speaker. Subsequent requests reuse its prompt tokens, speech
features, speech tokens, and embedding without reprocessing the reference WAV.
The downloaded model's `spk2info.pt` is not modified.

The bridge protocol uses newline-delimited JSON on standard input/output.
Diagnostics are written only to standard error so they cannot corrupt protocol
responses.
