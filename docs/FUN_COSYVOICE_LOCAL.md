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

The configured GRIM reference is a 20-second clip derived from the existing
voice material. Its exact local Whisper transcript is stored alongside the
reference path in `voice.cosyvoice`, allowing CosyVoice 3 to use its supported
zero-shot cloning path.

The bridge protocol uses newline-delimited JSON on standard input/output.
Diagnostics are written only to standard error so they cannot corrupt protocol
responses.
