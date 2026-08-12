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

To perform the later cutover, set `voice.engine` and any neural entries in
`voice.rules` to `cosyvoice`. Keep the engine unchanged until provisioning is
complete; missing local dependencies are treated as provider initialization
errors and never trigger an automatic download or Coqui fallback.

The bridge protocol uses newline-delimited JSON on standard input/output.
Diagnostics are written only to standard error so they cannot corrupt protocol
responses.
