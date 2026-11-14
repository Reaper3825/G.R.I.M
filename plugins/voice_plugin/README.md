# Voice provider — API v10

The DLL owns microphone capture, silence detection, Whisper contexts, transcription
workers, wake-word detection, TTS providers, Python bridge processes, synthesis and
playback, audio device queries, and cached audio. Native implementations live in
`runtime/`; the Coqui and CosyVoice bridge implementations live in `python/`.

## Breaking changes

The host `voice_service`, `GrimVoiceOperation`, `queue_tts`, `stop_tts`,
`is_speaking`, and `GrimVoiceParams` ABI entries are removed. All plugins must be
rebuilt for API v10. No host voice exports or compatibility wrappers remain.

Old voice tool names are removed, including the placeholder `nevermind` and
`create_embedding` tools. Old cache clear/reset and embedding-listing wrappers
are not retained. Native synthesis, cache and platform-query primitives remain
inside the provider. `voice.stop` actually stops the provider runtime.

There is no startup greeting, model warmup, implicit precaching, automatic
speech of command results, direct popup reaction, or settings side effect.
Keyboard wake remains an application UI action; it no longer captures speech.
Wake detection reports a wake event; it does not capture or dispatch a command.
Streaming reports transcripts; it does not call AI, access session history, or
write memory. Application consumers must explicitly route these outputs.

## Lifecycle and configuration

Loading the DLL only registers tools and an idle resource. It opens no device,
loads no model, starts no subprocess and writes no cache. Configuration is
explicit and is not read automatically from host ai_config.json:

```json
{"root":"D:/G.R.I.M","voice":{"engine":"cosyvoice","speaker":"grim","language":"en"},"whisper":{"whisper_model":"ggml-base.en.bin","language":"en"}}
```

Call `voice.configure` with an existing absolute root directory and optional
`voice`, `whisper`, and `wake` objects. Voice/Whisper fields retain the primitive
configuration names; CosyVoice paths and reference audio/text are nested under
`voice.cosyvoice`. Paths default relative to the explicitly configured root.
Coqui's Python bridge runs with that root as its working directory.
Reconfiguration stops the previous runtime. It does not persist configuration.

`voice.tts.start` explicitly initializes the selected provider and its cache.
`voice.capture` / `voice.stream.start` explicitly load Whisper and acquire input.
Only one of capture, streaming or wake listening may use the microphone at once.
Optional Porcupine support requires local SDK binaries and explicit
`wake.access_key`, `wake.model`, and `wake.keyword`; no embedded credential remains.

## Tools

All tools accept JSON objects and currently use Debug classification.

| Tool | Behavior |
|---|---|
| `voice.status` | Current status and the latest 64 transcript/wake/error events, each with a monotonic sequence number. |
| `voice.configure` | Explicit configuration described above. |
| `voice.capture` | Capture one bounded utterance and return a transcript event; does not execute text. |
| `voice.stream.start` / `voice.stream.stop` | Start/stop continuous transcription. |
| `voice.tts.start` | Initialize the configured TTS provider. |
| `voice.speak` | Queue `text` for speech. Completion means queued, not finished playback; inspect status. |
| `voice.synthesize` | Synthesize `text` to a file without playback; optional `speaker`, `language`, `rate`, `pitch`. Provider capabilities apply. |
| `voice.play` | Play a WAV at `path`; waits for playback completion. |
| `voice.stop` | Stop all workers, playback and models; keep configuration. |
| `voice.precache` | Explicit synchronous common-phrase generation; waits for the speech queue to stop, then resumes it. |
| `voice.wake.start` / `voice.wake.stop` | Start/stop wake-word detection. |
| `voice.devices` | Enumerate audio devices. This initializes PortAudio. |
| `voice.cache.status` | Read cache statistics. |

The `voice.status` resource is a copied snapshot refreshed by tool calls. Poll
`voice.status` for fresh streaming output; events are bounded and not durable.
No C++ pointers, Whisper contexts, UI objects or memory references cross the ABI.

Precaching has no detached worker. Shutdown joins transcription, wake and speech
workers before releasing models, subprocesses and audio resources. Playback
stops through its owning operation, rather than closing another thread's stream.
Native/model calls can delay shutdown; no hard real-time cancellation guarantee
is made. Loading the plugin alone does not test backend availability.

## Validation

```powershell
cmake -S plugins/voice_plugin -B .codex-build/voice-owned -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=OFF -DGRIM_VOICE_PROVIDER_TESTS=ON
cmake --build .codex-build/voice-owned --config Release --target voice_plugin voice_provider_tests --parallel 4
# Put existing native dependency DLL directories on PATH.
ctest --test-dir .codex-build/voice-owned -C Release --output-on-failure
```

The DLL test covers ABI rejection, idle initialization, configuration without
acquisition, invalid requests, repeated stop/reload, and publication failure. On Windows it also starts/stops/restarts the SAPI
speech queue with a private temporary cache, without queuing any speech.
It does not record audio, play sound, execute inference, or start Python bridges.
Windows is the validated build; Apple and optional Porcupine paths require their
platform/SDK validation. No host build or deployment is part of this migration.
