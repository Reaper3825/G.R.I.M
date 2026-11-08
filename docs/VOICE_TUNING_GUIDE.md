# Speech Recognition and Selected TTS Provider

Source-verified November 8, 2026. Whisper recognition and CosyVoice synthesis
are separate paths; TTS engine selection does not tune recognition.

## Current source: recognition

[Voice::captureAndTranscribeSpeech](../voice/voice.cpp#L124) captures mono 16 kHz
audio and reads the following settings:

| Config field | Actual use |
|---|---|
| `voice.input_device_index` | Capture device selection. |
| `voice.silence_threshold` | RMS silence gate; not the duplicated `whisper.silence_threshold` field. |
| `voice.silence_timeout_ms` | End-of-speech timeout. |
| `whisper.min_speech_ms`, `min_silence_ms` | Speech/silence duration checks. |
| `whisper.max_listening_ms` | No-speech listening deadline; not a hard maximum for ongoing speech. |
| `whisper.language`, `temperature`, `max_len`, `suppress_blank`, `no_speech_threshold`, `initial_prompt` | Passed into one-shot decoder parameters. |
| `whisper.beam_size` | Assigned to the beam substructure, but one-shot decoding initializes **GREEDY**, not beam search. |

The one-shot path does **not** select beam search from
`whisper.sampling_strategy`. Entropy/log-probability thresholds are hardcoded
there rather than read from the similarly named config fields.
`max_len = 1` is not a one-word or one-token transcript guarantee.

[VoiceStream](../voice/voice_stream.cpp#L434) also initializes greedy decoding
and uses its own language/token limits. Its ambient-noise calibration is not
proof that one-shot capture automatically calibrates.

## Tuning checklist

Compare captured transcripts with the intended words before diagnosing model
tool selection. Inspect the chosen microphone, RMS gate, capture endpoint, and
speech cutoffs. Change one consumed field at a time and compare recorded cases;
temperature/prompt changes cannot guarantee elimination of hallucinations.
Settings → Audio exposes recognition controls
([createAudioWidgets](../ui/ui_settings_menu.cpp#L611)); exposed labels alone do
not establish decoder strategy or GPU acceleration. The host requests GPU use,
but actual backend execution needs separate runtime evidence.

## Current source: synthesis and command limits

[initTTS](../voice/voice_speak.cpp#L531) selects the provider from `voice.engine`;
the checked-in choice is `cosyvoice`.
[CosyVoiceProvider::capabilities](../voice/cosyvoice_provider.cpp#L75) advertises
voice selection/cloning, language selection, and rate; pitch/emphasis are not
advertised. Local initialization currently requires Windows, model assets,
reference audio, and matching reference text.

[voice_plugin registrations](../plugins/voice_plugin/voice_plugin.cpp#L19)
retain legacy debug commands: `test_tts` calls Coqui, `list_voices` lists Coqui
config or Windows SAPI voices, and `tts_device` queries the SAPI device.
`test_sapi` plays an existing test WAV. These are not CosyVoice readiness tests.
`nevermind` and `create_embedding` remain acknowledgment placeholders, not
implemented playback cancellation/embedding submission.

No speech/model performance table or current runtime certification is claimed.

## Voice Settings UI

This section owns the voice-specific Settings integration previously documented
separately from recognition/provider behavior.

- [createVoiceWidgets](../ui/ui_settings_menu.cpp#L473) creates Voice Engine
  and Speaker cycle buttons, a Voice Mode dropdown, and speed/confidence
  controls. Speaker is not a Coqui-only embedding dropdown.
- [cycleSpeaker](../ui/ui_settings_menu.cpp#L164) uses
  `voice.available_speakers` when present; otherwise it scans embedding names
  and records the resulting list in pending config.
- [scanSpeakerEmbeddings](../settings/settings_apply.cpp#L59) supplies
  `default` plus `.npz` stems from the resource embeddings directory.
  Retaining this scanner does not mean CosyVoice consumes XTTS embeddings.
- [saveRuntimeAiConfig](../settings/runtime_ai_config.cpp#L66) persists edits;
  [applyRuntimeSideEffects](../settings/settings_apply.cpp#L13) calls
  `Voice::setSpeaker` and applies selected host UI/perception settings.
  It does not reinitialize every provider setting.

Open Settings through the [Console](ui/CONSOLE.md), configured shortcut, or
[generic panel service](ui/PANEL_OPERATIONS.md) using the stable `settings`
key. Poll pending panel operations; acceptance is not completion.
The retired `settings` agent wrapper is not the Settings panel.

Save & Close persists the selected controls, but a saved speaker label is not
proof of new voice cloning or immediate provider reconfiguration.
The selected provider uses its configured reference audio/text, speaker,
language, and rate. Retained `create_embedding` and legacy test commands do not
provide a CosyVoice acceptance workflow.

Historical XTTS UI/embedding instructions remain under
[legacy/SPEAKER_UI_SETTINGS.md](legacy/SPEAKER_UI_SETTINGS.md),
[legacy/SPEAKER_EMBEDDINGS.md](legacy/SPEAKER_EMBEDDINGS.md), and
[legacy/EMBEDDINGS_QUICKSTART.md](legacy/EMBEDDINGS_QUICKSTART.md).
Retained multi-command controls do not restore the old host execution workflow.
