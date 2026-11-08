# TTS Cache Management

Source-verified November 8, 2026.

## Current source and commands

[voice_plugin::grim_plugin_init](../plugins/voice_plugin/voice_plugin.cpp#L19)
registers `clear_tts_cache` and `reset_tts_cache`. Both are raw-text tools with
no declared parameters or aliases; their handlers ignore supplied arguments.
They require VOICE and FILESYSTEM permissions and request confirmation.
`clear_cache` / `reset_cache` are not the registered tool names.

| Tool | Implemented behavior |
|---|---|
| `clear_tts_cache` | Shut down cache, delete its index, remove immediate `.wav` entries in configured temp/cache directories, reinitialize cache. |
| `reset_tts_cache` | Perform the same wipe, then call host precaching. |

The [handlers](../plugins/voice_plugin/cache_commands.cpp#L12) implement filesystem
work in the plugin. The [host voice service](../core/plugin_api.hpp#L404), added
in API v3.7, owns cache lifecycle, synthesis, and precaching; this is not a core
plugin registration.

Paths come from `hostVoice::paths()` (`temp`, `cache`, `cache_index`), not a
hardcoded drive. This is not a recursive deletion of every file or directory.
Failures return `ERR_CACHE_CLEAR_FAILED` / `ERR_CACHE_RESET_FAILED`; a failed
operation is not transactional and can leave a partial wipe.

## Precaching limits

[Voice::preCacheCommonPhrases](../voice/voice_speak.cpp#L694) attempts 17 phrases
through `synthesize`, including `Yes` but **not** `Yes?`. Failed synthesis does
not make the reset handler's completion acknowledgment proof that all audio was
regenerated. There is no verified regeneration-time guarantee.

The selected provider is CosyVoice
([initTTS](../voice/voice_speak.cpp#L531)); XTTS is retained legacy source.
Cache clearing does not repair a duplicated phrase in reference audio.

## Historical context

The original report attributed a repeated "Yes?" to audio data rather than
duplicate command execution. That recording diagnosis was not re-measured here.
Obsolete registration instructions and XTTS timing estimates have been removed.
