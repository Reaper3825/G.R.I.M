# Workspace Agent Instructions

## Build policy

- Host-only, non-training utilities and data/configuration compilers may be configured, compiled, linked, and run when the current task explicitly requires them. This includes standalone tools such as `compile_model_config` and their schema-generation targets.
- Do not configure, compile, link, or run `grim.exe`, GRIM-text trainer binaries, the training loop, model servers, or other training/runtime targets unless the user explicitly requests that specific build or execution.
- Python scripts, data-generation scripts, migration scripts, linters, static diagnostics, and focused script/unit tests are allowed and encouraged when they validate the current task.
- Keep builds scoped to the explicitly requested target; do not build unrelated training or runtime targets as a side effect.
- Do not perform package installation or dependency mutation unless the user explicitly requests it. Existing toolchains and installed dependencies may be used for permitted builds.

## Retired agent tools

- `reload_nlp`, `clear_context`, `grim_ai`, `test_intent`, `test_nlp`, `nlp_stats`, `nlp_learn`, `nlp_save`, `settings`, `question`, and `ai_backend` are retired from agent tool registration. Do not re-register them, migrate them into tool plugins, or advertise them as available agent tools unless the user explicitly requests their reintroduction.
- Preserve their underlying handler logic and host services for potential host/developer use. Retirement of these tool registrations does not authorize deleting conversation handling, model routing, grammar support, or session administration.
- Read [retained-handler documentation](docs/RETIRED_AGENT_TOOLS.md) before changing or reusing these handlers. Keep that document current when their behavior or retention purpose changes.
- Input tools and native device execution are owned by `plugins/input_plugin` through API v11, independently of perception. Do not restore host input forwarding. Perception implementations, devices, workers, models, caches, and observation resources belong exclusively to `plugins/perception_plugin`. Do not restore host perception globals, legacy forwarding services, context-manager connections, or duplicate tool registrations. `click_on` and the old perception command names are retired; preserve the underlying capture/OCR/native-control/detection primitives. Generic panel operations are supplied by `ui_plugin`; fixed UI surface creation/destruction and OSINT remain separate scopes. The retired AI/NLP/settings wrappers are not pending migrations.

- OSINT commands, collectors, jobs, cache and Python bridge execution are owned exclusively by `plugins/osint_plugin`. Host consumers use copied tool results; do not restore host collector implementations or cache-file coupling.

- Voice capture, transcription, wake detection, TTS providers, playback, caches and bridge workers are owned by `plugins/voice_plugin` through API v10. Do not restore host voice exports, forwarding services, automatic conversation dispatch or detached precache threads. Application consumers must use explicit tools or copied status/transcript resources.

## Application alias ownership

- User application aliases, legacy user-alias compatibility reads, and portable/Start Menu discovery belong to `plugins/application_plugin` through API v12. Do not restore host alias caches, scanners, or alias-specific ABI services. Never overwrite the existing alias database during compatibility reads. ToolRegistry command aliases are separate.

## Reasoning data contract

- Before creating, transforming, or reviewing GRIM ConceptBlock reasoning examples, read `docs/GRIM_REASONING_DATA_CONTRACT.md`.
- Changes to `DataCollection/concept_block.fbs`, `DataCollection/concept_block.hpp`, `DataCollection/concept_block_canonical.hpp`, ConceptBlock/Goal span projection, or concept supervision policy must update `docs/GRIM_REASONING_DATA_CONTRACT.md` when they affect the documented contract.

## Removed collapse interventions

- Encoder residual centering, LM-head hidden/weight centering, logit centering, and PC1 projection have been removed. Do not design lens capture or generation around these historical paths. LM-head readout math is row-local; existing forward payload geometry checks still apply.
- Retired FlatBuffer slots and read-only checkpoint compatibility handling are legacy metadata, not supported model features. Historical collapse investigation notes and plans do not describe the current architecture.
- Preserve unrelated normalization, token-type gating, initialization mean subtraction, and softmax/loss gradient math.
