# Workspace Agent Instructions

## Build policy

- Host-only, non-training utilities and data/configuration compilers may be configured, compiled, linked, and run when the current task explicitly requires them. This includes standalone tools such as `compile_model_config` and their schema-generation targets.
- Do not configure, compile, link, or run `grim.exe`, GRIM-text trainer binaries, the training loop, model servers, or other training/runtime targets unless the user explicitly requests that specific build or execution.
- Python scripts, data-generation scripts, migration scripts, linters, static diagnostics, and focused script/unit tests are allowed and encouraged when they validate the current task.
- Keep builds scoped to the explicitly requested target; do not build unrelated training or runtime targets as a side effect.
- Do not perform package installation or dependency mutation unless the user explicitly requests it. Existing toolchains and installed dependencies may be used for permitted builds.
