# Workspace Agent Instructions

## Build policy

- Do not configure, compile, link, or run CMake/build targets unless the current task explicitly involves producing or modifying `grim.exe`.
- Python scripts, data-generation scripts, migration scripts, linters, static diagnostics, and focused script/unit tests are allowed and encouraged when they validate the current task.
- Do not run `grim.exe`, GRIM-text trainer binaries, the training loop, model servers, or other compiled runtime/test executables unless the user explicitly requests that execution.
- Do not invoke CMake, MSBuild, Ninja, Make, compiler frontends, or package-install/configure steps for tasks outside `grim.exe`. If such a task appears to require a build, stop and ask the user first.
