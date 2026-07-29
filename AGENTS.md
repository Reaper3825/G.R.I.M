# Workspace Agent Instructions

## Build policy

- Do not configure, compile, link, or run build or test targets unless the current task explicitly involves producing or modifying `grim.exe`.
- For every other task, use source inspection and static validation only. Do not invoke CMake, MSBuild, Ninja, Make, compiler frontends, test executables, or configure steps that install packages.
- If a task outside `grim.exe` appears to require a build, stop and ask the user before running one.
