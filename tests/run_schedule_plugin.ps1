$ErrorActionPreference = 'Stop'
# Native Calendar requires the real UI primitives and their headless test host.
& (Join-Path $PSScriptRoot 'run_ui_input_foundation.ps1')
