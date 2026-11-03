$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/environment-dispatch-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$fixture = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'environment_dispatch_tests.cpp') -Raw
# Compile the actual controller body with a scripted generation boundary. Only
# model forwarding and mailbox transport are substituted; production owner,
# command validation, projection and span validation are compiled unchanged.
$loop = Get-Content -LiteralPath (Join-Path $workspace 'resources/models/GRIM-text/training/Phases/Phase2_EnvironmentLoop.cu') -Raw
# The begin helper has its own focused host test; this fixture scripts its first
# return and compiles the actual controlling loop from the same production file.
$bodies = foreach ($signature in @(
    'Phase2EnvironmentRunResult executePhase2EnvironmentInteraction(',
    'Phase2EnvironmentLoopResult executePhase2EnvironmentLoop(')) {
    $start = $loop.IndexOf($signature)
    if ($start -lt 0) { throw "Environment loop entrypoint not found: $signature" }
    $end = $loop.IndexOf('{', $start)
    $depth = 1
    while ($depth -gt 0 -and ++$end -lt $loop.Length) {
        if ($loop[$end] -eq '{') { $depth++ }
        elseif ($loop[$end] -eq '}') { $depth-- }
    }
    if ($depth -ne 0) { throw 'Environment loop entrypoint is incomplete.' }
    $loop.Substring($start, $end - $start + 1)
}
$implementation = "namespace GRIMText::Training {`n" + ($bodies -join "`n") + "`n}"
$generated = Join-Path $output 'environment_dispatch_tests.cpp'
Set-Content -LiteralPath $generated -Value $fixture.Replace('// PRODUCTION_DISPATCH', $implementation) -Encoding utf8
$arguments = @('/nologo','/std:c++20','/EHsc','/W3','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$arguments += '"' + $generated + '"'
$arguments += @('resources/models/GRIM-text/Shared/Environment/EnvironmentSession.cpp','resources/models/GRIM-text/Shared/Environment/EnvironmentInput.cpp','resources/models/GRIM-text/Shared/Environment/EnvironmentBoundary.cpp','ui/world_lab/world_document.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:environment_dispatch_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp','if errorlevel 1 exit /b %errorlevel%','environment_dispatch_tests.exe') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Environment dispatch tests failed.' }
} finally { Pop-Location }
