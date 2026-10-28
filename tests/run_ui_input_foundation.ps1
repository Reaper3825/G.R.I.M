param([switch]$WorldLab, [switch]$TextWorld, [switch]$Tools)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/ui-input-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$sources = @('tests/ui_input_foundation_tests.cpp','ui/primitives/ui_inputbox.cpp','ui/primitives/ui_textarea.cpp','ui/primitives/ui_button.cpp','ui/primitives/ui_slider.cpp','ui/primitives/ui_scrollbox.cpp','ui/primitives/ui_dropdown.cpp','helpers/widget.cpp','helpers/vector2.cpp','ui/ui_focus_manager.cpp')
$arguments = @('/nologo','/std:c++20','/EHsc','/W3','/wd4251','/DGRIM_BUILD_HOST','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/helpers"'),('/I"' + $workspace + '/core"'),('/I"' + $workspace + '/ui"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$sources += 'ui/primitives/ui_panel.cpp', 'ui/primitives/ui_split_pane.cpp', 'tests/ui_split_pane_tests.cpp'
$sources += 'ui/primitives/ui_canvas.cpp', 'tests/ui_canvas_tests.cpp'
if ($Tools) {
    $sources += 'ui/training_panel/ui_training_tools_view.cpp', 'MMO/Core/ToolRegistry.cpp', 'tests/ui_training_tools_tests.cpp'
    $arguments += '/DGRIM_TOOLS_VIEW_TESTS'
}
if ($WorldLab -or $TextWorld) {
    $sources += 'ui/world_lab/world_actions.cpp', 'ui/world_lab/ui_world_actions.cpp'
    $sources += 'ui/world_lab/world_asset.cpp', 'ui/world_lab/ui_world_assets_view.cpp', 'ui/world_lab/world_document.cpp', 'ui/world_lab/ui_world_canvas.cpp', 'ui/world_lab/ui_world_lab_view.cpp', 'ui/world_lab/ui_world_session.cpp', 'ui/world_lab/textworld_client.cpp', 'tests/world_lab_tests.cpp', 'tests/world_asset_tests.cpp'
    $arguments += '/DGRIM_WORLD_LAB_TESTS'
}
if ($TextWorld) {
    $sources += 'tests/world_lab_client_tests.cpp'
    $arguments += '/DGRIM_TEXTWORLD_INTEGRATION'
}
$arguments += ('/I"' + $workspace + '/external/bgfx.cmake/bgfx/include"')
$arguments += $sources | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:ui_input_foundation_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Host-only test compilation failed.' }
    & (Join-Path $output 'ui_input_foundation_tests.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Host-only tests failed.' }
} finally { Pop-Location }
