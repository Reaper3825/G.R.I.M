param([switch]$WorldLab, [switch]$TextWorld, [switch]$ModelWorkspace, [switch]$DataHub)
$ErrorActionPreference = 'Stop'
if ($ModelWorkspace) { $WorldLab = $true }
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/ui-input-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$sources = @('tests/ui_input_foundation_tests.cpp','ui/primitives/ui_inputbox.cpp','ui/primitives/ui_textarea.cpp','ui/primitives/ui_button.cpp','ui/primitives/ui_slider.cpp','ui/primitives/ui_scrollbox.cpp','ui/primitives/ui_dropdown.cpp','helpers/widget.cpp','helpers/vector2.cpp','ui/ui_focus_manager.cpp')
$arguments = @('/nologo','/std:c++20','/EHsc','/MD','/W3','/wd4251','/DGRIM_BUILD_HOST','/DNOMINMAX','/DGRIM_PROVIDER_REGISTRY_TESTS',('/I"' + $workspace + '"'),('/I"' + $workspace + '/helpers"'),('/I"' + $workspace + '/core"'),('/I"' + $workspace + '/ui"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$sources += 'ui/primitives/ui_panel.cpp', 'ui/primitives/ui_split_pane.cpp', 'tests/ui_split_pane_tests.cpp'
$sources += 'ui/ui_root_panels.cpp', 'ui/ui_panel_service.cpp', 'ui/dynamic_surface_panel.cpp', 'ui/ui_surface_renderer_bridge.cpp', 'MMO/UI/UISurfaceRegistry.cpp', 'MMO/UI/UISurfaceSpec.cpp', 'tests/ui_panel_operations_tests.cpp'
$sources += 'tests/calendar_panel_tests.cpp', 'tests/schedule_plugin_tests.cpp'
$sources += 'tests/settings_panel_tests.cpp', 'tests/settings_host_stubs.cpp', 'ui/primitives/ui_keybind.cpp', 'ui/primitives/ui_toggle.cpp'
$sources += 'tests/console_panel_tests.cpp', 'tests/console_host_stubs.cpp', 'console_history.cpp', 'ui/primitives/ui_layout_box.cpp'
$sources += 'tests/storage_panel_tests.cpp'
$sources += 'ui/primitives/ui_canvas.cpp', 'ui/primitives/ui_node.cpp', 'tests/ui_canvas_tests.cpp'
$sources += @('MMO/Core/Registries/ToolRegistry.cpp', 'MMO/Core/ToolResultJson.cpp', 'MMO/Core/Registries/ProviderRegistry.cpp') # Panel resource publication is a base UI dependency.
if ($ModelWorkspace) {
    $sources += 'tests/model_workspace_panel_tests.cpp', 'tests/model_workspace_host_stubs.cpp', 'MMO/Core/Registries/ModelRegistry.cpp', 'ui/primitives/ui_progress_bar.cpp', 'ui/primitives/ui_label.cpp'
    $arguments += '/DGRIM_MODEL_WORKSPACE_TESTS'
}
if ($DataHub) {
    $generated = Join-Path $output 'generated/grim_concept'
    New-Item -ItemType Directory -Force -Path $generated | Out-Null
    & (Join-Path $workspace 'vcpkg_installed/x64-windows/tools/flatbuffers/flatc.exe') --cpp --scoped-enums -o $generated (Join-Path $workspace 'DataCollection/concept_block.fbs')
    if ($LASTEXITCODE -ne 0) { throw 'DataHub fixture schema generation failed.' }
    $arguments += '/DGRIM_DATA_HUB_TESTS', '/utf-8', ('/I"' + $output + '/generated"')
    $sources += 'tests/data_hub_panel_tests.cpp', 'tests/data_hub_host_stubs.cpp', 'ui/data_hub_services.cpp', 'ui/primitives/ui_action_menu.cpp'
    $sources += 'DataCollection/dataset_stream.cpp', 'DataCollection/dataset_catalog.cpp', 'DataCollection/dataset_target.cpp', 'DataCollection/dataset_creation_service.cpp', 'DataCollection/concept_generator.cpp', 'DataCollection/io/dataset_io_json.cpp', 'DataCollection/io/concept_block_io_flatbuffer.cpp'
    if (-not $ModelWorkspace) { $sources += 'MMO/Core/Registries/ModelRegistry.cpp', 'ui/primitives/ui_progress_bar.cpp' }
    if (-not ($WorldLab -or $TextWorld)) { $sources += 'ui/world_lab/world_document.cpp' }
}
if ($WorldLab -or $TextWorld) {
    $sources += 'ui/world_lab/world_actions.cpp', 'ui/world_lab/ui_world_actions.cpp'
    $sources += 'resources/models/GRIM-text/Shared/Environment/EnvironmentInput.cpp'
    $sources += 'ui/world_lab/world_asset.cpp', 'ui/world_lab/ui_world_assets_view.cpp', 'ui/world_lab/world_document.cpp', 'ui/world_lab/ui_world_canvas.cpp', 'ui/world_lab/ui_world_lab_view.cpp', 'ui/world_lab/ui_world_session.cpp', 'ui/world_lab/textworld_client.cpp', 'tests/world_lab_tests.cpp', 'tests/world_asset_tests.cpp'
    $arguments += '/DGRIM_WORLD_LAB_TESTS'
    $sources += 'tests/live_world_ui_tests.cpp', 'ui/primitives/ui_grid.cpp'
}
if ($TextWorld) {
    $sources += 'tests/world_lab_client_tests.cpp'
    $arguments += '/DGRIM_TEXTWORLD_INTEGRATION'
}
$arguments += ('/I"' + $workspace + '/external/bgfx.cmake/bgfx/include"')
$arguments += $sources | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:ui_input_foundation_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
$nativeIncludes = @('/core','/helpers','/ui') | ForEach-Object { '/I"' + $workspace + $_ + '"' }
$nativeArgs = $nativeIncludes + @('/nologo','/std:c++20','/EHsc','/MD','/LD','/DGRIM_BUILD_PLUGIN','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'),('"' + $workspace + '/plugins/schedule_plugin/schedule_plugin.cpp"'),'/Fe:schedule_native_fixture.dll','/link','ui_input_foundation_tests.lib')
Set-Content -LiteralPath (Join-Path $output 'native.rsp') -Value $nativeArgs -Encoding ascii
$consoleArgs = $nativeIncludes + @('/nologo','/std:c++20','/EHsc','/MD','/LD','/DGRIM_BUILD_PLUGIN','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'),('"' + $workspace + '/plugins/console_plugin/console_plugin.cpp"'),('"' + $workspace + '/plugins/console_plugin/console_panel.cpp"'),'/Fe:console_native_fixture.dll','/link','ui_input_foundation_tests.lib')
Set-Content -LiteralPath (Join-Path $output 'console.rsp') -Value $consoleArgs -Encoding ascii
$settingsArgs = $nativeIncludes + @('/nologo','/std:c++20','/EHsc','/MD','/LD','/DGRIM_BUILD_PLUGIN','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'),('"' + $workspace + '/plugins/settings_plugin/settings_plugin.cpp"'),('"' + $workspace + '/plugins/settings_plugin/ui_settings_menu.cpp"'),'/Fe:settings_native_fixture.dll','/link','ui_input_foundation_tests.lib')
Set-Content -LiteralPath (Join-Path $output 'settings.rsp') -Value $settingsArgs -Encoding ascii
$storageArgs = $nativeIncludes + @('/nologo','/std:c++20','/EHsc','/MD','/LD','/utf-8','/DGRIM_BUILD_PLUGIN','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'),('"' + $workspace + '/plugins/storage_plugin/storage_plugin.cpp"'),('"' + $workspace + '/plugins/storage_plugin/ui_storage_panel.cpp"'),'/Fe:storage_native_fixture.dll','/link','ui_input_foundation_tests.lib')
Set-Content -LiteralPath (Join-Path $output 'storage.rsp') -Value $storageArgs -Encoding ascii
if ($DataHub) {
    $dataHubSources = @('plugins/data_hub_plugin/data_hub_plugin.cpp')
    $dataHubSources += Get-ChildItem -LiteralPath (Join-Path $workspace 'plugins/data_hub_plugin/data_hub') -Filter '*.cpp' | ForEach-Object { 'plugins/data_hub_plugin/data_hub/' + $_.Name }
    $dataHubArgs = $nativeIncludes + @('/nologo','/std:c++20','/EHsc','/MD','/LD','/utf-8','/DGRIM_BUILD_PLUGIN','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
    $dataHubArgs += $dataHubSources | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
    $dataHubArgs += @('/Fe:data_hub_native_fixture.dll','/link','ui_input_foundation_tests.lib')
    Set-Content -LiteralPath (Join-Path $output 'datahub.rsp') -Value $dataHubArgs -Encoding ascii
}
if ($ModelWorkspace) {
    $workspaceSources = @('plugins/model_workspace_plugin/model_workspace_plugin.cpp','plugins/model_workspace_plugin/ui_training_panel.cpp')
    $workspaceSources += Get-ChildItem -LiteralPath (Join-Path $workspace 'plugins/model_workspace_plugin/training_panel') -Filter '*.cpp' | ForEach-Object { 'plugins/model_workspace_plugin/training_panel/' + $_.Name }
    $workspaceArgs = $nativeIncludes + @('/nologo','/std:c++20','/EHsc','/MD','/LD','/utf-8','/DGRIM_BUILD_PLUGIN','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
    $workspaceArgs += $workspaceSources | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
    $workspaceArgs += @('/Fe:model_workspace_native_fixture.dll','/link','ui_input_foundation_tests.lib','ws2_32.lib','crypt32.lib')
    Set-Content -LiteralPath (Join-Path $output 'workspace.rsp') -Value $workspaceArgs -Encoding ascii
}
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp','if errorlevel 1 exit /b %errorlevel%','cl @native.rsp','if errorlevel 1 exit /b %errorlevel%','cl @console.rsp','if errorlevel 1 exit /b %errorlevel%','cl @settings.rsp') -Encoding ascii
Add-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('if errorlevel 1 exit /b %errorlevel%', 'cl @storage.rsp') -Encoding ascii
if ($DataHub) { Add-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('if errorlevel 1 exit /b %errorlevel%', 'cl @datahub.rsp') -Encoding ascii }
if ($ModelWorkspace) { Add-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('if errorlevel 1 exit /b %errorlevel%', 'cl @workspace.rsp') -Encoding ascii }
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Host-only test compilation failed.' }
    $workspaceFixture = if ($ModelWorkspace) { Join-Path $output 'model_workspace_native_fixture.dll' } else { '-' }
    $dataHubFixture = if ($DataHub) { Join-Path $output 'data_hub_native_fixture.dll' } else { '-' }
    & (Join-Path $output 'ui_input_foundation_tests.exe') (Join-Path $output 'schedule_native_fixture.dll') (Join-Path $output 'console_native_fixture.dll') (Join-Path $output 'settings_native_fixture.dll') (Join-Path $output 'storage_native_fixture.dll') $dataHubFixture $workspaceFixture
    if ($LASTEXITCODE -ne 0) { throw 'Host-only tests failed.' }
} finally { Pop-Location }
