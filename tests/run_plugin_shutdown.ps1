param([string]$PluginDirectory = "", [string]$FilesystemPluginDirectory = "", [string]$UtilityPluginDirectory = "", [string]$ApplicationPluginDirectory = "", [string]$ToolInspectionPluginDirectory = "", [string]$SystemInfoPluginDirectory = "", [string]$TimersPluginDirectory = "", [string]$MemoryPluginDirectory = "", [string]$PerceptionPluginDirectory = "", [string]$VoicePluginDirectory = "", [string]$UIPluginDirectory = "", [string]$InputPluginDirectory = "", [switch]$BuildUIFixture, [switch]$BuildFilesystemFixture, [switch]$BuildApplicationFixture, [switch]$BuildSystemInfoFixture)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/plugin-shutdown-tests'
$fixtures = Join-Path $output 'fixtures'
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $output 'rejected') | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$memoryGenerated = Join-Path $output 'generated/grim_memory'
New-Item -ItemType Directory -Force -Path $memoryGenerated | Out-Null
Copy-Item -LiteralPath (Join-Path $workspace 'memory/unified_memory_generated.h') -Destination (Join-Path $memoryGenerated 'unified_memory_generated.h')
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$common = @('/nologo','/std:c++20','/EHsc','/W3','/DNOMINMAX')
$common += @('', '/helpers','/core','/commands','/vcpkg_installed/x64-windows/include') | ForEach-Object { '/I"' + $workspace + $_ + '"' }
if ($BuildFilesystemFixture) {
    $FilesystemPluginDirectory = Join-Path $output 'filesystem-fixture'
    New-Item -ItemType Directory -Force -Path $FilesystemPluginDirectory | Out-Null
    $filesystemArgs = $common + @('/LD', '/DGRIM_BUILD_PLUGIN', '/utf-8', ('"' + (Join-Path $workspace 'plugins/filesystem_plugin/filesystem_plugin.cpp') + '"'), '/Fe:filesystem-fixture/filesystem_plugin.dll')
    Set-Content -LiteralPath (Join-Path $output 'filesystem-fixture.rsp') -Value $filesystemArgs -Encoding ascii
}
if ($BuildUIFixture) {
    $UIPluginDirectory = Join-Path $output 'ui-fixture'
    New-Item -ItemType Directory -Force -Path $UIPluginDirectory | Out-Null
    $uiFixtureArgs = $common + @('/LD', ('"' + (Join-Path $workspace 'plugins/ui_plugin/ui_plugin.cpp') + '"'), '/Fe:ui-fixture/ui_plugin.dll')
    Set-Content -LiteralPath (Join-Path $output 'ui-fixture.rsp') -Value $uiFixtureArgs -Encoding ascii
}
$fixtureSource = '"' + (Join-Path $workspace 'tests/fixtures/plugin_shutdown_fixture.cpp') + '"'
foreach ($fixture in @('alpha', 'beta')) {
    $argsForFixture = $common + @('/LD',('/DFIXTURE_NAME=\"Declared-' + $fixture + '\"'),$fixtureSource,('/Fe:fixtures/' + $fixture + '.dll'))
    Set-Content -LiteralPath (Join-Path $output ($fixture + '.rsp')) -Value $argsForFixture -Encoding ascii
}
foreach ($fixture in @('failed', 'legacy')) {
    $argsForFixture = $common + @('/LD',('/DFIXTURE_NAME=\"Declared-' + $fixture + '\"'),$fixtureSource,('/Fe:rejected/' + $fixture + '.dll'))
    if ($fixture -eq 'failed') { $argsForFixture += '/DFIXTURE_FAIL_INIT' }
    else { $argsForFixture += '/DFIXTURE_API_VERSION=65536' }
    Set-Content -LiteralPath (Join-Path $output ($fixture + '.rsp')) -Value $argsForFixture -Encoding ascii
}
foreach ($folder in @('integration', 'integration-rejected')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $output $folder) | Out-Null
}
foreach ($fixture in @('alpha', 'beta', 'failed', 'legacy')) {
    $folder = if ($fixture -in @('failed', 'legacy')) { 'integration-rejected' } else { 'integration' }
    $argsForFixture = $common + @('/LD','/DFIXTURE_TOOLS',('/DFIXTURE_NAME=\"Declared-' + $fixture + '\"'),$fixtureSource,('/Fe:' + $folder + '/' + $fixture + '.dll'))
    if ($fixture -eq 'failed') { $argsForFixture += '/DFIXTURE_FAIL_INIT' }
    if ($fixture -eq 'legacy') { $argsForFixture += '/DFIXTURE_API_VERSION=393216' }
    Set-Content -LiteralPath (Join-Path $output ('integration-' + $fixture + '.rsp')) -Value $argsForFixture -Encoding ascii
}
$variants = @{
    old = @('/DFIXTURE_EXTRA', '/DFIXTURE_RESULT=\"old-result\"');
    new = @('/DFIXTURE_RESULT=\"new-result\"');
    failed = @('/DFIXTURE_FAIL_INIT', '/DFIXTURE_RESULT=\"failed-result\"');
    collision = @('/DFIXTURE_COLLISION');
    legacy = @('/DFIXTURE_API_VERSION=393216')
}
foreach ($variant in $variants.Keys) {
    New-Item -ItemType Directory -Force -Path (Join-Path $output ('replacement/' + $variant)) | Out-Null
    $variantArgs = $common + @('/LD','/DFIXTURE_TOOLS','/DFIXTURE_NAME=\"Declared-alpha\"',$fixtureSource,('/Fe:replacement/' + $variant + '/alpha.dll')) + $variants[$variant]
    Set-Content -LiteralPath (Join-Path $output ('replacement-' + $variant + '.rsp')) -Value $variantArgs -Encoding ascii
}
$testArgs = $common + '/DGRIM_BUILD_HOST'
$testArgs += @('tests/plugin_shutdown_tests.cpp','core/plugin_manager.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$testArgs += '/Fe:plugin_shutdown_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'tests.rsp') -Value $testArgs -Encoding ascii
$apiArgs = $common + '/DGRIM_BUILD_HOST'
$apiArgs += '/I"' + (Join-Path $output 'generated') + '"'
$apiArgs += @('tests/plugin_api_ownership_tests.cpp','tests/plugin_reconcile_tests.cpp','tests/filesystem_plugin_tests.cpp','tests/utility_plugin_tests.cpp','tests/application_plugin_tests.cpp','tests/tool_inspection_plugin_tests.cpp','tests/host_alias_service_tests.cpp','tests/system_info_plugin_tests.cpp','ui/world_lab/world_document.cpp','tests/runtime_plugin_tests.cpp','tests/perception_plugin_tests.cpp','tests/voice_plugin_tests.cpp','tests/ui_plugin_tests.cpp','tests/input_plugin_tests.cpp','core/plugin_api_impl.cpp','core/plugin_manager.cpp','MMO/Core/Registries/ToolRegistry.cpp','MMO/Core/Registries/ProviderRegistry.cpp','MMO/Core/SessionContextManager.cpp','console_history.cpp','commands/command_dispatch.cpp','commands/commands_timers.cpp','memory/command_memory_services.cpp','memory/unified_memory_storage.cpp','memory/memory_buffer_rotation.cpp','voice/voice_command_services.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$apiArgs += @('/Fe:plugin_api_ownership_tests.exe', '/link', 'psapi.lib', ('"' + (Join-Path $workspace 'vcpkg_installed/x64-windows/lib/flatbuffers.lib') + '"'))
Set-Content -LiteralPath (Join-Path $output 'api.rsp') -Value $apiArgs -Encoding ascii
$syntaxArgs = $common + @('/Zs','/utf-8','/DWIN32_LEAN_AND_MEAN',('/I"' + $workspace + '/external/bgfx.cmake/bgfx/include"'))
$syntaxArgs += @('plugins/osint_plugin.cpp','plugins/network_test_plugin/network_test_plugin.cpp','commands/commands_osint.cpp','commands/commands_osint_ui.cpp','plugins/application_plugin/shell_open.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
if ($VoicePluginDirectory) { $syntaxArgs += '"' + (Join-Path $workspace 'voice/voice_platform_queries.cpp') + '"' }
if ($PerceptionPluginDirectory) {
    $perceptionOpenCvInclude = Join-Path $workspace 'vcpkg_installed/x64-windows/include/opencv4'
    if (-not (Test-Path -LiteralPath (Join-Path $perceptionOpenCvInclude 'opencv2/core.hpp'))) {
        $perceptionOpenCvInclude = Join-Path $workspace 'vcpkg_installed/x64-windows/include'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $perceptionOpenCvInclude 'opencv2/core.hpp'))) { throw 'Existing OpenCV headers are required for the host perception syntax check.' }
    $syntaxArgs += '/I"' + $perceptionOpenCvInclude + '"'
    $syntaxArgs += '"' + (Join-Path $workspace 'commands/commands_perception.cpp') + '"'
}
Set-Content -LiteralPath (Join-Path $output 'plugin-syntax.rsp') -Value $syntaxArgs -Encoding ascii
$registryArgs = $common + '/DGRIM_BUILD_HOST'
$registryArgs += @('tests/tool_registry_tests.cpp','MMO/Core/Registries/ToolRegistry.cpp','MMO/Core/Registries/ProviderRegistry.cpp','MMO/Core/ToolGapPlanner.cpp','commands/command_dispatch.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$registryArgs += '/Fe:tool_registry_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'registry.rsp') -Value $registryArgs -Encoding ascii
if ($BuildApplicationFixture) {
    $applicationFixture = Join-Path $output 'application-fixture'
    New-Item -ItemType Directory -Force -Path $applicationFixture | Out-Null
    $ApplicationPluginDirectory = $applicationFixture
    $applicationArgs = $common + @('/LD', '/DGRIM_BUILD_PLUGIN', '/utf-8')
    $applicationArgs += @('plugins/application_plugin/application_plugin.cpp', 'plugins/application_plugin/application_tools.cpp', 'plugins/application_plugin/application_catalog.cpp', 'plugins/application_plugin/application_backends.cpp', 'tests/fixtures/application_native_fixture.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
    $applicationArgs += '/Fe:application-fixture/application_plugin.dll'
    Set-Content -LiteralPath (Join-Path $output 'application-fixture.rsp') -Value $applicationArgs -Encoding ascii
}
if ($BuildSystemInfoFixture) {
    $SystemInfoPluginDirectory = Join-Path $output 'system-info-fixture'
    New-Item -ItemType Directory -Force -Path $SystemInfoPluginDirectory | Out-Null
    $hardwareArgs = $common + @('/LD', '/DGRIM_BUILD_PLUGIN', '/utf-8')
    $hardwareArgs += @('plugins/system_info_plugin/system_info_plugin.cpp', 'plugins/system_info_plugin/system_info_command.cpp', 'plugins/system_info_plugin/system_info_provider.cpp', 'tests/fixtures/system_info_hardware_fixture.cpp', 'plugins/plugin_logging.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
    $hardwareArgs += '/Fe:system-info-fixture/system_info_plugin.dll'
    Set-Content -LiteralPath (Join-Path $output 'system-info-fixture.rsp') -Value $hardwareArgs -Encoding ascii
}
$pluginTestCommand = 'plugin_api_ownership_tests.exe integration integration-rejected replacement'
foreach ($testDirectory in @($PluginDirectory, $FilesystemPluginDirectory, $UtilityPluginDirectory, $ApplicationPluginDirectory, $ToolInspectionPluginDirectory, $SystemInfoPluginDirectory, $TimersPluginDirectory, $MemoryPluginDirectory, $PerceptionPluginDirectory, $VoicePluginDirectory, $UIPluginDirectory, $InputPluginDirectory)) {
    $resolvedTestDirectory = if ($testDirectory) { (Resolve-Path -LiteralPath $testDirectory).Path } else { '' }
    $pluginTestCommand += ' "' + $resolvedTestDirectory + '"'
}
$commands = @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%',
    'cl @alpha.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @beta.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @failed.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @legacy.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @tests.rsp','if errorlevel 1 exit /b %errorlevel%',
    'plugin_shutdown_tests.exe fixtures rejected','if errorlevel 1 exit /b %errorlevel%',
    'cl @integration-alpha.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @integration-beta.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @integration-failed.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @integration-legacy.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @replacement-old.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @replacement-new.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @replacement-failed.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @replacement-collision.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @replacement-legacy.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @api.rsp','if errorlevel 1 exit /b %errorlevel%',
    $pluginTestCommand,'if errorlevel 1 exit /b %errorlevel%',
    'cl @plugin-syntax.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @registry.rsp','if errorlevel 1 exit /b %errorlevel%',
    'tool_registry_tests.exe','if errorlevel 1 exit /b %errorlevel%')
Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
if ($BuildUIFixture) {
    $insert = @('cl @ui-fixture.rsp', 'if errorlevel 1 exit /b %errorlevel%')
    $commands = $commands[0..2] + $insert + $commands[3..($commands.Length - 1)]
    Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
}
if ($BuildApplicationFixture) {
    $commands = $commands[0..2] + @('cl @application-fixture.rsp', 'if errorlevel 1 exit /b %errorlevel%') + $commands[3..($commands.Length - 1)]
    Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
}
if ($BuildSystemInfoFixture) {
    $commands = $commands[0..2] + @('cl @system-info-fixture.rsp', 'if errorlevel 1 exit /b %errorlevel%') + $commands[3..($commands.Length - 1)]
    Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
}
if ($BuildFilesystemFixture) {
    $commands = $commands[0..2] + @('cl @filesystem-fixture.rsp', 'if errorlevel 1 exit /b %errorlevel%') + $commands[3..($commands.Length - 1)]
    Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
}
Push-Location $output
try {
    & $env:ComSpec /d /c test.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Host-only plugin shutdown tests failed.' }
} finally { Pop-Location }
