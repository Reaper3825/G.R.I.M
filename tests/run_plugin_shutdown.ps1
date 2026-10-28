$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/plugin-shutdown-tests'
$fixtures = Join-Path $output 'fixtures'
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $output 'rejected') | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$common = @('/nologo','/std:c++20','/EHsc','/W3','/DNOMINMAX')
$common += @('', '/helpers','/core','/commands','/vcpkg_installed/x64-windows/include') | ForEach-Object { '/I"' + $workspace + $_ + '"' }
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
$testArgs = $common + '/DGRIM_BUILD_HOST'
$testArgs += @('tests/plugin_shutdown_tests.cpp','core/plugin_manager.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$testArgs += '/Fe:plugin_shutdown_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'tests.rsp') -Value $testArgs -Encoding ascii
$apiArgs = $common + '/DGRIM_BUILD_HOST'
$apiArgs += @('tests/plugin_api_ownership_tests.cpp','core/plugin_api_impl.cpp','MMO/Core/ToolRegistry.cpp','commands/command_dispatch.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$apiArgs += @('/Fe:plugin_api_ownership_tests.exe', '/link', 'psapi.lib')
Set-Content -LiteralPath (Join-Path $output 'api.rsp') -Value $apiArgs -Encoding ascii
$registryArgs = $common + '/DGRIM_BUILD_HOST'
$registryArgs += @('tests/tool_registry_tests.cpp','MMO/Core/ToolRegistry.cpp','commands/command_dispatch.cpp','core/plugin.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$registryArgs += '/Fe:tool_registry_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'registry.rsp') -Value $registryArgs -Encoding ascii
$commands = @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%',
    'cl @alpha.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @beta.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @failed.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @legacy.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @tests.rsp','if errorlevel 1 exit /b %errorlevel%',
    'plugin_shutdown_tests.exe fixtures rejected','if errorlevel 1 exit /b %errorlevel%',
    'cl @api.rsp','if errorlevel 1 exit /b %errorlevel%',
    'plugin_api_ownership_tests.exe','if errorlevel 1 exit /b %errorlevel%',
    'cl @registry.rsp','if errorlevel 1 exit /b %errorlevel%',
    'tool_registry_tests.exe','if errorlevel 1 exit /b %errorlevel%')
Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c test.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Host-only plugin shutdown tests failed.' }
} finally { Pop-Location }
