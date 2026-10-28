$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/plugin-shutdown-tests'
$fixtures = Join-Path $output 'fixtures'
New-Item -ItemType Directory -Force -Path $fixtures | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$common = @('/nologo','/std:c++20','/EHsc','/W3','/DNOMINMAX')
$common += @('', '/helpers','/core','/commands') | ForEach-Object { '/I"' + $workspace + $_ + '"' }
$fixtureSource = '"' + (Join-Path $workspace 'tests/fixtures/plugin_shutdown_fixture.cpp') + '"'
foreach ($fixture in @('alpha', 'beta')) {
    $argsForFixture = $common + @('/LD',('/DFIXTURE_NAME=\"Declared-' + $fixture + '\"'),$fixtureSource,('/Fe:fixtures/' + $fixture + '.dll'))
    Set-Content -LiteralPath (Join-Path $output ($fixture + '.rsp')) -Value $argsForFixture -Encoding ascii
}
$testArgs = $common + '/DGRIM_BUILD_HOST'
$testArgs += @('tests/plugin_shutdown_tests.cpp','core/plugin_manager.cpp','commands/command_registry.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$testArgs += '/Fe:plugin_shutdown_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'tests.rsp') -Value $testArgs -Encoding ascii
$commands = @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%',
    'cl @alpha.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @beta.rsp','if errorlevel 1 exit /b %errorlevel%',
    'cl @tests.rsp','if errorlevel 1 exit /b %errorlevel%',
    'plugin_shutdown_tests.exe fixtures','if errorlevel 1 exit /b %errorlevel%')
Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c test.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Host-only plugin shutdown tests failed.' }
} finally { Pop-Location }
