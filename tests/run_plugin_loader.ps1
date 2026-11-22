$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace ('.codex-build\plugin-loader-' + [guid]::NewGuid().ToString('N'))
$toolkit = Join-Path $output 'toolkit bin'
foreach ($folder in @($output, $toolkit, (Join-Path $output 'fixtures'), (Join-Path $output 'rejected'))) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
$common = @('/nologo', '/std:c++20', '/EHsc', '/MD', '/W3', '/utf-8', '/DNOMINMAX')
foreach ($include in @($workspace, (Join-Path $workspace 'core'), (Join-Path $workspace 'helpers'), (Join-Path $workspace 'commands'), (Join-Path $workspace 'vcpkg_installed\x64-windows\include'))) {
    $common += '/I"' + $include + '"'
}
$commands = @('@echo off', ('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'), 'if errorlevel 1 exit /b %errorlevel%')
$dependency = $common + @('/LD', ('"' + (Join-Path $workspace 'tests\fixtures\plugin_native_dependency.cpp') + '"'), '/Fe:"toolkit bin\grim_test_dependency.dll"')
Set-Content -LiteralPath (Join-Path $output 'dependency.rsp') -Value $dependency -Encoding ascii
$commands += @('cl @dependency.rsp', 'if errorlevel 1 exit /b %errorlevel%')
foreach ($fixture in @('alpha', 'beta', 'failed', 'legacy')) {
    $folder = if ($fixture -in @('failed', 'legacy')) { 'rejected' } else { 'fixtures' }
    $folder += '/' + $fixture
    New-Item -ItemType Directory -Force -Path (Join-Path $output $folder) | Out-Null
    $argsForFixture = $common + @('/LD', ('/DFIXTURE_NAME=\"Declared-' + $fixture + '\"'), ('"' + (Join-Path $workspace 'tests\fixtures\plugin_shutdown_fixture.cpp') + '"'), ('/Fe:' + $folder + '\' + $fixture + '.dll'))
    if ($fixture -eq 'failed') { $argsForFixture += '/DFIXTURE_FAIL_INIT' }
    $apiVersion = if ($fixture -eq 'legacy') { 65536 } else { 1769472 }
    if ($fixture -eq 'alpha') { $argsForFixture += @('/DFIXTURE_NATIVE_DEPENDENCY', '/link', '"toolkit bin\grim_test_dependency.lib"') }
    Set-Content -LiteralPath (Join-Path $output ($fixture + '.rsp')) -Value $argsForFixture -Encoding ascii
    @{schema_version = 2; api_version = $apiVersion; provider = "Declared-$fixture"; version = '1.0'; tools = @()} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output "$folder/$fixture.tools.json") -Encoding ascii
    $commands += @("cl @$fixture.rsp", 'if errorlevel 1 exit /b %errorlevel%')
}
$testArgs = $common + @('/DGRIM_BUILD_HOST', ('/D"GRIM_PLUGIN_CUDA_BIN_DIR=\"' + $toolkit.Replace('\', '/') + '\""'))
$testArgs += @('tests\plugin_shutdown_tests.cpp', 'core\plugin_manager.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$testArgs += '/Fe:plugin_shutdown_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'tests.rsp') -Value $testArgs -Encoding ascii
$commands += @('cl @tests.rsp', 'if errorlevel 1 exit /b %errorlevel%', 'plugin_shutdown_tests.exe fixtures rejected', 'if errorlevel 1 exit /b %errorlevel%')
$commands += @('ren "toolkit bin" "hidden toolkit"', 'if errorlevel 1 exit /b %errorlevel%', 'plugin_shutdown_tests.exe fixtures rejected --missing-toolkit', 'if errorlevel 1 exit /b %errorlevel%')
$cpuArgs = $common + '/DGRIM_BUILD_HOST'
$cpuArgs += @('tests\plugin_shutdown_tests.cpp', 'core\plugin_manager.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$cpuArgs += '/Fe:plugin_shutdown_cpu_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'cpu-tests.rsp') -Value $cpuArgs -Encoding ascii
# Exercise the same native import with only the plugin-local dependency directory.
$commands += @('mkdir "fixtures\alpha\dependencies"', 'if errorlevel 1 exit /b %errorlevel%',
    'copy /y "hidden toolkit\grim_test_dependency.dll" "fixtures\alpha\dependencies\" >nul', 'if errorlevel 1 exit /b %errorlevel%',
    'cl @cpu-tests.rsp', 'if errorlevel 1 exit /b %errorlevel%',
    'plugin_shutdown_cpu_tests.exe fixtures rejected', 'exit /b %errorlevel%')
Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value $commands -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /c test.cmd
    if ($LASTEXITCODE -ne 0) { throw "Plugin loader tests failed ($LASTEXITCODE); artifacts: $output" }
} finally {
    Pop-Location
}
# This exact directory was created exclusively for this test invocation.
Remove-Item -LiteralPath $output -Recurse -Force
