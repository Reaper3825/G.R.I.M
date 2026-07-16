[CmdletBinding()]
param(
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifestPath = Join-Path $PSScriptRoot 'mediapipe_backend_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$cacheRoot = Join-Path $repoRoot '.cache\mediapipe'
$externalRoot = Join-Path $repoRoot 'external'
$sourceRoot = Join-Path $externalRoot 'mediapipe'
$toolRoot = Join-Path $externalRoot 'mediapipe-tools'
$modelPath = Join-Path $repoRoot 'resources\models\vision\mediapipe\gesture_recognizer.task'
$sourceArchive = Join-Path $cacheRoot "mediapipe-$($manifest.mediapipe.version).zip"
$bazeliskPath = Join-Path $toolRoot 'bazelisk.exe'
$vcpkgOpenCvRoot = Join-Path $repoRoot 'vcpkg_installed\x64-windows'

function Get-NormalizedHash {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-PinnedFile {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][string]$Sha256
    )

    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-NormalizedHash $Destination) -eq $Sha256.ToLowerInvariant()) {
            Write-Host "Verified cached asset: $Destination"
            return
        }
        Remove-Item -LiteralPath $Destination -Force
    }

    Write-Host "Downloading pinned asset: $Uri"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 3 --output $Destination $Uri
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed while downloading $Uri"
        }
    } else {
        Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    }

    $actual = Get-NormalizedHash $Destination
    if ($actual -ne $Sha256.ToLowerInvariant()) {
        Remove-Item -LiteralPath $Destination -Force
        throw "SHA256 mismatch for $Uri (expected $Sha256, received $actual)"
    }
}

function Find-HostPython {
    foreach ($name in @('python.exe', 'python3.exe', 'py.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        try {
            $resolved = (& $command.Source -c `
                'import os,sys; print(os.path.realpath(sys.executable))' `
                2>$null | Select-Object -First 1).Trim()
            if ($resolved -and (Test-Path -LiteralPath $resolved) -and
                (Get-Item -LiteralPath $resolved).Length -gt 0) {
                return [System.IO.Path]::GetFullPath($resolved)
            }
        } catch {
            continue
        }
    }
    throw 'LLVM repository setup requires a runnable host Python interpreter.'
}

New-Item -ItemType Directory -Path $cacheRoot, $externalRoot, $toolRoot -Force |
    Out-Null

Get-PinnedFile -Uri $manifest.mediapipe.source_url `
    -Destination $sourceArchive -Sha256 $manifest.mediapipe.source_sha256
Get-PinnedFile -Uri $manifest.bazelisk.windows_x86_64_url `
    -Destination $bazeliskPath `
    -Sha256 $manifest.bazelisk.windows_x86_64_sha256
Get-PinnedFile -Uri $manifest.gesture_model.url -Destination $modelPath `
    -Sha256 $manifest.gesture_model.sha256

if (-not (Test-Path -LiteralPath $sourceRoot)) {
    Write-Host "Expanding MediaPipe $($manifest.mediapipe.version)..."
    Expand-Archive -LiteralPath $sourceArchive -DestinationPath $externalRoot
    $expandedRoot = Join-Path $externalRoot "mediapipe-$($manifest.mediapipe.version)"
    if (-not (Test-Path -LiteralPath $expandedRoot)) {
        throw "The pinned MediaPipe archive did not contain $expandedRoot"
    }
    Move-Item -LiteralPath $expandedRoot -Destination $sourceRoot
}

$bazelVersionPath = Join-Path $sourceRoot '.bazelversion'
if (-not (Test-Path -LiteralPath $bazelVersionPath)) {
    throw "MediaPipe source is incomplete: .bazelversion is missing"
}
$sourceBazelVersion = (Get-Content -LiteralPath $bazelVersionPath -Raw).Trim()
if ($sourceBazelVersion -ne '7.4.1') {
    throw "Unexpected MediaPipe Bazel version: $sourceBazelVersion"
}

foreach ($critical in $manifest.mediapipe.critical_files) {
    $criticalPath = Join-Path $sourceRoot $critical.path
    if (-not (Test-Path -LiteralPath $criticalPath)) {
        throw "Pinned MediaPipe source file is missing: $($critical.path)"
    }
    $criticalHash = Get-NormalizedHash $criticalPath
    if ($criticalHash -ne $critical.sha256.ToLowerInvariant()) {
        throw "Pinned MediaPipe source file changed: $($critical.path)"
    }
}

# v0.10.35 removed the analytics logger from the public Tasks C build graph.
# Guard that property before allowing an audited backend artifact.
$tasksBuildPath = Join-Path $sourceRoot 'mediapipe\tasks\c\BUILD'
$tasksBuild = Get-Content -LiteralPath $tasksBuildPath -Raw
if ($tasksBuild -match 'util/analytics|ClearcutLoggingClient|TasksStatsProtoLogger') {
    throw 'The MediaPipe Tasks C build graph contains a usage-logging dependency.'
}

Write-Host "MediaPipe source: $sourceRoot"
Write-Host "Gesture model:   $modelPath"
Write-Host "Bazelisk:        $bazeliskPath"

if (-not $Build) {
    Write-Host 'Preparation complete. No native code was built.'
    Write-Host 'Run this script again with -Build to compile and audit libmediapipe.'
    exit 0
}

if ($env:OS -ne 'Windows_NT') {
    throw 'This setup script currently builds the Windows x86-64 backend only.'
}

$opencvHeader = Join-Path $vcpkgOpenCvRoot 'include\opencv4\opencv2\core.hpp'
$opencvCoreLibrary = Join-Path $vcpkgOpenCvRoot 'lib\opencv_core4.lib'
if (-not (Test-Path -LiteralPath $opencvHeader) -or
    -not (Test-Path -LiteralPath $opencvCoreLibrary)) {
    throw "GRIM's x64-windows vcpkg OpenCV installation is incomplete: $vcpkgOpenCvRoot"
}

# MediaPipe's upstream Windows workspace assumes the legacy standalone SDK at
# C:\opencv\build. Repoint only that repository to GRIM's existing vcpkg tree
# and supply a BUILD adapter for its modular OpenCV 4 import libraries.
$workspacePath = Join-Path $sourceRoot 'WORKSPACE'
$workspaceText = Get-Content -LiteralPath $workspacePath -Raw
$windowsOpenCvPattern = [regex]::new(
    '(?s)(name\s*=\s*"windows_opencv".{0,300}?path\s*=\s*)"[^"]+"')
$windowsOpenCvMatches = $windowsOpenCvPattern.Matches($workspaceText)
if ($windowsOpenCvMatches.Count -ne 1) {
    throw 'Unable to locate the windows_opencv repository in MediaPipe WORKSPACE.'
}
$opencvWorkspacePath = $vcpkgOpenCvRoot.Replace('\', '/')
$workspaceText = $windowsOpenCvPattern.Replace(
    $workspaceText,
    [System.Text.RegularExpressions.MatchEvaluator]{
        param($match)
        return $match.Groups[1].Value + '"' + $opencvWorkspacePath + '"'
    },
    1)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($workspacePath, $workspaceText, $utf8NoBom)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot `
    'mediapipe\opencv_windows_vcpkg.BUILD') `
    -Destination (Join-Path $sourceRoot 'third_party\opencv_windows.BUILD') `
    -Force

# Add a GRIM-only Bazel package rather than modifying a pinned upstream BUILD
# file. Its output is copied to the stable path consumed by CMake below.
$grimBazelPackage = Join-Path $sourceRoot 'mediapipe\tasks\c\grim'
New-Item -ItemType Directory -Path $grimBazelPackage -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot `
    'mediapipe\grim_gesture_backend.BUILD') `
    -Destination (Join-Path $grimBazelPackage 'BUILD') -Force

$gitBash = Join-Path $env:ProgramFiles 'Git\bin\bash.exe'
if (Test-Path -LiteralPath $gitBash) {
    $env:BAZEL_SH = $gitBash
}

# TensorFlow uses the hermetic 3.12 interpreter selected below. LLVM's Bazel
# repository overlay is configured earlier and separately calls `which()` for
# a host python/python3 executable. Resolve past WindowsApps aliases and expose
# the real interpreter directory explicitly to repository rules.
$hostPython = Find-HostPython
$repositoryPath = "$(Split-Path -Parent $hostPython);$env:PATH"
$env:PATH = $repositoryPath
Write-Host "LLVM setup Python: $hostPython"

Write-Host 'Building the CPU-only MediaPipe Tasks C runtime...'
Push-Location $sourceRoot
try {
    $bazelArguments = @(
        'build',
        '-c', 'opt',
        '--strip', 'always',
        '--repo_env=HERMETIC_PYTHON_VERSION=3.12',
        "--repo_env=PATH=$repositoryPath",
        '--conlyopt=/std:c11',
        '--host_conlyopt=/std:c11',
        '--define', 'MEDIAPIPE_DISABLE_GPU=1',
        '//mediapipe/tasks/c/grim:libmediapipe.dll'
    )
    & $bazeliskPath @bazelArguments
    if ($LASTEXITCODE -ne 0) {
        throw "MediaPipe Bazel build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$builtRuntimePath = Join-Path $sourceRoot `
    'bazel-bin\mediapipe\tasks\c\grim\libmediapipe.dll'
if (-not (Test-Path -LiteralPath $builtRuntimePath)) {
    throw "Bazel completed without producing $builtRuntimePath"
}
$runtimePath = Join-Path $sourceRoot `
    'bazel-bin\mediapipe\tasks\c\libmediapipe.dll'
Copy-Item -LiteralPath $builtRuntimePath -Destination $runtimePath -Force

$runtimeBytes = [System.IO.File]::ReadAllBytes($runtimePath)
$runtimeText = [System.Text.Encoding]::ASCII.GetString($runtimeBytes)
$forbiddenMarkers = @(
    'https://play.googleapis.com/log',
    'ClearcutLoggingClient',
    'TasksStatsProtoLogger'
)
foreach ($marker in $forbiddenMarkers) {
    if ($runtimeText.Contains($marker)) {
        throw "Offline audit rejected libmediapipe.dll: found '$marker'"
    }
}

$audit = [ordered]@{
    schema = 1
    mediapipe_version = $manifest.mediapipe.version
    runtime_sha256 = Get-NormalizedHash $runtimePath
    built_utc = [DateTime]::UtcNow.ToString('o')
    build_target = '//mediapipe/tasks/c/grim:libmediapipe.dll'
    cpu_only = $true
    usage_logging_markers_absent = $true
}
$auditPath = Join-Path $sourceRoot '.grim-offline-audit.json'
$audit | ConvertTo-Json | Set-Content -LiteralPath $auditPath -Encoding UTF8

Write-Host "Backend ready: $runtimePath"
Write-Host "Offline audit: $auditPath"
Write-Host 'Configure GRIM with -DGRIM_USE_MEDIAPIPE_HAND_GESTURES=ON.'
