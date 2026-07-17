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

function Get-ShortBazelUserRoot {
    $defaultRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $env:USERPROFILE "_bazel_$env:USERNAME"))
    New-Item -ItemType Directory -Path $defaultRoot -Force | Out-Null

    $workspaceDrive = [System.IO.Path]::GetPathRoot($repoRoot)
    $shortRoot = Join-Path $workspaceDrive '.gbz'
    if (Test-Path -LiteralPath $shortRoot) {
        $existing = Get-Item -LiteralPath $shortRoot -Force
        if ($existing.LinkType -eq 'Junction') {
            $target = @($existing.Target) | Select-Object -First 1
            if (-not $target -or -not [string]::Equals(
                    [System.IO.Path]::GetFullPath($target),
                    $defaultRoot,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Short Bazel root is an unrelated junction: $shortRoot"
            }
            # Earlier setup revisions used a junction, but Bazel canonicalizes
            # it and recreates the long path. Delete only the verified reparse
            # point, never its target, then create a genuine short directory.
            [System.IO.Directory]::Delete($shortRoot)
            New-Item -ItemType Directory -Path $shortRoot | Out-Null
        } elseif (-not $existing.PSIsContainer) {
            throw "Short Bazel root is not a directory: $shortRoot"
        }
    } else {
        New-Item -ItemType Directory -Path $shortRoot | Out-Null
    }
    return [System.IO.Path]::GetFullPath($shortRoot)
}

function Find-Dumpbin {
    $direct = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($direct) { return $direct.Source }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} `
        'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installation) { return $null }
    return Get-ChildItem -LiteralPath (Join-Path $installation `
        'VC\Tools\MSVC') -Recurse -Filter dumpbin.exe -File `
        -ErrorAction SilentlyContinue |
        Where-Object FullName -Match 'Hostx64\\x64' |
        Select-Object -First 1 -ExpandProperty FullName
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

# MediaPipe v0.10.35 declares kGpuService with ABSL_CONST_INIT but omits the
# specifier on its definition. MSVC correctly diagnoses that C++20 mismatch.
# Apply only the exact pinned-source transformation and remain idempotent.
$gpuServicePath = Join-Path $sourceRoot 'mediapipe\gpu\gpu_service.cc'
$gpuServiceText = Get-Content -LiteralPath $gpuServicePath -Raw
$gpuServiceDefinition = @'
const GraphService<GpuResources> kGpuService(
    "kGpuService", GraphServiceBase::kAllowDefaultInitialization);
'@
$patchedGpuServiceDefinition = @'
ABSL_CONST_INIT const GraphService<GpuResources> kGpuService(
    "kGpuService", GraphServiceBase::kAllowDefaultInitialization);
'@
if ($gpuServiceText.Contains($patchedGpuServiceDefinition)) {
    Write-Host 'MediaPipe MSVC constinit patch already applied.'
} elseif ($gpuServiceText.Contains($gpuServiceDefinition)) {
    $gpuServiceText = $gpuServiceText.Replace(
        $gpuServiceDefinition, $patchedGpuServiceDefinition)
    [System.IO.File]::WriteAllText(
        $gpuServicePath, $gpuServiceText, $utf8NoBom)
    Write-Host 'Applied MediaPipe MSVC constinit patch.'
} else {
    throw 'Pinned MediaPipe gpu_service.cc no longer matches the expected source.'
}

# MSVC rejects the internal VisitPacket* helper templates because the
# non-deducible int-reference sentinel pack follows the payload type pack.
# The helpers are private and every call site supplies only payload types, so
# remove just that sentinel while preserving visitor deduction and recursion.
$calculatorContextPath = Join-Path $sourceRoot `
    'mediapipe\framework\api3\calculator_context.h'
$calculatorContextText = Get-Content -LiteralPath $calculatorContextPath -Raw
$singleVisitTemplate = `
    'template <typename T, int&... DoNotSpecify, typename F>'
$patchedSingleVisitTemplate = 'template <typename T, typename F>'
$recursiveVisitTemplate = @'
template <typename T, typename U, typename... Rest, int&... DoNotSpecify,
          typename F>
'@
$patchedRecursiveVisitTemplate = @'
template <typename T, typename U, typename... Rest, typename F>
'@
$singleOriginalCount = [regex]::Matches(
    $calculatorContextText, [regex]::Escape($singleVisitTemplate)).Count
$singlePatchedCount = [regex]::Matches(
    $calculatorContextText, [regex]::Escape($patchedSingleVisitTemplate)).Count
$recursiveOriginalCount = [regex]::Matches(
    $calculatorContextText, [regex]::Escape($recursiveVisitTemplate)).Count
$recursivePatchedCount = [regex]::Matches(
    $calculatorContextText,
    [regex]::Escape($patchedRecursiveVisitTemplate)).Count
if ($singleOriginalCount -eq 2 -and $singlePatchedCount -eq 0 -and
    $recursiveOriginalCount -eq 2 -and $recursivePatchedCount -eq 0) {
    $calculatorContextText = $calculatorContextText.Replace(
        $singleVisitTemplate, $patchedSingleVisitTemplate)
    $calculatorContextText = $calculatorContextText.Replace(
        $recursiveVisitTemplate, $patchedRecursiveVisitTemplate)
    [System.IO.File]::WriteAllText(
        $calculatorContextPath, $calculatorContextText, $utf8NoBom)
    Write-Host 'Applied MediaPipe MSVC VisitPacket template patch.'
} elseif ($singleOriginalCount -eq 0 -and $singlePatchedCount -eq 2 -and
          $recursiveOriginalCount -eq 0 -and $recursivePatchedCount -eq 2) {
    Write-Host 'MediaPipe MSVC VisitPacket template patch already applied.'
} else {
    throw 'Pinned MediaPipe calculator_context.h no longer matches the expected source.'
}

# graph.h befriends api3::SubgraphContext<NodeT> before that template has been
# declared. MSVC instead finds mediapipe::SubgraphContext in the outer
# namespace and rejects a template friend declaration for the non-template
# class. Forward-declare the intended API3 template.
$api3GraphPath = Join-Path $sourceRoot 'mediapipe\framework\api3\graph.h'
$api3GraphText = Get-Content -LiteralPath $api3GraphPath -Raw
$graphForwardDeclaration = @'
template <template <typename, typename...> typename ContractT, typename... Ts>
class Graph;
'@
$patchedGraphForwardDeclaration = @'
template <template <typename, typename...> typename ContractT, typename... Ts>
class Graph;

template <typename NodeT>
class SubgraphContext;
'@
if ($api3GraphText.Contains($patchedGraphForwardDeclaration)) {
    Write-Host 'MediaPipe MSVC SubgraphContext declaration patch already applied.'
} elseif ($api3GraphText.Contains($graphForwardDeclaration)) {
    $api3GraphText = $api3GraphText.Replace(
        $graphForwardDeclaration, $patchedGraphForwardDeclaration)
    [System.IO.File]::WriteAllText($api3GraphPath, $api3GraphText, $utf8NoBom)
    Write-Host 'Applied MediaPipe MSVC SubgraphContext declaration patch.'
} else {
    throw 'Pinned MediaPipe graph.h no longer matches the expected source.'
}

# Mirror the header's ABSL_CONST_INIT specifier onto the two explicit
# thread-local current_ definitions. MSVC requires declaration and definition
# to carry matching C++20 constinit semantics.
$legacySupportPath = Join-Path $sourceRoot `
    'mediapipe\framework\legacy_calculator_support.cc'
$legacySupportText = Get-Content -LiteralPath $legacySupportPath -Raw
$legacyCurrentDefinitions = @'
template <>
thread_local CalculatorContext*
    LegacyCalculatorSupport::Scoped<CalculatorContext>::current_ = nullptr;
template <>
thread_local CalculatorContract*
    LegacyCalculatorSupport::Scoped<CalculatorContract>::current_ = nullptr;
'@
$patchedLegacyCurrentDefinitions = @'
template <>
ABSL_CONST_INIT thread_local CalculatorContext*
    LegacyCalculatorSupport::Scoped<CalculatorContext>::current_ = nullptr;
template <>
ABSL_CONST_INIT thread_local CalculatorContract*
    LegacyCalculatorSupport::Scoped<CalculatorContract>::current_ = nullptr;
'@
if ($legacySupportText.Contains($patchedLegacyCurrentDefinitions)) {
    Write-Host 'MediaPipe MSVC legacy constinit patch already applied.'
} elseif ($legacySupportText.Contains($legacyCurrentDefinitions)) {
    $legacySupportText = $legacySupportText.Replace(
        $legacyCurrentDefinitions, $patchedLegacyCurrentDefinitions)
    [System.IO.File]::WriteAllText(
        $legacySupportPath, $legacySupportText, $utf8NoBom)
    Write-Host 'Applied MediaPipe MSVC legacy constinit patch.'
} else {
    throw 'Pinned MediaPipe legacy_calculator_support.cc no longer matches expected source.'
}

# image_c_lib is marked alwayslink upstream, but it has no source of its own;
# image.cc remains in the non-alwayslink :image archive and is discarded when
# no linked C++ implementation calls MpImageCreate*. GRIM calls those exports
# dynamically, so force the implementation archive into the shared library.
$imageBuildPath = Join-Path $sourceRoot `
    'mediapipe\tasks\c\vision\core\BUILD'
$imageBuildText = Get-Content -LiteralPath $imageBuildPath -Raw
$imageLibraryRule = @'
cc_library(
    name = "image",
    srcs = ["image.cc"],
    hdrs = ["image.h"],
    deps = IMAGE_DEPS,
)
'@
$patchedImageLibraryRule = @'
cc_library(
    name = "image",
    srcs = ["image.cc"],
    hdrs = ["image.h"],
    deps = IMAGE_DEPS,
    alwayslink = 1,
)
'@
if ($imageBuildText.Contains($patchedImageLibraryRule)) {
    Write-Host 'MediaPipe MpImage export patch already applied.'
} elseif ($imageBuildText.Contains($imageLibraryRule)) {
    $imageBuildText = $imageBuildText.Replace(
        $imageLibraryRule, $patchedImageLibraryRule)
    [System.IO.File]::WriteAllText($imageBuildPath, $imageBuildText, $utf8NoBom)
    Write-Host 'Applied MediaPipe MpImage export patch.'
} else {
    throw 'Pinned MediaPipe vision/core/BUILD no longer matches expected source.'
}

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
$shortBazelRoot = Get-ShortBazelUserRoot
$sharedRepositoryCache = Join-Path `
    (Join-Path $env:USERPROFILE "_bazel_$env:USERNAME") `
    'cache\repos\v1'
New-Item -ItemType Directory -Path $sharedRepositoryCache -Force | Out-Null
Write-Host "Short Bazel root:   $shortBazelRoot"

Write-Host 'Building the CPU-only MediaPipe Tasks C runtime...'
Push-Location $sourceRoot
try {
    $bazelArguments = @(
        "--output_user_root=$shortBazelRoot",
        'build',
        "--repository_cache=$sharedRepositoryCache",
        '-c', 'opt',
        '--strip', 'always',
        '--repo_env=HERMETIC_PYTHON_VERSION=3.12',
        "--repo_env=PATH=$repositoryPath",
        '--conlyopt=/std:c11',
        '--conlyopt=/experimental:c11atomics',
        '--host_conlyopt=/std:c11',
        '--host_conlyopt=/experimental:c11atomics',
        '--cxxopt=/Zc:preprocessor',
        '--host_cxxopt=/Zc:preprocessor',
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

$dumpbin = Find-Dumpbin
if (-not $dumpbin) {
    throw 'dumpbin.exe is required to validate the MediaPipe runtime exports.'
}
$exportTable = (& $dumpbin /exports $runtimePath 2>&1) -join "`n"
$requiredExports = @(
    'MpErrorFree',
    'MpImageCreateFromUint8Data',
    'MpImageFree',
    'MpGestureRecognizerCreate',
    'MpGestureRecognizerRecognizeForVideo',
    'MpGestureRecognizerCloseResult',
    'MpGestureRecognizerClose'
)
foreach ($requiredExport in $requiredExports) {
    if ($exportTable -notmatch "(?m)\b$([regex]::Escape($requiredExport))\s*$") {
        throw "MediaPipe runtime is missing required export: $requiredExport"
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
    required_exports_verified = $true
}
$auditPath = Join-Path $sourceRoot '.grim-offline-audit.json'
$audit | ConvertTo-Json | Set-Content -LiteralPath $auditPath -Encoding UTF8

Write-Host "Backend ready: $runtimePath"
Write-Host "Offline audit: $auditPath"
Write-Host 'Configure GRIM with -DGRIM_USE_MEDIAPIPE_HAND_GESTURES=ON.'
