[CmdletBinding()]
param(
    [string]$WorkspaceRoot = "",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Split-Path -Parent $PSScriptRoot
}

$resolvedRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
$repositoryPath = Join-Path $resolvedRoot "external\CosyVoice"
$environmentPath = Join-Path $resolvedRoot ".venv-cosyvoice"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"
$modelPath = Join-Path $resolvedRoot "resources\models\Fun-CosyVoice3-0.5B"
$textNormalizationPath = Join-Path $resolvedRoot "resources\models\wetext"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to provision CosyVoice."
}
if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python launcher '$PythonLauncher' was not found."
}

if (-not (Test-Path -LiteralPath $repositoryPath -PathType Container)) {
    & git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $repositoryPath
    if ($LASTEXITCODE -ne 0) {
        throw "CosyVoice repository clone failed."
    }
} else {
    & git -C $repositoryPath submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        throw "CosyVoice submodule initialization failed."
    }
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    & $PythonLauncher "-$PythonVersion" -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the dedicated Python $PythonVersion environment."
    }
}

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip bootstrap failed."
}

& $pythonPath -m pip install "setuptools<81" wheel
if ($LASTEXITCODE -ne 0) {
    throw "Python build tooling installation failed."
}

& $pythonPath -m pip install --no-build-isolation -r (Join-Path $repositoryPath "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "CosyVoice dependency installation failed."
}

# The pinned PyPI sdist can produce a metadata-only wheel on current Windows
# tooling. Repair it from the same tagged source release only when import fails.
& $pythonPath -c "import whisper"
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install --force-reinstall --no-deps --no-cache-dir `
        --no-build-isolation "git+https://github.com/openai/whisper.git@v20231117"
    if ($LASTEXITCODE -ne 0) {
        throw "OpenAI Whisper source installation failed."
    }
}

& $pythonPath -m pip install huggingface_hub
if ($LASTEXITCODE -ne 0) {
    throw "huggingface_hub installation failed."
}

if (-not (Test-Path -LiteralPath (Join-Path $modelPath "cosyvoice3.yaml") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $modelPath "llm.pt") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $modelPath "flow.pt") -PathType Leaf)) {
    $downloadCode = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir=r'''$modelPath''',
)
"@
    & $pythonPath -c $downloadCode
    if ($LASTEXITCODE -ne 0) {
        throw "Fun-CosyVoice 3 model download failed."
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $textNormalizationPath "en\tn\tagger.fst") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $textNormalizationPath "en\tn\verbalizer.fst") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $textNormalizationPath "zh\tn\tagger.fst") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $textNormalizationPath "zh\tn\verbalizer.fst") -PathType Leaf)) {
    $normalizationCode = @"
from modelscope import snapshot_download
snapshot_download(
    'pengzhendong/wetext',
    local_dir=r'''$textNormalizationPath''',
)
"@
    & $pythonPath -c $normalizationCode
    if ($LASTEXITCODE -ne 0) {
        throw "WeText normalization asset download failed."
    }
}

Write-Host "Fun-CosyVoice 3 local runtime is provisioned."
Write-Host "Python: $pythonPath"
Write-Host "Repository: $repositoryPath"
Write-Host "Model: $modelPath"
Write-Host "Text normalization: $textNormalizationPath"
Write-Host "The active ai_config voice engine was not changed."
