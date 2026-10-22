[CmdletBinding()]
param(
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$imageName = 'grim-textworld:1.7.0'
$contextPath = Join-Path $PSScriptRoot 'textworld'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed or is not on PATH. Install/start Docker Desktop with Linux containers, then rerun this script.'
}
$serverOS = & docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Start Docker Desktop and retry.' }
if ($serverOS.Trim() -ne 'linux') { throw 'Switch Docker Desktop to Linux containers.' }

if (-not $CheckOnly) {
    & docker build --platform linux/amd64 --tag $imageName $contextPath
    if ($LASTEXITCODE -ne 0) { throw 'TextWorld image build failed.' }
}

# The smoke check needs neither network access nor workspace mounts.
& docker run --rm --network none --platform linux/amd64 $imageName
if ($LASTEXITCODE -ne 0) { throw 'TextWorld smoke check failed.' }
Write-Host "TextWorld is ready in $imageName. No inference integration is installed."
