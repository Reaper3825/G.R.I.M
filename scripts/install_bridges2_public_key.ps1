# ============================================================
# Install local Windows public key into Bridges-2 authorized_keys
# Uses the existing SSH host alias so the user only needs to
# authenticate once if password login is still required.
# ============================================================

param(
    [string]$HostAlias = "bridges2",
    [string]$PublicKeyPath = "~/.ssh/id_ed25519.pub"
)

$ErrorActionPreference = "Stop"

function Resolve-HomePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path.StartsWith("~/") -or $Path.StartsWith("~\")) {
        return Join-Path $HOME $Path.Substring(2)
    }

    if ($Path -eq "~") {
        return $HOME
    }

    return [System.IO.Path]::GetFullPath($Path)
}

function Test-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name was not found on PATH."
    }
}

$resolvedPublicKeyPath = Resolve-HomePath $PublicKeyPath
if (-not (Test-Path $resolvedPublicKeyPath)) {
    throw "Public key file not found: $resolvedPublicKeyPath"
}

Test-CommandAvailable -Name "ssh"

$publicKey = (Get-Content -Path $resolvedPublicKeyPath -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($publicKey)) {
    throw "Public key file is empty: $resolvedPublicKeyPath"
}

$escapedPublicKey = $publicKey.Replace("'", "'\''")
$remoteCommand = @"
set -eu
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
if grep -qxF '$escapedPublicKey' ~/.ssh/authorized_keys; then
  echo '__GRIM_KEY_ALREADY_PRESENT__'
else
  printf '%s\n' '$escapedPublicKey' >> ~/.ssh/authorized_keys
  echo '__GRIM_KEY_INSTALLED__'
fi
"@

Write-Host "Installing public key on $HostAlias (Bridges-2) ..." -ForegroundColor Cyan
Write-Host "If prompted, enter your PSC password once so the key can be installed." -ForegroundColor Yellow

& ssh $HostAlias "sh -lc \"$remoteCommand\""
if ($LASTEXITCODE -ne 0) {
    throw "SSH command failed while installing the public key on $HostAlias."
}

Write-Host "`nDone. Test with: ssh $HostAlias hostname" -ForegroundColor Green
