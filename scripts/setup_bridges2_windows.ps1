# ============================================================
# GRIM Bridges-2 (PSC /ocean) Windows + VS Code Remote-SSH Setup
# Ensures the local Windows OpenSSH config matches the workflow
# used by scripts/run_train_on_bridges2.sh and sync_models_bridges2.sh.
# ============================================================

param(
    [string]$HostAlias = "bridges2",
    [string]$HostName = "bridges2.psc.edu",
    [string]$Bridges2User = "",
    [string]$AccessAllocation = "cis210058p",
    [string]$IdentityFile = "~/.ssh/id_ed25519",
    [string]$RemoteRepoDir = "",
    [switch]$GenerateKeyIfMissing,
    [switch]$PrintPublicKey,
    [switch]$SkipConnectionTest
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

function Get-SshConfigPath {
    return Join-Path $HOME ".ssh\config"
}

function Format-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-CapturedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = (($Arguments | ForEach-Object { Format-ProcessArgument $_ }) -join ' ')
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void]$process.Start()

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    $output = @()
    if ($stdout) {
        $output += ($stdout -split "`r?`n")
    }
    if ($stderr) {
        $output += ($stderr -split "`r?`n")
    }

    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output = ($output | Where-Object { $_ -ne '' })
    }
}

function Get-IdentityFileForConfig {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path -match '^[A-Za-z]:\\') {
        return ($Path -replace '\\', '/')
    }

    return $Path
}

function Get-HostBlock {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$ConfigText,
        [Parameter(Mandatory = $true)][string]$Alias
    )

    $pattern = "(?ms)^Host\s+$([regex]::Escape($Alias))\s*$\r?\n(?<body>(?:^[ \t].*\r?\n)*)"
    $match = [regex]::Match($ConfigText, $pattern)
    if (-not $match.Success) {
        return $null
    }

    return $match.Value
}

function Get-ExistingHostUser {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$ConfigText,
        [Parameter(Mandatory = $true)][string]$Alias
    )

    $hostBlock = Get-HostBlock -ConfigText $ConfigText -Alias $Alias
    if (-not $hostBlock) {
        return $null
    }

    $userMatch = [regex]::Match($hostBlock, '(?im)^\s*User\s+(\S+)\s*$')
    if ($userMatch.Success) {
        return $userMatch.Groups[1].Value
    }

    return $null
}

function Test-OpenSSHClient {
    $ssh = Get-Command ssh -ErrorAction SilentlyContinue
    if (-not $ssh) {
        throw "OpenSSH client was not found on PATH. Install the Windows OpenSSH Client feature first."
    }

    $versionResult = Invoke-CapturedCommand -FilePath 'ssh' -Arguments @('-V')
    if ($versionResult.ExitCode -ne 0) {
        throw "ssh -V failed: $($versionResult.Output -join ' ')"
    }

    Write-Host "SSH client: $($versionResult.Output -join ' ')" -ForegroundColor Gray
}

function Start-SshAgentService {
    $service = Get-Service ssh-agent -ErrorAction SilentlyContinue
    if (-not $service) {
        Write-Host "ssh-agent service not found; skipping agent setup." -ForegroundColor Yellow
        return
    }

    try {
        if ($service.StartType -eq 'Disabled') {
            Set-Service -Name ssh-agent -StartupType Manual
            $service = Get-Service ssh-agent
        }

        if ($service.Status -ne 'Running') {
            Start-Service ssh-agent
            $service = Get-Service ssh-agent
        }

        Write-Host "ssh-agent: $($service.Status) ($($service.StartType))" -ForegroundColor Gray
    } catch {
        Write-Host "ssh-agent: could not start/configure (run PowerShell as Administrator, or start manually). Continuing." -ForegroundColor Yellow
    }
}

function Initialize-IdentityFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$User
    )

    $sshDir = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $sshDir | Out-Null

    if (Test-Path $Path) {
        return
    }

    if (-not $GenerateKeyIfMissing) {
        throw "SSH private key not found at $Path. Re-run with -GenerateKeyIfMissing to create one."
    }

    Write-Host "Creating new ED25519 key at $Path" -ForegroundColor Yellow
    & ssh-keygen -t ed25519 -f $Path -C "$User@bridges2.psc.edu"
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen failed while creating $Path"
    }
}

function Add-KeyToAgent {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Get-Command ssh-add -ErrorAction SilentlyContinue)) {
        Write-Host "ssh-add not found; skipping agent load." -ForegroundColor Yellow
        return
    }

    $fingerprintResult = Invoke-CapturedCommand -FilePath 'ssh-keygen' -Arguments @('-lf', $Path)
    if ($fingerprintResult.ExitCode -ne 0) {
        throw "ssh-keygen could not read fingerprint for $Path"
    }

    $fingerprintText = $fingerprintResult.Output -join "`n"
    $fingerprintMatch = [regex]::Match($fingerprintText, 'SHA256:[A-Za-z0-9+/=]+')
    if (-not $fingerprintMatch.Success) {
        throw "Could not parse SSH fingerprint for $Path"
    }

    $fingerprint = $fingerprintMatch.Value
    $listResult = Invoke-CapturedCommand -FilePath 'ssh-add' -Arguments @('-l')
    $listText = $listResult.Output -join "`n"
    if ($listResult.ExitCode -eq 0 -and $listText -match [regex]::Escape($fingerprint)) {
        Write-Host "SSH key already loaded in ssh-agent." -ForegroundColor Green
        return
    }

    Write-Host "Adding SSH key to ssh-agent: $Path" -ForegroundColor Cyan
    $addResult = Invoke-CapturedCommand -FilePath 'ssh-add' -Arguments @($Path)
    if ($addResult.ExitCode -ne 0) {
        Write-Host "ssh-add failed (agent may be stopped). SSH will still use IdentityFile from config." -ForegroundColor Yellow
    }
}

function Set-Bridges2HostBlock {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$Alias,
        [Parameter(Mandatory = $true)][string]$HostNameValue,
        [Parameter(Mandatory = $true)][string]$User,
        [Parameter(Mandatory = $true)][string]$IdentityFileValue
    )

    $managedBlock = @"
# >>> GRIM Bridges-2 (PSC /ocean) >>>
Host $Alias
  HostName $HostNameValue
  User $User
  IdentityFile $IdentityFileValue
  IdentitiesOnly yes
  PubkeyAuthentication yes
  ServerAliveInterval 60
  ServerAliveCountMax 30
  TCPKeepAlive yes
  StrictHostKeyChecking accept-new
# <<< GRIM Bridges-2 (PSC /ocean) <<<
"@

    $configText = ""
    if (Test-Path $ConfigPath) {
        $configText = Get-Content -Raw -Path $ConfigPath
    }

    $managedPattern = '(?ms)^# >>> GRIM Bridges-2 \(PSC /ocean\) >>>\r?\n.*?^# <<< GRIM Bridges-2 \(PSC /ocean\) <<<\r?\n?'
    if ([regex]::IsMatch($configText, $managedPattern)) {
        $newText = [regex]::Replace($configText, $managedPattern, $managedBlock + [Environment]::NewLine, 1)
    } else {
        $hostPattern = "(?ms)^Host\s+$([regex]::Escape($Alias))\s*$\r?\n(?:^[ \t].*\r?\n)*"
        if ([regex]::IsMatch($configText, $hostPattern)) {
            $newText = [regex]::Replace($configText, $hostPattern, $managedBlock + [Environment]::NewLine, 1)
        } elseif ([string]::IsNullOrWhiteSpace($configText)) {
            $newText = $managedBlock + [Environment]::NewLine
        } else {
            $trimmed = $configText.TrimEnd("`r", "`n")
            $newText = $trimmed + [Environment]::NewLine + [Environment]::NewLine + $managedBlock + [Environment]::NewLine
        }
    }

    Set-Content -Path $ConfigPath -Value $newText -NoNewline
}

function Test-Bridges2BatchConnection {
    param(
        [Parameter(Mandatory = $true)][string]$Alias,
        [Parameter(Mandatory = $true)][string]$PublicKeyPath
    )

    $probeResult = Invoke-CapturedCommand -FilePath 'ssh' -Arguments @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', $Alias, 'hostname')
    if ($probeResult.ExitCode -eq 0) {
        return [pscustomobject]@{
            Success = $true
            Summary = "Bridges-2 accepted the configured SSH key in batch mode."
            Details = ($probeResult.Output -join "`n")
        }
    }

    $debugResult = Invoke-CapturedCommand -FilePath 'ssh' -Arguments @('-vv', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', $Alias, 'true')
    $text = $debugResult.Output -join "`n"

    if ($text -match 'Offering public key:' -and $text -match 'Permission denied') {
        return [pscustomobject]@{
            Success = $false
            Summary = "Windows-side SSH is configured, but Bridges-2 rejected the public key."
            Details = "Run scripts/install_bridges2_public_key.ps1 (or upload $PublicKeyPath via ACCESS / PSC account portal), then re-run this script."
        }
    }

    return [pscustomobject]@{
        Success = $false
        Summary = "SSH to Bridges-2 still failed."
        Details = $text
    }
}

$ResolvedIdentityFile = Resolve-HomePath $IdentityFile
$ResolvedPublicKey = "$ResolvedIdentityFile.pub"
$ConfigPath = Get-SshConfigPath
$configText = ""
if (Test-Path $ConfigPath) {
    $configText = Get-Content -Raw -Path $ConfigPath
    if ($null -eq $configText) {
        $configText = ""
    }
}
$existingUser = Get-ExistingHostUser -ConfigText $configText -Alias $HostAlias

if ([string]::IsNullOrWhiteSpace($Bridges2User)) {
    $Bridges2User = $existingUser
}

if ([string]::IsNullOrWhiteSpace($Bridges2User)) {
    throw "Bridges-2 username is required. Pass -Bridges2User <psc_username> the first time you run this script."
}

if ([string]::IsNullOrWhiteSpace($RemoteRepoDir)) {
    $RemoteRepoDir = "/ocean/projects/$AccessAllocation/$Bridges2User/G.R.I.M"
}

$IdentityFileForConfig = Get-IdentityFileForConfig $IdentityFile

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  GRIM Bridges-2 /ocean Windows + VS Code SSH" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "SSH config : $ConfigPath" -ForegroundColor Gray
Write-Host "Host alias : $HostAlias" -ForegroundColor Gray
Write-Host "Host name  : $HostName" -ForegroundColor Gray
Write-Host "PSC user   : $Bridges2User" -ForegroundColor Gray
Write-Host "Allocation : $AccessAllocation" -ForegroundColor Gray
Write-Host "Identity   : $ResolvedIdentityFile" -ForegroundColor Gray
Write-Host "Remote dir : $RemoteRepoDir" -ForegroundColor Gray

Test-OpenSSHClient
Start-SshAgentService
Initialize-IdentityFile -Path $ResolvedIdentityFile -User $Bridges2User
Set-Bridges2HostBlock -ConfigPath $ConfigPath -Alias $HostAlias -HostNameValue $HostName -User $Bridges2User -IdentityFileValue $IdentityFileForConfig
Add-KeyToAgent -Path $ResolvedIdentityFile

$envSnippet = @"
# Paste into PowerShell profile or set before running run_train_on_bridges2.sh:
`$env:GRIM_BRIDGES2_SSH = '$HostAlias'
`$env:GRIM_BRIDGES2_ACCOUNT = '$AccessAllocation'
`$env:GRIM_BRIDGES2_DIR = '$RemoteRepoDir'
"@

Write-Host "`nLauncher environment (save these):" -ForegroundColor Cyan
Write-Host $envSnippet -ForegroundColor Gray

if ($PrintPublicKey) {
    if (-not (Test-Path $ResolvedPublicKey)) {
        throw "Missing public key: $ResolvedPublicKey"
    }

    Write-Host "`nPublic key ($ResolvedPublicKey):" -ForegroundColor Yellow
    Get-Content -Path $ResolvedPublicKey
}

if (-not $SkipConnectionTest) {
    Write-Host "`nTesting non-interactive SSH to $HostAlias ..." -ForegroundColor Yellow
    $result = Test-Bridges2BatchConnection -Alias $HostAlias -PublicKeyPath $ResolvedPublicKey
    if ($result.Success) {
        Write-Host "[OK] $($result.Summary)" -ForegroundColor Green
    } else {
        Write-Host "[WARN] $($result.Summary)" -ForegroundColor Yellow
        Write-Host $result.Details -ForegroundColor Yellow
    }
}

Write-Host "`nVS Code Remote-SSH next steps:" -ForegroundColor Cyan
Write-Host "  1. Ctrl+Shift+P" -ForegroundColor Gray
Write-Host "  2. Remote-SSH: Connect to Host..." -ForegroundColor Gray
Write-Host "  3. Select '$HostAlias'" -ForegroundColor Gray
Write-Host "  4. Open folder: $RemoteRepoDir" -ForegroundColor Gray
Write-Host "`nWorkflow note:" -ForegroundColor Cyan
Write-Host "  - From Git Bash / WSL at local repo: bash scripts/run_train_on_bridges2.sh" -ForegroundColor Gray
Write-Host "  - From a Remote-SSH session already on Bridges-2: run native cluster commands there; do not SSH to '$HostAlias' again from inside that session." -ForegroundColor Gray
