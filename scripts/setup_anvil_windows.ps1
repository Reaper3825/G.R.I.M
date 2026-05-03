# ============================================================
# GRIM Anvil Windows + VS Code Remote-SSH Setup
# Ensures the local Windows OpenSSH config matches the Anvil
# workflow used by scripts/run_train_on_anvil.sh.
# ============================================================

param(
    [string]$HostAlias = "anvil",
    [string]$HostName = "anvil.rcac.purdue.edu",
    [string]$AnvilUser = "",
    [string]$IdentityFile = "~/.ssh/id_ed25519",
    [string]$RemoteRepoDir = "/anvil/projects/x-cis210085/GRIM/G.R.I.M",
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
        [Parameter(Mandatory = $true)][string]$ConfigText,
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
        [Parameter(Mandatory = $true)][string]$ConfigText,
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
    $service = Get-Service ssh-agent -ErrorAction Stop
    if ($service.StartType -eq 'Disabled') {
        Set-Service -Name ssh-agent -StartupType Manual
        $service = Get-Service ssh-agent
    }

    if ($service.Status -ne 'Running') {
        Start-Service ssh-agent
        $service = Get-Service ssh-agent
    }

    Write-Host "ssh-agent: $($service.Status) ($($service.StartType))" -ForegroundColor Gray
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
    & ssh-keygen -t ed25519 -f $Path -C "$User@anvil.rcac.purdue.edu"
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen failed while creating $Path"
    }
}

function Add-KeyToAgent {
    param([Parameter(Mandatory = $true)][string]$Path)

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

    if ($listResult.ExitCode -eq 0 -and $listText -match 'SHA256:') {
        Write-Host "ssh-agent already has identities; adding the configured Anvil key as well." -ForegroundColor Yellow
    }

    Write-Host "Adding SSH key to ssh-agent: $Path" -ForegroundColor Cyan
    $addResult = Invoke-CapturedCommand -FilePath 'ssh-add' -Arguments @($Path)
    if ($addResult.ExitCode -ne 0) {
        throw "ssh-add failed for $Path"
    }
}

function Set-AnvilHostBlock {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$Alias,
        [Parameter(Mandatory = $true)][string]$HostNameValue,
        [Parameter(Mandatory = $true)][string]$User,
        [Parameter(Mandatory = $true)][string]$IdentityFileValue
    )

    $managedBlock = @"
# >>> GRIM Anvil >>>
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
# <<< GRIM Anvil <<<
"@

    $configText = ""
    if (Test-Path $ConfigPath) {
        $configText = Get-Content -Raw -Path $ConfigPath
    }

    $managedPattern = '(?ms)^# >>> GRIM Anvil >>>\r?\n.*?^# <<< GRIM Anvil <<<\r?\n?'
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

function Test-AnvilBatchConnection {
    param(
        [Parameter(Mandatory = $true)][string]$Alias,
        [Parameter(Mandatory = $true)][string]$PublicKeyPath
    )

    $probeResult = Invoke-CapturedCommand -FilePath 'ssh' -Arguments @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', $Alias, 'true')
    if ($probeResult.ExitCode -eq 0) {
        return [pscustomobject]@{
            Success = $true
            Summary = "Anvil accepted the configured SSH key in batch mode."
            Details = ($probeResult.Output -join "`n")
        }
    }

    $debugResult = Invoke-CapturedCommand -FilePath 'ssh' -Arguments @('-vv', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', $Alias, 'true')
    $text = $debugResult.Output -join "`n"

    if ($text -match 'Offering public key: .*id_ed25519' -and $text -match 'Permission denied') {
        return [pscustomobject]@{
            Success = $false
            Summary = "Windows-side SSH is configured, but Anvil rejected the public key."
            Details = "Upload $PublicKeyPath to your RCAC/Anvil account (or append it to ~/.ssh/authorized_keys on Anvil), then re-run this script."
        }
    }

    return [pscustomobject]@{
        Success = $false
        Summary = "SSH to Anvil still failed."
        Details = $text
    }
}

$ResolvedIdentityFile = Resolve-HomePath $IdentityFile
$ResolvedPublicKey = "$ResolvedIdentityFile.pub"
$ConfigPath = Get-SshConfigPath
$configText = if (Test-Path $ConfigPath) { Get-Content -Raw -Path $ConfigPath } else { "" }
$existingUser = Get-ExistingHostUser -ConfigText $configText -Alias $HostAlias

if ([string]::IsNullOrWhiteSpace($AnvilUser)) {
    $AnvilUser = $existingUser
}

if ([string]::IsNullOrWhiteSpace($AnvilUser)) {
    throw "Anvil username is required. Pass -AnvilUser <username> the first time you run this script."
}

$IdentityFileForConfig = Get-IdentityFileForConfig $IdentityFile

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  GRIM Anvil Windows + VS Code SSH Setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "SSH config : $ConfigPath" -ForegroundColor Gray
Write-Host "Host alias : $HostAlias" -ForegroundColor Gray
Write-Host "Host name  : $HostName" -ForegroundColor Gray
Write-Host "Anvil user : $AnvilUser" -ForegroundColor Gray
Write-Host "Identity   : $ResolvedIdentityFile" -ForegroundColor Gray
Write-Host "Remote dir : $RemoteRepoDir" -ForegroundColor Gray

Test-OpenSSHClient
Start-SshAgentService
Initialize-IdentityFile -Path $ResolvedIdentityFile -User $AnvilUser
Set-AnvilHostBlock -ConfigPath $ConfigPath -Alias $HostAlias -HostNameValue $HostName -User $AnvilUser -IdentityFileValue $IdentityFileForConfig
Add-KeyToAgent -Path $ResolvedIdentityFile

if ($PrintPublicKey) {
    if (-not (Test-Path $ResolvedPublicKey)) {
        throw "Missing public key: $ResolvedPublicKey"
    }

    Write-Host "`nPublic key ($ResolvedPublicKey):" -ForegroundColor Yellow
    Get-Content -Path $ResolvedPublicKey
}

if (-not $SkipConnectionTest) {
    Write-Host "`nTesting non-interactive SSH to $HostAlias ..." -ForegroundColor Yellow
    $result = Test-AnvilBatchConnection -Alias $HostAlias -PublicKeyPath $ResolvedPublicKey
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
Write-Host "  - From Windows/local repo: run scripts/run_train_on_anvil.sh" -ForegroundColor Gray
Write-Host "  - From a Remote-SSH session already on Anvil: run native Anvil commands from the remote terminal instead of SSHing to 'anvil' again." -ForegroundColor Gray
