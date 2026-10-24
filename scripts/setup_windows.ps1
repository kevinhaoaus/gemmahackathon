<#
.SYNOPSIS
  SwarmMind setup for the Windows laptop node (Windows PowerShell 5.1 compatible).

.DESCRIPTION
  1. Creates .venv with Python 3.11 (py -3.11) and installs requirements.txt
     (online from PyPI, or offline from a wheels folder with -Wheels).
  2. Checks that Ollama answers at 127.0.0.1:11434 and that gemma4:e4b is listed.
     It never pulls a model; it only warns.
  3. Adds Windows Firewall inbound rules: UDP 50000 (heartbeat), TCP 8700 (node API / peer bus),
     TCP 8800 (LAN uplink sink). Existing rules are left alone. Needs an elevated (admin) shell.
  4. Prints the laptop's LAN IP for configs/pi.yaml static_peers.

.PARAMETER Wheels
  Folder of pre-downloaded wheels (made on a connected machine with
  "pip download -r requirements.txt -d wheels"). Installs with --no-index.
  Default: <repo>\wheels if that folder exists, otherwise PyPI.

.PARAMETER SkipPip
  Do not install Python packages.

.PARAMETER SkipFirewall
  Do not touch firewall rules.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Wheels D:\wheels
#>
[CmdletBinding()]
param(
    [string]$Wheels = "",
    [switch]$SkipPip,
    [switch]$SkipFirewall
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
$VenvPy = Join-Path $RepoDir ".venv\Scripts\python.exe"
$OllamaUrl = "http://127.0.0.1:11434"
$Model = "gemma4:e4b"
$Warnings = New-Object System.Collections.Generic.List[string]

function Write-Step([string]$msg) { Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "[ ok  ] $msg" -ForegroundColor Green }
function Write-Warn2([string]$msg) {
    Write-Host "[warn ] $msg" -ForegroundColor Yellow
    $Warnings.Add($msg) | Out-Null
}

Write-Step "repo: $RepoDir"

# ---------------------------------------------------------------------------
# 1. Python venv + packages
# ---------------------------------------------------------------------------
if (Test-Path $VenvPy) {
    Write-Ok "venv exists: $RepoDir\.venv"
} else {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $py) {
        throw "The 'py' launcher was not found. Install Python 3.11 (python.org installer, with the py launcher) and re-run."
    }
    Write-Step "creating .venv with py -3.11"
    & py -3.11 -m venv (Join-Path $RepoDir ".venv")
    if ($LASTEXITCODE -ne 0) { throw "py -3.11 -m venv failed. Is Python 3.11 installed? (py -0 lists versions)" }
    Write-Ok "venv created"
}

$pyVer = (& $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ($pyVer -ne "3.11") { Write-Warn2 "venv Python is $pyVer; SwarmMind targets 3.11" } else { Write-Ok "venv Python $pyVer" }

if ($SkipPip) {
    Write-Step "skipping pip (-SkipPip)"
} else {
    $req = Join-Path $RepoDir "requirements.txt"
    if ($Wheels -eq "") {
        $defaultWheels = Join-Path $RepoDir "wheels"
        if (Test-Path $defaultWheels) { $Wheels = $defaultWheels }
    }
    if ($Wheels -ne "") {
        if (-not (Test-Path $Wheels)) { throw "wheels folder not found: $Wheels" }
        Write-Step "pip install (offline) from $Wheels"
        & $VenvPy -m pip install --no-index --find-links $Wheels -r $req
    } else {
        Write-Step "pip install from PyPI (needs network; use -Wheels for offline)"
        & $VenvPy -m pip install -r $req
    }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    Write-Ok "Python packages installed"
}

# ---------------------------------------------------------------------------
# 2. Ollama (local only, never pull)
# ---------------------------------------------------------------------------
Write-Step "checking Ollama at $OllamaUrl"
$tags = $null
try {
    $tags = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -Method Get -TimeoutSec 3
} catch {
    $tags = $null
}
if ($null -eq $tags) {
    Write-Warn2 "Ollama is not reachable at $OllamaUrl. Start the Ollama app (or run 'ollama serve') and re-run."
} else {
    $names = @()
    if ($null -ne $tags.models) { $names = @($tags.models | ForEach-Object { $_.name }) }
    $found = $false
    foreach ($n in $names) {
        if ($n -eq $Model -or $n -eq "$Model`:latest") { $found = $true }
    }
    if ($found) {
        Write-Ok "Ollama is up and $Model is installed"
    } else {
        Write-Warn2 "Ollama is up but $Model is not listed (have: $($names -join ', ')). This script never pulls models; import it from a local copy."
    }
}
# Each node calls only its own Ollama on 127.0.0.1, so OLLAMA_HOST does not need to change.
$ollamaHost = [Environment]::GetEnvironmentVariable("OLLAMA_HOST", "User")
if ($null -ne $ollamaHost -and $ollamaHost -ne "") {
    Write-Step "note: OLLAMA_HOST=$ollamaHost (user env). Not needed by SwarmMind: peers never call another node's Ollama."
}

# ---------------------------------------------------------------------------
# 3. Firewall rules (admin)
# ---------------------------------------------------------------------------
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

$rules = @(
    @{ Name = "SwarmMind heartbeat UDP 50000"; Protocol = "UDP"; Port = 50000 },
    @{ Name = "SwarmMind node TCP 8700";       Protocol = "TCP"; Port = 8700 },
    @{ Name = "SwarmMind uplink sink TCP 8800"; Protocol = "TCP"; Port = 8800 }
)

if ($SkipFirewall) {
    Write-Step "skipping firewall (-SkipFirewall)"
} elseif (-not $isAdmin) {
    Write-Warn2 "not running as Administrator: firewall rules not added. Re-run from an elevated PowerShell, or add inbound rules for UDP 50000, TCP 8700, TCP 8800 by hand."
} else {
    foreach ($r in $rules) {
        $existing = Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            Write-Ok "firewall rule exists: $($r.Name)"
        } else {
            New-NetFirewallRule -DisplayName $r.Name -Direction Inbound -Action Allow `
                -Protocol $r.Protocol -LocalPort $r.Port -Profile Any | Out-Null
            Write-Ok "firewall rule added: $($r.Name)"
        }
    }
    # A "Block" rule for python.exe (created when someone clicked Cancel on the Windows
    # firewall prompt) overrides the Allow rules above.
    $blocks = Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue |
        Where-Object { $_.Program -like "*python*.exe" } |
        Get-NetFirewallRule -ErrorAction SilentlyContinue |
        Where-Object { $_.Direction -eq "Inbound" -and $_.Action -eq "Block" -and $_.Enabled -eq "True" }
    foreach ($b in @($blocks)) {
        if ($null -ne $b) {
            Write-Warn2 "inbound BLOCK rule for Python found: '$($b.DisplayName)'. Disable it: Disable-NetFirewallRule -Name '$($b.Name)'"
        }
    }
}

# ---------------------------------------------------------------------------
# 4. LAN IP + network profile
# ---------------------------------------------------------------------------
$ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" })
Write-Host ""
if ($ips.Count -eq 0) {
    Write-Warn2 "no LAN IPv4 address found. Connect to the swarm network (router, Ethernet or offline hotspot)."
} else {
    foreach ($ip in $ips) {
        Write-Host ("  LAN IP: {0,-16} interface: {1}" -f $ip.IPAddress, $ip.InterfaceAlias) -ForegroundColor White
    }
    Write-Host "  Put '<LAN IP>:8700' in configs/pi.yaml static_peers." -ForegroundColor White
}
$profiles = @(Get-NetConnectionProfile -ErrorAction SilentlyContinue)
foreach ($p in $profiles) {
    Write-Host ("  network '{0}' ({1}): {2}" -f $p.Name, $p.InterfaceAlias, $p.NetworkCategory)
    if ("$($p.NetworkCategory)" -eq "Public") {
        Write-Step "network '$($p.Name)' is Public. The SwarmMind rules use -Profile Any, so they apply; other apps' rules may not. To switch: Set-NetConnectionProfile -InterfaceAlias '$($p.InterfaceAlias)' -NetworkCategory Private"
    }
}

Write-Host ""
Write-Step "run the node:      .venv\Scripts\python -m swarmmind.agent --config configs/laptop.yaml"
Write-Step "run the simulator: .venv\Scripts\python scripts\simulate.py"
Write-Step "dashboard:         http://127.0.0.1:8700/"
if ($Warnings.Count -gt 0) {
    Write-Host ""
    Write-Host "Finished with $($Warnings.Count) warning(s):" -ForegroundColor Yellow
    foreach ($w in $Warnings) { Write-Host "  - $w" -ForegroundColor Yellow }
} else {
    Write-Ok "setup complete"
}
