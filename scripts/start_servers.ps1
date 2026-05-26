# scripts\start_servers.ps1
# SimpleT2M backend launcher (Windows / PowerShell).
#   fused_server.exe  -> 127.0.0.1:8421 (hidden) / :8422 (simpletool)
#   t2m_infer.exe     -> 127.0.0.1:8423
# Ctrl-C to bring everything down.
#
# All paths and ports come from config.toml (project root by default).
# Override the config file with:
#   $env:CONFIG = "C:\path\to\other.toml"; .\scripts\start_servers.ps1

$ErrorActionPreference = "Stop"

# ---------- Locate project root ----------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $ScriptDir
Set-Location $Root

$Config = if ($env:CONFIG) { $env:CONFIG } else { Join-Path $Root "config.toml" }
if (-not (Test-Path $Config)) {
    Write-Host "[start_servers] config not found: $Config" -ForegroundColor Red
    exit 1
}
$Config    = (Resolve-Path $Config).Path
$ConfigDir = Split-Path -Parent $Config

# ---------- mini TOML reader ----------
# Reads scalar [section].key from the config. Strips quotes and inline
# comments. NOT a real parser (no arrays, no nesting) — enough for paths
# and ports. Anything more complex stays inside C++ binaries that use toml++.
function Get-TomlValue($section, $key) {
    $inSect = $false
    $sectHeader = "[$section]"
    foreach ($line in Get-Content -LiteralPath $Config) {
        $t = $line.Trim()
        if ($t -match '^\[.*\]$') { $inSect = ($t -eq $sectHeader); continue }
        if (-not $inSect) { continue }
        if ($t -match "^\s*$([regex]::Escape($key))\s*=\s*(.*)$") {
            $v = $matches[1]
            # strip inline comment (only when '#' is outside quotes — naive, good enough)
            if ($v -notmatch '^".*"$' -and $v -notmatch "^'.*'$") {
                $v = ($v -split '#', 2)[0]
            }
            $v = $v.Trim()
            # strip surrounding quotes
            if ($v -match '^"(.*)"$') { $v = $matches[1] }
            elseif ($v -match "^'(.*)'$") { $v = $matches[1] }
            return $v
        }
    }
    return ""
}

# Resolve a path relative to ConfigDir (matches AppConfig::resolve in C++).
# Returns "" for empty input so callers can detect a missing config key.
function Resolve-ConfigPath($p) {
    if ([string]::IsNullOrEmpty($p)) { return "" }
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return [System.IO.Path]::GetFullPath((Join-Path $ConfigDir $p))
}

# ---------- Pull config ----------
$FusedModel          = Resolve-ConfigPath (Get-TomlValue "paths"        "llama_model")
$FusedPortHidden     = [int](Get-TomlValue "fused_server" "port_hidden")
$FusedPortSimpletool = [int](Get-TomlValue "fused_server" "port_simpletool")
$FusedLayer          = [int](Get-TomlValue "fused_server" "hidden_layer")

$T2mOnnxDir          = Resolve-ConfigPath (Get-TomlValue "paths"     "t2m_onnx_dir")
$T2mPort             = [int](Get-TomlValue "t2m_infer" "server_port")

$LogDir              = Resolve-ConfigPath (Get-TomlValue "paths" "logs_dir")
if ([string]::IsNullOrEmpty($LogDir)) { $LogDir = Join-Path $Root "logs" }

# Windows binary locations are hardcoded. MSVC puts everything under
# build\Release\ alongside the DLLs (ggml*, llama*, onnxruntime*) — running
# the exe from anywhere else would fail with STATUS_DLL_NOT_FOUND. The
# `binary` keys in config.toml are POSIX paths used by the Linux/Mac
# launcher (scripts/start_servers.sh) and are ignored here on purpose.
$FusedBin            = Join-Path $Root "build\Release\fused_server.exe"
$T2mBin              = Join-Path $Root "build\Release\t2m_infer.exe"

# ---------- Args ----------
$SkipFused = $false; $SkipT2m = $false; $TailLog = $false
foreach ($a in $args) {
    switch ($a) {
        "--skip-fused" { $SkipFused = $true }
        "--skip-t2m"   { $SkipT2m   = $true }
        "--tail"       { $TailLog   = $true }
        "-h"           { Get-Content $PSCommandPath | Select-Object -First 12; exit 0 }
        "--help"       { Get-Content $PSCommandPath | Select-Object -First 12; exit 0 }
        default { Write-Error "unknown arg: $a"; exit 1 }
    }
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---------- Pretty print ----------
function Log  ($m) { Write-Host "[start_servers] $m" }
function Dim  ($m) { Write-Host "  $m" -ForegroundColor DarkGray }
function Ok   ($m) { Write-Host "[start_servers] OK  $m" -ForegroundColor Green }
function Err  ($m) { Write-Host "[start_servers] ERR $m" -ForegroundColor Red }
function Warn ($m) { Write-Host "[start_servers] !   $m" -ForegroundColor Yellow }

function Test-PortOpen($port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", [int]$port)
        $c.Close(); return $true
    } catch { return $false }
}

# Wait for a port to open, but bail out immediately if the process dies first.
# A process that exits before opening its port (e.g. missing DLL -> exit code
# -1073741515 / 0xC0000135) used to make us wait the full timeout with an empty
# log and no clue. Now we surface the exit code the moment it happens.
function Wait-ForPort($port, $name, $timeoutSec, $proc) {
    $ticks = $timeoutSec * 5
    for ($i = 0; $i -lt $ticks; $i++) {
        if ($proc -and $proc.HasExited) {
            Err "$name exited early (code $($proc.ExitCode)) before opening port $port"
            return $false
        }
        if (Test-PortOpen $port) { return $true }
        Start-Sleep -Milliseconds 200
    }
    return $false
}

# ---------- Preflight ----------
# Use [string]::IsNullOrEmpty guards before every Test-Path so that a missing
# config key surfaces as a clean "config value missing" error instead of a
# noisy PowerShell "cannot bind null to Path" stack trace.
$fail = $false
if (-not $SkipFused) {
    if (-not (Test-Path $FusedBin)) {
        Err "fused_server bin missing: $FusedBin"; $fail = $true
    }
    if ([string]::IsNullOrEmpty($FusedModel)) {
        Err "config value missing: [paths] llama_model in $Config"; $fail = $true
    } elseif (-not (Test-Path $FusedModel)) {
        Err "fused_server model missing: $FusedModel"; $fail = $true
    }
    if (Test-PortOpen $FusedPortHidden)     { Err "port $FusedPortHidden in use (hidden)"; $fail = $true }
    if (Test-PortOpen $FusedPortSimpletool) { Err "port $FusedPortSimpletool in use (simpletool)"; $fail = $true }
}
if (-not $SkipT2m) {
    if (-not (Test-Path $T2mBin)) {
        Err "t2m_infer bin missing: $T2mBin"; $fail = $true
    }
    if ([string]::IsNullOrEmpty($T2mOnnxDir)) {
        Err "config value missing: [paths] t2m_onnx_dir in $Config"; $fail = $true
    } elseif (-not (Test-Path $T2mOnnxDir)) {
        Err "t2m onnx dir missing: $T2mOnnxDir"; $fail = $true
    }
    if (Test-PortOpen $T2mPort) { Err "port $T2mPort in use (t2m)"; $fail = $true }
}
if ($fail) { Err "preflight failed"; exit 1 }

$global:Procs = @()

function Cleanup {
    Log "shutting down..."
    foreach ($p in $global:Procs) {
        if ($p -and -not $p.HasExited) {
            try { $p.CloseMainWindow() | Out-Null } catch {}
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
    Remove-Item "$LogDir\fused_server.pid","$LogDir\t2m_infer.pid" -ErrorAction SilentlyContinue
    Ok "all servers stopped"
}

# ---------- Launch fused_server ----------
function Start-Fused {
    Log "starting fused_server..."
    Dim "bin:    $FusedBin"
    Dim "model:  $FusedModel"
    Dim "ports:  hidden=$FusedPortHidden  simpletool=$FusedPortSimpletool  layer=$FusedLayer"
    Dim "log:    $LogDir\fused_server.log"

    $argList = @($FusedModel,
        "--config",          $Config,
        "--layer",           $FusedLayer,
        "--port-hidden",     $FusedPortHidden,
        "--port-simpletool", $FusedPortSimpletool)
    $p = Start-Process -FilePath $FusedBin -ArgumentList $argList `
            -WorkingDirectory (Split-Path -Parent $FusedBin) `
            -RedirectStandardOutput "$LogDir\fused_server.log" `
            -RedirectStandardError  "$LogDir\fused_server.err.log" `
            -NoNewWindow -PassThru
    $global:Procs += $p
    $p.Id | Out-File "$LogDir\fused_server.pid"
    Dim "pid:    $($p.Id)"

    if (Wait-ForPort $FusedPortHidden "fused_server" 60 $p) {
        Ok "fused_server ready (hidden:$FusedPortHidden, simpletool:$FusedPortSimpletool)"
    } else {
        Err "fused_server failed to open port $FusedPortHidden within 60s"
        Get-Content "$LogDir\fused_server.log"     -Tail 30 -ErrorAction SilentlyContinue
        Get-Content "$LogDir\fused_server.err.log" -Tail 30 -ErrorAction SilentlyContinue
        Cleanup; exit 1
    }
}

# ---------- Launch t2m_infer ----------
function Start-T2m {
    Log "starting t2m_infer..."
    Dim "bin:      $T2mBin"
    Dim "onnx_dir: $T2mOnnxDir"
    Dim "port:     $T2mPort"
    Dim "log:      $LogDir\t2m_infer.log"

    # No PATH juggling needed: CMake POST_BUILD copies onnxruntime*.dll
    # (incl. providers_cuda.dll for the GPU build) next to t2m_infer.exe.
    $argList = @("--server", "--config", $Config)
    $p = Start-Process -FilePath $T2mBin -ArgumentList $argList `
            -WorkingDirectory (Split-Path -Parent $T2mBin) `
            -RedirectStandardOutput "$LogDir\t2m_infer.log" `
            -RedirectStandardError  "$LogDir\t2m_infer.err.log" `
            -NoNewWindow -PassThru
    $global:Procs += $p
    $p.Id | Out-File "$LogDir\t2m_infer.pid"
    Dim "pid:      $($p.Id)"

    if (Wait-ForPort $T2mPort "t2m_infer" 30 $p) {
        Ok "t2m_infer ready (port $T2mPort)"
    } else {
        Err "t2m_infer failed to open port $T2mPort within 30s"
        Get-Content "$LogDir\t2m_infer.log"     -Tail 30 -ErrorAction SilentlyContinue
        Get-Content "$LogDir\t2m_infer.err.log" -Tail 30 -ErrorAction SilentlyContinue
        Cleanup; exit 1
    }
}

# ---------- Main ----------
try {
    Log "platform: windows"
    Log "config:   $Config"
    if (-not $SkipFused) { Start-Fused }
    if (-not $SkipT2m)   { Start-T2m }

    Log "all servers ready. press Ctrl-C to stop."
    Log ""
    Log "to bench:"
    Dim ".\build\Release\bench_t2m.exe --config $Config"
    Log ""

    if ($TailLog) {
        Start-Job -ScriptBlock {
            param($path)
            Get-Content $path -Wait -Tail 0
        } -ArgumentList "$LogDir\t2m_infer.log" | Out-Null
    }

    while ($true) {
        foreach ($p in $global:Procs) {
            if ($p.HasExited) {
                Err "a server (pid $($p.Id)) died unexpectedly (exit $($p.ExitCode))"
                Cleanup; exit 1
            }
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Cleanup
}