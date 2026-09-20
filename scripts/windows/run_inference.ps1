<#
.SYNOPSIS
    Start the local results browser and inference UI server.

.DESCRIPTION
    Activates the project virtual environment, sets PYTHONPATH, and
    launches inference_server.py.  Open http://127.0.0.1:8000 in a
    browser to inspect results or use checkpoint inference after startup.

    The server auto-discovers trained checkpoints under the results
    directory (default: ./results).  If no checkpoints are found for
    a model it will appear greyed-out in the UI.

.PARAMETER Host
    IP address to bind the server to. Default: 127.0.0.1 (localhost only).
    Use 0.0.0.0 to expose on all interfaces (LAN / remote access).

.PARAMETER Port
    TCP port for the HTTP server. Default: 8000.

.PARAMETER ResultsDir
    Root directory where trained model results live. Each sub-directory
    inside should correspond to one experiment (e.g. results/fm_cifar10).
    Default: ./results

.PARAMETER SelfTest
    Run a single-image generation smoke test for each available model
    then exit, instead of starting the persistent server.

.EXAMPLE
    # Start with defaults (localhost:8000, results in ./results):
    .\scripts\windows\run_inference.ps1

.EXAMPLE
    # Custom port and results directory:
    .\scripts\windows\run_inference.ps1 -Port 9000 -ResultsDir D:\models\diffusion

.EXAMPLE
    # Quick smoke test — generate one image per available model and exit:
    .\scripts\windows\run_inference.ps1 -SelfTest
#>

[CmdletBinding()]
param(
    [string]$Host       = "127.0.0.1",
    [int]   $Port       = 8000,
    [string]$ResultsDir = "",
    [switch]$SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found at '$VenvActivate'."
}
. $VenvActivate

# ---------------------------------------------------------------------------
# Set PYTHONPATH
# ---------------------------------------------------------------------------
$env:PYTHONPATH = $ProjectRoot

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
$Args = @("web/inference_server.py", "--host", $Host, "--port", $Port)

if ($ResultsDir -ne "") {
    $Args += @("--results-dir", $ResultsDir)
}
if ($SelfTest) {
    $Args += "--self-test"
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
Write-Host "=== DiffusionProject Inference Server ===" -ForegroundColor Cyan
if (-not $SelfTest) {
    Write-Host "UI will be available at: http://${Host}:${Port}" -ForegroundColor Green
}
Write-Host "Command: python $($Args -join ' ')"
Write-Host ""

Set-Location $ProjectRoot
python @Args
