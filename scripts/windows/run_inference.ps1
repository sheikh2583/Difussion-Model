<#
.SYNOPSIS
    Start the local read-only thesis UI server.

.DESCRIPTION
    Uses the project virtual-environment interpreter directly, sets PYTHONPATH,
    and launches inference_server.py. Open http://127.0.0.1:8000 in a
    browser to inspect recorded results and UI reconstructions after startup.
    The server does not load checkpoints or run model inference.

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
    Validate the recorded-result catalog and exit without running a model.

.PARAMETER GenerateOutputs
    Generate real checkpoint outputs before starting the read-only UI.

.PARAMETER WaitForGpu
    With GenerateOutputs, wait for an active project GPU workflow to finish.

.EXAMPLE
    # Start with defaults (localhost:8000, results in ./results):
    .\scripts\windows\run_inference.ps1

.EXAMPLE
    # Custom port and results directory:
    .\scripts\windows\run_inference.ps1 -Port 9000 -ResultsDir D:\models\diffusion

.EXAMPLE
    # Quick catalog validation with no model execution:
    .\scripts\windows\run_inference.ps1 -SelfTest
#>

[CmdletBinding()]
param(
    [string]$Host       = "127.0.0.1",
    [int]   $Port       = 8000,
    [string]$ResultsDir = "",
    [switch]$SelfTest,
    [switch]$GenerateOutputs,
    [switch]$WaitForGpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# ---------------------------------------------------------------------------
# Resolve the project interpreter
# ---------------------------------------------------------------------------
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project interpreter not found at '$Python'. Run scripts\windows\init.cmd first."
}

# ---------------------------------------------------------------------------
# Set PYTHONPATH
# ---------------------------------------------------------------------------
$env:PYTHONPATH = $ProjectRoot

if ($GenerateOutputs) {
    $PrepareScript = Join-Path $PSScriptRoot "generate_inference_outputs.ps1"
    & $PrepareScript -Wait:$WaitForGpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

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
Write-Host "=== DiffusionProject Read-Only Thesis UI ===" -ForegroundColor Cyan
if (-not $SelfTest) {
    Write-Host "UI will be available at: http://${Host}:${Port}" -ForegroundColor Green
    Write-Host "Mode: browser reconstruction only; no checkpoint/GPU execution" -ForegroundColor DarkGray
}
Write-Host "Command: $Python $($Args -join ' ')"
Write-Host ""

Set-Location $ProjectRoot
& $Python @Args
exit $LASTEXITCODE
