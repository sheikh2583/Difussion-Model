<#
.SYNOPSIS
    Train a generative model on CIFAR-10.

.DESCRIPTION
    Activates the project virtual environment, sets PYTHONPATH to the
    project root, and launches train.py with the supplied arguments.

.PARAMETER Algorithm
    Algorithm to train. One of: mock | fm | fm_lognorm | mf
    Default: fm

.PARAMETER Config
    Path to a JSON config file. If omitted, built-in defaults are used.
    Preset configs live under config/:
        config/smoke_fast.json          -- quick smoke test (few epochs)
        config/fm_full.json             -- full FM run
        config/fm_lognorm.json          -- FM + logit-normal time sampling
        config/fm_lognorm_rtx3060.json  -- FM-LN tuned for RTX 3060
        config/mf_full.json             -- full Mean Flow run

.PARAMETER ExperimentName
    Override the experiment_name field in the config (used as the output
    sub-directory under results/). Optional.

.PARAMETER Epochs
    Override the epoch count from the config. Optional.

.EXAMPLE
    # Quick smoke test (uses defaults, very fast):
    .\scripts\run_train.ps1 -Algorithm mock

.EXAMPLE
    # Full Flow Matching run with a preset config:
    .\scripts\run_train.ps1 -Algorithm fm -Config config/fm_full.json

.EXAMPLE
    # Mean Flow, custom name and epoch count:
    .\scripts\run_train.ps1 -Algorithm mf -Config config/mf_full.json `
        -ExperimentName mf_run2 -Epochs 200
#>

[CmdletBinding()]
param(
    [ValidateSet("mock", "fm", "fm_lognorm", "mf")]
    [string]$Algorithm = "fm",

    [string]$Config = "",

    [string]$ExperimentName = "",

    [int]$Epochs = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve the project root (one level up from scripts/)
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# ---------------------------------------------------------------------------
# Activate the virtual environment
# ---------------------------------------------------------------------------
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found at '$VenvActivate'. " +
                "Create it with: python -m venv venv && pip install -r requirements.txt"
}
. $VenvActivate

# ---------------------------------------------------------------------------
# Set PYTHONPATH so all sub-packages resolve correctly
# ---------------------------------------------------------------------------
$env:PYTHONPATH = $ProjectRoot

# ---------------------------------------------------------------------------
# Build the argument list
# ---------------------------------------------------------------------------
$Args = @("train.py", "--algorithm", $Algorithm)

if ($Config -ne "") {
    $Args += @("--config", $Config)
}
if ($ExperimentName -ne "") {
    $Args += @("--experiment-name", $ExperimentName)
}
if ($Epochs -gt 0) {
    $Args += @("--epochs", $Epochs)
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
Write-Host "=== DiffusionProject Training ===" -ForegroundColor Cyan
Write-Host "Algorithm : $Algorithm"
Write-Host "Config    : $(if ($Config) { $Config } else { '(defaults)' })"
Write-Host "Command   : python $($Args -join ' ')"
Write-Host ""

Set-Location $ProjectRoot
python @Args
