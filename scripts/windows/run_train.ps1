<#
.SYNOPSIS
    Train a generative model on CIFAR-10.

.DESCRIPTION
    Uses the project virtual-environment interpreter directly, sets PYTHONPATH
    to the project root, and launches train.py with the supplied arguments.

.PARAMETER Algorithm
    Algorithm to train. One of: mock | fm | fm_lognorm | mf | mf_hutchinson |
    mf_distill | consistency | reflow
    Default: fm

.PARAMETER Config
    Path to a JSON config file. If omitted, built-in defaults are used.
    Preset configs live under config/:
        config/smoke_fast.json          -- quick smoke test (few epochs)
        config/fm_full.json             -- full FM run (CIFAR-10)
        config/fm_lognorm_full.json     -- FM + logit-normal time sampling
        config/fm_lognorm_budget.json   -- reduced batch/samples for limited VRAM
        config/mf_full.json             -- full Mean Flow run
        config/mf_distill_full.json     -- MF distillation (needs FM checkpoint)
        config/consistency_full.json    -- Consistency Models
        config/reflow_full.json         -- Rectified Flow Reflow
        config/fm_celeba64.json         -- FM on CelebA 64x64
        config/mf_celeba64.json         -- MF on CelebA 64x64
        config/mf_hutchinson_cv_celeba_latent.json -- latent-only CV-Hutchinson MF

.PARAMETER ExperimentName
    Override the experiment_name field in the config (used as the output
    sub-directory under results/). Optional.

.PARAMETER Epochs
    Override the epoch count from the config. Optional.

.EXAMPLE
    # Quick smoke test (uses defaults, very fast):
    .\scripts\windows\run_train.ps1 -Algorithm mock

.EXAMPLE
    # Full Flow Matching run with a preset config:
    .\scripts\windows\run_train.ps1 -Algorithm fm -Config config/fm_full.json

.EXAMPLE
    # Mean Flow, custom name and epoch count:
    .\scripts\windows\run_train.ps1 -Algorithm mf -Config config/mf_full.json `
        -ExperimentName mf_run2 -Epochs 200

.EXAMPLE
    # Hutchinson MF is supported only on CelebA latent space:
    .\scripts\windows\run_train.ps1 -Algorithm mf_hutchinson `
        -Config config/mf_hutchinson_cv_celeba_latent.json -Mode fresh
#>

[CmdletBinding()]
param(
    [ValidateSet("mock", "fm", "fm_lognorm", "mf", "mf_hutchinson", "mf_distill", "consistency", "reflow")]
    [string]$Algorithm = "fm",

    [string]$Config = "",

    [string]$ExperimentName = "",

    [int]$Epochs = 0,

    [ValidateRange(0, 1048576)]
    [int]$BatchSize = 0,

    [ValidateRange(0, 1000000)]
    [int]$CheckpointEvery = 0,

    [ValidateSet("continue", "fresh")]
    [string]$Mode = "continue",

    [switch]$TrainOnly,

    [string]$MachineLabel = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve the project root (one level up from scripts/)
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
# Set PYTHONPATH so all sub-packages resolve correctly
# ---------------------------------------------------------------------------
$env:PYTHONPATH = $ProjectRoot

# ---------------------------------------------------------------------------
# Build the argument list
# ---------------------------------------------------------------------------
$Args = @("train.py", "--algorithm", $Algorithm, "--mode", $Mode)

if ($Config -ne "") {
    $Args += @("--config", $Config)
}
if ($ExperimentName -ne "") {
    $Args += @("--experiment-name", $ExperimentName)
}
if ($Epochs -gt 0) {
    $Args += @("--epochs", $Epochs)
}
if ($BatchSize -gt 0) { $Args += @("--batch-size", $BatchSize) }
if ($CheckpointEvery -gt 0) { $Args += @("--checkpoint-every", $CheckpointEvery) }
if ($TrainOnly) { $Args += "--train-only" }
if ($MachineLabel) { $Args += @("--machine-label", $MachineLabel) }

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
Write-Host "=== DiffusionProject Training ===" -ForegroundColor Cyan
Write-Host "Algorithm : $Algorithm"
Write-Host "Config    : $(if ($Config) { $Config } else { '(defaults)' })"
Write-Host "Mode      : $Mode"
Write-Host "Command   : $Python $($Args -join ' ')"
Write-Host ""

Set-Location $ProjectRoot
& $Python @Args
exit $LASTEXITCODE
