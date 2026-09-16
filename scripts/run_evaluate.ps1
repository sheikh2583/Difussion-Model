<#
.SYNOPSIS
    Evaluate a trained checkpoint without retraining.

.DESCRIPTION
    Activates the project virtual environment, sets PYTHONPATH, and
    launches evaluate.py to run sampling + FID/IS evaluation on an
    existing checkpoint.  Useful for re-evaluating at different NFE
    values or regenerating plots without retraining.

.PARAMETER Algorithm
    Algorithm the checkpoint belongs to: mock | fm | fm_lognorm | mf
    Must match the algorithm that was used to produce the checkpoint.

.PARAMETER Checkpoint
    Path to the .pt checkpoint file, e.g.:
        results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt

.PARAMETER Config
    Path to the config.json saved alongside the checkpoint, e.g.:
        results/fm_cifar10/config.json

.PARAMETER MakePlots
    If supplied, generate all standard evaluation plots after evaluation
    and save them to results/<experiment>/metrics/plots/.

.EXAMPLE
    .\scripts\run_evaluate.ps1 `
        -Algorithm fm `
        -Checkpoint results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt `
        -Config     results/fm_cifar10/config.json

.EXAMPLE
    .\scripts\run_evaluate.ps1 `
        -Algorithm mf `
        -Checkpoint results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch100.pt `
        -Config     results/mf_cifar10/config.json `
        -MakePlots
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("mock", "fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow")]
    [string]$Algorithm,

    [Parameter(Mandatory)]
    [string]$Checkpoint,

    [Parameter(Mandatory)]
    [string]$Config,

    [switch]$MakePlots
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found at '$VenvActivate'."
}
. $VenvActivate
$env:PYTHONPATH = $ProjectRoot

$Args = @("evaluate.py",
          "--algorithm",  $Algorithm,
          "--checkpoint", $Checkpoint,
          "--config",     $Config)
if ($MakePlots) { $Args += "--make-plots" }

Write-Host "=== DiffusionProject Evaluation ===" -ForegroundColor Cyan
Write-Host "Algorithm  : $Algorithm"
Write-Host "Checkpoint : $Checkpoint"
Write-Host "Config     : $Config"
Write-Host "Command    : python $($Args -join ' ')"
Write-Host ""

Set-Location $ProjectRoot
python @Args
