<#
.SYNOPSIS
    Evaluate a trained checkpoint without retraining.

.DESCRIPTION
    Uses the project virtual-environment interpreter directly, sets PYTHONPATH,
    and launches evaluate.py to run sampling + FID/IS evaluation on an
    existing checkpoint.  Useful for re-evaluating at different NFE
    values or regenerating plots without retraining.

.PARAMETER Algorithm
    Algorithm the checkpoint belongs to: mock | fm | fm_lognorm | mf
    Must match the algorithm that was used to produce the checkpoint.

.PARAMETER Checkpoint
    Path to the .pt checkpoint file, e.g.:
        results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt

.PARAMETER Config
    Path to the config.json saved alongside the checkpoint, e.g.:
        results/fm_cifar10/config.json

.PARAMETER MakePlots
    If supplied, generate all standard evaluation plots after evaluation
    and save them to results/<experiment>/metrics/plots/.

.EXAMPLE
    .\scripts\windows\run_evaluate.ps1 `
        -Algorithm fm `
        -Checkpoint results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt `
        -Config     results/fm_cifar10/config.json

.EXAMPLE
    .\scripts\windows\run_evaluate.ps1 `
        -Algorithm mf `
        -Checkpoint results/mf_cifar10/checkpoints/run_1/MeanFlowAlgorithm_epoch100.pt `
        -Config     results/mf_cifar10/config.json `
        -MakePlots
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("mock", "fm", "fm_lognorm", "mf", "mf_hutchinson", "mf_distill", "consistency", "reflow")]
    [string]$Algorithm,

    [Parameter(Mandatory)]
    [string]$Checkpoint,

    [Parameter(Mandatory)]
    [string]$Config,

    [switch]$MakePlots
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project interpreter not found at '$Python'. Run scripts\windows\init.cmd first."
}
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
Write-Host "Command    : $Python $($Args -join ' ')"
Write-Host ""

Set-Location $ProjectRoot
& $Python @Args
exit $LASTEXITCODE
