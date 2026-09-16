# Evaluate available epoch checkpoints for the six research algorithms.
# Usage: .\scripts\evaluate_all.ps1 [-Epoch 100] [-DryRun]
[CmdletBinding()]
param(
    [ValidateRange(1, [int]::MaxValue)]
    [int]$Epoch = 100,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Run .\scripts\setup.ps1 -Yes first."
}

$script:Available = 0
function Invoke-Evaluation {
    param([string]$Algorithm, [string]$Checkpoint, [string]$Config)
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        Write-Host "[SKIP] $Algorithm checkpoint not found: $Checkpoint"
        return
    }
    $script:Available += 1
    Write-Host "[PLAN] $Python evaluate.py --algorithm $Algorithm --checkpoint $Checkpoint --config $Config --make-plots"
    if (-not $DryRun) {
        & $Python evaluate.py --algorithm $Algorithm --checkpoint $Checkpoint --config $Config --make-plots
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $Algorithm" }
    }
}

Invoke-Evaluation "fm" "results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch$Epoch.pt" "config/fm_full.json"
Invoke-Evaluation "fm_lognorm" "results/fm_lognorm_cifar10/checkpoints/FlowMatchingLognormAlgorithm_epoch$Epoch.pt" "config/fm_lognorm_full.json"
Invoke-Evaluation "mf" "results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch$Epoch.pt" "config/mf_full.json"
Invoke-Evaluation "mf_distill" "results/mf_distill_cifar10/checkpoints/MeanFlowDistillAlgorithm_epoch$Epoch.pt" "config/mf_distill_full.json"
Invoke-Evaluation "consistency" "results/consistency_cifar10/checkpoints/ConsistencyAlgorithm_epoch$Epoch.pt" "config/consistency_full.json"
Invoke-Evaluation "reflow" "results/reflow_cifar10/checkpoints/ReflowAlgorithm_epoch$Epoch.pt" "config/reflow_full.json"

if ($script:Available -eq 0) {
    throw "No epoch-$Epoch checkpoints were found."
}
Write-Host "Evaluation workflow found $script:Available checkpoint(s)." -ForegroundColor Green
