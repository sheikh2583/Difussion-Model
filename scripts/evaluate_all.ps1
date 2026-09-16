# Evaluate available epoch checkpoints for the six research algorithms.
# Usage: .\scripts\evaluate_all.ps1 [-Dataset cifar10|celeba] [-Epoch 100] [-DryRun]
[CmdletBinding()]
param(
    [ValidateRange(1, [int]::MaxValue)]
    [int]$Epoch = 100,
    [ValidateSet("cifar10", "celeba")]
    [string]$Dataset = "cifar10",
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

if ($Dataset -eq "celeba") {
    $Suffix = "celeba"
    $ConfigSuffix = "celeba64"
} else {
    $Suffix = "cifar10"
    $ConfigSuffix = "full"
}

Invoke-Evaluation "fm" "results/fm_$Suffix/checkpoints/FlowMatchingAlgorithm_epoch$Epoch.pt" "config/fm_$ConfigSuffix.json"
Invoke-Evaluation "fm_lognorm" "results/fm_lognorm_$Suffix/checkpoints/FlowMatchingLognormAlgorithm_epoch$Epoch.pt" "config/fm_lognorm_$ConfigSuffix.json"
Invoke-Evaluation "mf" "results/mf_$Suffix/checkpoints/MeanFlowAlgorithm_epoch$Epoch.pt" "config/mf_$ConfigSuffix.json"
Invoke-Evaluation "mf_distill" "results/mf_distill_$Suffix/checkpoints/MeanFlowDistillAlgorithm_epoch$Epoch.pt" "config/mf_distill_$ConfigSuffix.json"
Invoke-Evaluation "consistency" "results/consistency_$Suffix/checkpoints/ConsistencyAlgorithm_epoch$Epoch.pt" "config/consistency_$ConfigSuffix.json"
Invoke-Evaluation "reflow" "results/reflow_$Suffix/checkpoints/ReflowAlgorithm_epoch$Epoch.pt" "config/reflow_$ConfigSuffix.json"

if ($script:Available -eq 0) {
    throw "No epoch-$Epoch checkpoints were found."
}
Write-Host "Evaluation workflow found $script:Available checkpoint(s)." -ForegroundColor Green
