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
    param([string]$Algorithm, [string]$RunDir, [string]$ClassName, [string]$Config)
    $Checkpoint = & $Python scripts/checkpoint_path.py --run-dir $RunDir `
        --class-name $ClassName --epoch $Epoch 2>$null
    if ($LASTEXITCODE -ne 0) { $Checkpoint = $null }
    if (-not $Checkpoint -or -not (Test-Path -LiteralPath $Checkpoint)) {
        Write-Host "[SKIP] $Algorithm epoch-$Epoch checkpoint not found in: $RunDir/checkpoints"
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

Invoke-Evaluation "fm" "results/fm_$Suffix" "FlowMatchingAlgorithm" "config/fm_$ConfigSuffix.json"
Invoke-Evaluation "fm_lognorm" "results/fm_lognorm_$Suffix" "FlowMatchingLognormAlgorithm" "config/fm_lognorm_$ConfigSuffix.json"
Invoke-Evaluation "mf" "results/mf_$Suffix" "MeanFlowAlgorithm" "config/mf_$ConfigSuffix.json"
Invoke-Evaluation "mf_distill" "results/mf_distill_$Suffix" "MeanFlowDistillAlgorithm" "config/mf_distill_$ConfigSuffix.json"
Invoke-Evaluation "consistency" "results/consistency_$Suffix" "ConsistencyAlgorithm" "config/consistency_$ConfigSuffix.json"
Invoke-Evaluation "reflow" "results/reflow_$Suffix" "ReflowAlgorithm" "config/reflow_$ConfigSuffix.json"

if ($script:Available -eq 0) {
    throw "No epoch-$Epoch checkpoints were found."
}
Write-Host "Evaluation workflow found $script:Available checkpoint(s)." -ForegroundColor Green
