# Train the supported algorithm suite in dependency order.
# Usage: .\scripts\train_all.ps1 [-Dataset cifar10|celeba] [-Only ALGORITHM]
#                                  [-SkipReflow] [-DryRun]
[CmdletBinding()]
param(
    [switch]$SkipFm,
    [switch]$SkipFmLognorm,
    [switch]$SkipMf,
    [switch]$SkipMfDistill,
    [switch]$SkipConsistency,
    [switch]$SkipReflow,
    [ValidateSet("", "fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow")]
    [string]$Only = "",
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

if ($Dataset -eq "celeba") {
    if ($Only -and $Only -notin @("fm", "mf")) {
        throw "CelebA currently supports only fm and mf presets."
    }
    $SkipFmLognorm = $true
    $SkipMfDistill = $true
    $SkipConsistency = $true
    $SkipReflow = $true
}

$script:Failed = $false

function Invoke-Training {
    param([string]$Algorithm, [string]$Config, [bool]$Skip)
    if ($Skip -or ($Only -and $Only -ne $Algorithm)) { return }
    if (-not (Test-Path -LiteralPath $Config)) {
        Write-Error "[MISSING] config: $Config"
        $script:Failed = $true
        return
    }
    Write-Host "[PLAN] $Python train.py --algorithm $Algorithm --config $Config"
    if (-not $DryRun) {
        & $Python train.py --algorithm $Algorithm --config $Config
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $Algorithm" }
    }
}

function Test-Prerequisite {
    param([string]$Path, [string]$Purpose)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Warning "[BLOCKED] ${Purpose}: $Path"
        $script:Failed = $true
        return $false
    }
    Write-Host "[OK] ${Purpose}: $Path"
    return $true
}

if ($Dataset -eq "celeba") {
    Invoke-Training "fm" "config/fm_celeba64.json" $SkipFm
    Invoke-Training "mf" "config/mf_celeba64.json" $SkipMf
} else {
    $FmCheckpoint = "results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt"
    Invoke-Training "fm" "config/fm_full.json" $SkipFm
    Invoke-Training "fm_lognorm" "config/fm_lognorm_full.json" $SkipFmLognorm
    Invoke-Training "mf" "config/mf_full.json" $SkipMf

    if (-not $SkipMfDistill -and (-not $Only -or $Only -eq "mf_distill")) {
        $Ready = Test-Prerequisite $FmCheckpoint "FM teacher checkpoint"
        if ($Ready -or $DryRun) {
            Invoke-Training "mf_distill" "config/mf_distill_full.json" $false
        }
    }
    if (-not $SkipConsistency -and (-not $Only -or $Only -eq "consistency")) {
        $Ready = Test-Prerequisite $FmCheckpoint "FM teacher checkpoint"
        if ($Ready -or $DryRun) {
            Invoke-Training "consistency" "config/consistency_full.json" $false
        }
    }
    if (-not $SkipReflow -and (-not $Only -or $Only -eq "reflow")) {
        $Ready = Test-Prerequisite "data/reflow_pairs_cifar10.pt" "Reflow pairs"
        if ($Ready -or $DryRun) {
            Invoke-Training "reflow" "config/reflow_full.json" $false
        }
    }
}

if ($script:Failed) {
    throw "Workflow validation found missing prerequisites."
}
if ($DryRun) {
    Write-Host "Dry-run validation complete." -ForegroundColor Green
} else {
    Write-Host "Training workflow complete." -ForegroundColor Green
}
