# Train the supported algorithm suite in dependency order.
# Usage: .\scripts\windows\train_all.ps1 [-Dataset cifar10|celeba] [-Only ALGORITHM]
#                                  [-SkipReflow] [-Mode continue|fresh] [-DryRun]
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
    [ValidateSet("continue", "fresh")]
    [string]$Mode = "continue",
    [ValidateSet("current", "legacy")]
    [string]$CifarBackbone = "current",
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Run .\scripts\windows\setup.ps1 -Yes first."
}
if ($Dataset -eq "celeba" -and $CifarBackbone -ne "current") {
    throw "-CifarBackbone only applies when -Dataset cifar10."
}

if ($Dataset -eq "celeba") {
    $FmConfig = "config/fm_celeba64.json"
    $FmLognormConfig = "config/fm_lognorm_celeba64.json"
    $MfConfig = "config/mf_celeba64.json"
    $MfDistillConfig = "config/mf_distill_celeba64.json"
    $ConsistencyConfig = "config/consistency_celeba64.json"
    $ReflowConfig = "config/reflow_celeba64.json"
    $FmRunDir = "results/fm_celeba"
    $ReflowPairs = "data/reflow_pairs_celeba.pt"
} else {
    if ($CifarBackbone -eq "legacy") {
        $FmConfig = "config/cifar_legacy/fm.json"
        $FmLognormConfig = "config/cifar_legacy/fm_lognorm.json"
        $MfConfig = "config/cifar_legacy/mf.json"
        $MfDistillConfig = "config/cifar_legacy/mf_distill.json"
        $ConsistencyConfig = "config/cifar_legacy/consistency.json"
        $ReflowConfig = "config/cifar_legacy/reflow.json"
        $FmRunDir = "results/fm_legacy_backbone_cifar10"
        $ReflowPairs = "data/reflow_pairs_cifar10_legacy_backbone.pt"
    } else {
        $FmConfig = "config/fm_full.json"
        $FmLognormConfig = "config/fm_lognorm_full.json"
        $MfConfig = "config/mf_full.json"
        $MfDistillConfig = "config/mf_distill_full.json"
        $ConsistencyConfig = "config/consistency_full.json"
        $ReflowConfig = "config/reflow_full.json"
        $FmRunDir = "results/fm_cifar10"
        $ReflowPairs = "data/reflow_pairs_cifar10.pt"
    }
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
    Write-Host "[PLAN] $Python train.py --algorithm $Algorithm --config $Config --mode $Mode"
    if (-not $DryRun) {
        & $Python train.py --algorithm $Algorithm --config $Config --mode $Mode
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $Algorithm" }
    }
}

function Test-Prerequisite {
    param([string]$Path, [string]$Purpose)
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($DryRun) {
            Write-Host "[PLANNED] ${Purpose}: $Path"
            return $true
        }
        Write-Warning "[BLOCKED] ${Purpose}: $Path"
        $script:Failed = $true
        return $false
    }
    Write-Host "[OK] ${Purpose}: $Path"
    return $true
}

Invoke-Training "fm" $FmConfig $SkipFm
Invoke-Training "fm_lognorm" $FmLognormConfig $SkipFmLognorm
Invoke-Training "mf" $MfConfig $SkipMf
$FmCheckpoint = & $Python scripts/checkpoint_path.py --run-dir $FmRunDir `
    --class-name FlowMatchingAlgorithm --epoch 100 --planned-mode $Mode
if ($LASTEXITCODE -ne 0 -or -not $FmCheckpoint) {
    throw "Could not resolve the FM teacher checkpoint path."
}
$FmCheckpoint = "$FmCheckpoint".Trim()

if (-not $SkipMfDistill -and (-not $Only -or $Only -eq "mf_distill")) {
    $Ready = Test-Prerequisite $FmCheckpoint "FM teacher checkpoint"
    if ($Ready -or $DryRun) {
        Invoke-Training "mf_distill" $MfDistillConfig $false
    }
}
if (-not $SkipConsistency -and (-not $Only -or $Only -eq "consistency")) {
    $Ready = Test-Prerequisite $FmCheckpoint "FM teacher checkpoint"
    if ($Ready -or $DryRun) {
        Invoke-Training "consistency" $ConsistencyConfig $false
    }
}
if (-not $SkipReflow -and (-not $Only -or $Only -eq "reflow")) {
    if ($Mode -eq "fresh" -and (Test-Path -LiteralPath $ReflowPairs)) {
        $HistoryDir = Join-Path $ProjectRoot "data\history"
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $PairName = [System.IO.Path]::GetFileNameWithoutExtension($ReflowPairs)
        $PairExtension = [System.IO.Path]::GetExtension($ReflowPairs)
        $ArchivedPairs = Join-Path $HistoryDir "${PairName}_${Timestamp}_pid${PID}${PairExtension}"
        Write-Host "[PLAN] Preserve existing Reflow pairs: $ReflowPairs -> $ArchivedPairs"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Force -Path $HistoryDir | Out-Null
            Move-Item -LiteralPath $ReflowPairs -Destination $ArchivedPairs
        }
    }
    if ($Mode -eq "fresh" -or -not (Test-Path -LiteralPath $ReflowPairs)) {
        $TeacherReady = Test-Prerequisite $FmCheckpoint "FM teacher checkpoint"
        if ($TeacherReady -or $DryRun) {
            Write-Host "[PLAN] Generate Reflow pairs: $ReflowPairs"
            if (-not $DryRun) {
                & $Python scripts/generate_reflow_pairs.py `
                    --checkpoint $FmCheckpoint --config $FmConfig `
                    --n-pairs 50000 --nfe 50 --output $ReflowPairs
                if ($LASTEXITCODE -ne 0) { throw "Reflow pair generation failed" }
            }
        }
    }
    $Ready = Test-Prerequisite $ReflowPairs "Reflow pairs"
    if ($Ready) { Invoke-Training "reflow" $ReflowConfig $false }
}

if ($script:Failed) {
    throw "Workflow validation found missing prerequisites."
}
if ($DryRun) {
    Write-Host "Dry-run validation complete." -ForegroundColor Green
} else {
    Write-Host "Training workflow complete." -ForegroundColor Green
}
