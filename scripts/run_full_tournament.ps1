# Run the complete experiment tournament sequentially on one GPU.
#
# Usage:
#   .\scripts\run_full_tournament.ps1 -DryRun
#   .\scripts\run_full_tournament.ps1
#   .\scripts\run_full_tournament.ps1 -Dataset cifar10
#   .\scripts\run_full_tournament.ps1 -Dataset celeba -BatchSize 32
[CmdletBinding()]
param(
    [ValidateSet("all", "cifar10", "celeba")]
    [string]$Dataset = "all",
    [ValidateRange(1, [int]::MaxValue)]
    [int]$Epoch = 100,
    [int]$BatchSize = 0,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Run INIT_ALL.cmd first."
}
if ($BatchSize -lt 0) {
    throw "-BatchSize must be 0 (use config values) or a positive integer."
}

$ResultsDir = Join-Path $ProjectRoot "results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $ResultsDir "tournament_run_${Timestamp}_pid${PID}.log"
$LockFile = Join-Path $ResultsDir ".lock"
$PowerShellExe = (Get-Process -Id $PID).Path
$TranscriptStarted = $false

function Invoke-WithGpuLock {
    param(
        [string]$Description,
        [scriptblock]$Action
    )

    $Token = "pid=$PID`ncommand=run_full_tournament.ps1`nstep=$Description`n"
    try {
        $Stream = [System.IO.File]::Open(
            $LockFile,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        throw "GPU lock already exists: $LockFile. Wait for the active job to finish, or remove it only after confirming it is stale."
    }

    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Token)
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Dispose()
        & $Action
    } finally {
        if ($null -ne $Stream) { $Stream.Dispose() }
        if (Test-Path -LiteralPath $LockFile) {
            $Current = Get-Content -LiteralPath $LockFile -Raw
            if ($Current -eq $Token) {
                Remove-Item -LiteralPath $LockFile -Force
            }
        }
    }
}

function Invoke-Step {
    param(
        [string]$Description,
        [string]$Executable,
        [string[]]$Arguments,
        [switch]$UseGpuLock
    )

    Write-Host ""
    Write-Host "=============================================================="
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Description"
    Write-Host "  $Executable $($Arguments -join ' ')"
    Write-Host "=============================================================="
    if ($DryRun) {
        Write-Host "  (dry-run, not executing)"
        return
    }

    $RunCommand = {
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Step failed with exit code ${LASTEXITCODE}: $Description"
        }
    }
    if ($UseGpuLock) {
        Invoke-WithGpuLock -Description $Description -Action $RunCommand
    } else {
        & $RunCommand
    }
}

function Assert-FileReady {
    param([string]$Path, [string]$Purpose)
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($DryRun) {
            Write-Host "[planned prerequisite] ${Purpose}: $Path"
            return
        }
        throw "Missing ${Purpose}: $Path"
    }
}

function Invoke-Algorithm {
    param(
        [string]$Algorithm,
        [string]$Config,
        [string]$RunName,
        [string]$ClassName,
        [string]$Description
    )
    $Checkpoint = "results/$RunName/checkpoints/${ClassName}_epoch$Epoch.pt"
    if (Test-Path -LiteralPath $Checkpoint) {
        Write-Host ""
        Write-Host "[skip] $Description - checkpoint already exists: $Checkpoint"
        return
    }
    Assert-FileReady -Path $Config -Purpose "config"
    $TrainArguments = @("train.py", "--algorithm", $Algorithm, "--config", $Config)
    if ($BatchSize -gt 0) {
        $TrainArguments += @("--batch-size", "$BatchSize")
    }
    Invoke-Step -Description $Description -Executable $Python -UseGpuLock `
        -Arguments $TrainArguments
}

function Invoke-DatasetTournament {
    param([string]$Name)

    if ($Name -eq "celeba") {
        $Configs = @{
            fm = "config/fm_celeba64.json"
            fm_lognorm = "config/fm_lognorm_celeba64.json"
            mf = "config/mf_celeba64.json"
            consistency = "config/consistency_celeba64.json"
            mf_distill = "config/mf_distill_celeba64.json"
            reflow = "config/reflow_celeba64.json"
        }
    } else {
        $Configs = @{
            fm = "config/fm_full.json"
            fm_lognorm = "config/fm_lognorm_full.json"
            mf = "config/mf_full.json"
            consistency = "config/consistency_full.json"
            mf_distill = "config/mf_distill_full.json"
            reflow = "config/reflow_full.json"
        }
    }

    $Teacher = "results/fm_$Name/checkpoints/FlowMatchingAlgorithm_epoch$Epoch.pt"
    $Pairs = "data/reflow_pairs_$Name.pt"

    Write-Host ""
    Write-Host "##############################################################"
    Write-Host "# DATASET: $Name"
    Write-Host "##############################################################"

    Invoke-Algorithm "fm" $Configs.fm "fm_$Name" "FlowMatchingAlgorithm" `
        "[1/7] Flow Matching - $Name"
    Invoke-Algorithm "fm_lognorm" $Configs.fm_lognorm "fm_lognorm_$Name" `
        "FlowMatchingLognormAlgorithm" "[2/7] FM + Logit-Normal - $Name"
    Invoke-Algorithm "mf" $Configs.mf "mf_$Name" "MeanFlowAlgorithm" `
        "[3/7] Mean Flow - $Name"

    Assert-FileReady -Path $Teacher -Purpose "FM teacher checkpoint"
    Invoke-Algorithm "consistency" $Configs.consistency "consistency_$Name" `
        "ConsistencyAlgorithm" "[4/7] Consistency Models - $Name"
    Invoke-Algorithm "mf_distill" $Configs.mf_distill "mf_distill_$Name" `
        "MeanFlowDistillAlgorithm" "[5/7] Mean Flow Distillation - $Name"

    if (Test-Path -LiteralPath $Pairs) {
        Write-Host ""
        Write-Host "[skip] [6/7] Reflow pair generation - artifact exists: $Pairs"
    } else {
        Invoke-Step -Description "[6/7] Reflow pair generation - $Name" `
            -Executable $Python -Arguments @(
                "scripts/generate_reflow_pairs.py",
                "--checkpoint", $Teacher,
                "--config", $Configs.fm,
                "--n-pairs", "50000",
                "--nfe", "50",
                "--output", $Pairs
            )
    }

    Assert-FileReady -Path $Pairs -Purpose "Reflow pair artifact"
    Invoke-Algorithm "reflow" $Configs.reflow "reflow_$Name" "ReflowAlgorithm" `
        "[7/7] Rectified Flow Reflow - $Name"
}

try {
    Start-Transcript -LiteralPath $LogFile | Out-Null
    $TranscriptStarted = $true
    Write-Host "Logging to $LogFile"
    Write-Host "Completed epoch-$Epoch checkpoints and existing pair artifacts will be skipped."

    $Datasets = if ($Dataset -eq "all") { @("cifar10", "celeba") } else { @($Dataset) }
    foreach ($Name in $Datasets) {
        Invoke-DatasetTournament -Name $Name
    }

    foreach ($Name in $Datasets) {
        Invoke-Step -Description "Evaluate epoch-$Epoch checkpoints - $Name" `
            -Executable $PowerShellExe -UseGpuLock -Arguments @(
                "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                (Join-Path $PSScriptRoot "evaluate_all.ps1"),
                "-Dataset", $Name, "-Epoch", "$Epoch"
            )
    }
    Invoke-Step -Description "Aggregate tournament results" -Executable $Python `
        -Arguments @("scripts/aggregate_results.py")

    Write-Host ""
    Write-Host "Tournament workflow complete. Log: $LogFile" -ForegroundColor Green
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
