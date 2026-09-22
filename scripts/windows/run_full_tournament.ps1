# Run the complete experiment tournament sequentially on one GPU.
#
# Usage:
#   .\scripts\windows\run_full_tournament.ps1 -DryRun
#   .\scripts\windows\run_full_tournament.ps1
#   .\scripts\windows\run_full_tournament.ps1 -Dataset cifar10
#   .\scripts\windows\run_full_tournament.ps1 -Dataset celeba -BatchSize 32
#   .\scripts\windows\run_full_tournament.ps1 -Dataset cifar10 -Mode fresh
[CmdletBinding()]
param(
    [ValidateSet("all", "cifar10", "celeba")]
    [string]$Dataset = "all",
    [ValidateRange(1, [int]::MaxValue)]
    [int]$Epoch = 100,
    [int]$BatchSize = 0,
    [ValidateSet("continue", "fresh")]
    [string]$Mode = "continue",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Run scripts\windows\init.cmd first."
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
$LockToken = $null
$LockOwned = $false
$SourceManifest = $null

if (-not $DryRun) {
    $AcquireArgs = @(
        "scripts/workflow_guard.py", "lock-acquire",
        "--lock-file", $LockFile, "--command", "run_full_tournament.ps1",
        "--owner-pid", "$PID"
    )
    if ($env:DIFFUSION_GPU_LOCK_TOKEN) {
        $AcquireArgs += @("--token", $env:DIFFUSION_GPU_LOCK_TOKEN)
    }
    $Lease = (& $Python @AcquireArgs).Trim().Split("`t")
    if ($LASTEXITCODE -ne 0 -or $Lease.Count -ne 2) {
        throw "Could not acquire the shared GPU lock."
    }
    $LockToken = $Lease[0]
    $LockOwned = $Lease[1] -eq "true"
    $env:DIFFUSION_GPU_LOCK_TOKEN = $LockToken
    if (-not $env:DIFFUSION_PARENT_SUITE_TIMESTAMP) {
        $env:DIFFUSION_PARENT_SUITE_TIMESTAMP = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    }
    $SourceManifest = "results/source_manifests/$($env:DIFFUSION_PARENT_SUITE_TIMESTAMP)_pid${PID}.json"
    $IdentityArgs = @(
        "scripts/workflow_guard.py", "source-freeze", "--manifest", $SourceManifest,
        "--launcher", "scripts/windows/run_full_tournament.ps1"
    )
    Get-ChildItem config -Filter "*_full.json" | ForEach-Object {
        $IdentityArgs += @("--config", $_.FullName)
    }
    Get-ChildItem config -Filter "*_celeba64.json" | ForEach-Object {
        $IdentityArgs += @("--config", $_.FullName)
    }
    $env:DIFFUSION_SOURCE_IDENTITY = (& $Python @IdentityArgs).Trim()
    $env:DIFFUSION_SOURCE_MANIFEST = $SourceManifest
    $env:DIFFUSION_LIFECYCLE_MODE = $Mode
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

    & $Python scripts/workflow_guard.py source-verify --manifest $env:DIFFUSION_SOURCE_MANIFEST | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Training source identity changed." }

    $RunCommand = {
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Step failed with exit code ${LASTEXITCODE}: $Description"
        }
    }
    & $RunCommand
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

function Resolve-Checkpoint {
    param([string]$RunName, [string]$ClassName)
    $Resolved = & $Python scripts/checkpoint_path.py `
        --run-dir "results/$RunName" --class-name $ClassName --epoch $Epoch `
        --planned-mode $Mode
    if ($LASTEXITCODE -ne 0 -or -not $Resolved) {
        throw "Could not resolve checkpoint path for $RunName ($ClassName)."
    }
    return "$Resolved".Trim()
}

function Invoke-Algorithm {
    param(
        [string]$Algorithm,
        [string]$Config,
        [string]$RunName,
        [string]$ClassName,
        [string]$Description
    )
    $Checkpoint = Resolve-Checkpoint -RunName $RunName -ClassName $ClassName
    Assert-FileReady -Path $Config -Purpose "config"
    $TrainArguments = @(
        "train.py", "--algorithm", $Algorithm, "--config", $Config,
        "--epochs", "$Epoch", "--mode", $Mode
    )
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

    $Teacher = Resolve-Checkpoint -RunName "fm_$Name" -ClassName "FlowMatchingAlgorithm"
    Assert-FileReady -Path $Teacher -Purpose "FM teacher checkpoint"
    Invoke-Algorithm "consistency" $Configs.consistency "consistency_$Name" `
        "ConsistencyAlgorithm" "[4/7] Consistency Models - $Name"
    Invoke-Algorithm "mf_distill" $Configs.mf_distill "mf_distill_$Name" `
        "MeanFlowDistillAlgorithm" "[5/7] Mean Flow Distillation - $Name"

    if ($Mode -eq "fresh" -and (Test-Path -LiteralPath $Pairs)) {
        $HistoryDir = Join-Path $ProjectRoot "data\history"
        $PairName = [System.IO.Path]::GetFileNameWithoutExtension($Pairs)
        $PairExtension = [System.IO.Path]::GetExtension($Pairs)
        $ArchivedPairs = Join-Path $HistoryDir "${PairName}_${Timestamp}_pid${PID}${PairExtension}"
        Write-Host ""
        Write-Host "[preserve] [6/7] Existing Reflow pairs: $Pairs -> $ArchivedPairs"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Force -Path $HistoryDir | Out-Null
            Move-Item -LiteralPath $Pairs -Destination $ArchivedPairs
        }
    }

    if ($Mode -eq "continue" -and (Test-Path -LiteralPath $Pairs)) {
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
    if ($Mode -eq "continue") {
        Write-Host "Continue mode: completed epoch-$Epoch checkpoints and existing pair artifacts will be skipped."
    } else {
        Write-Host "Fresh mode: existing runs are preserved by train.py and Reflow pairs are archived before regeneration."
    }

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
    if ($LockOwned -and $LockToken) {
        & $Python scripts/workflow_guard.py lock-release `
            --lock-file $LockFile --token $LockToken | Out-Null
    }
}
