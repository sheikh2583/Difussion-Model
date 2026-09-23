<#
.SYNOPSIS
    Train the MF-Hutchinson control-variate variant on CelebA latent space
    with full tracking logs.

.DESCRIPTION
    Acquires the GPU lock, freezes source identity, writes provenance headers,
    tees all stdout/stderr to a timestamped log, and writes metadata on
    completion. The log captures every epoch loss, checkpoint save, and
    evaluation result.

.PARAMETER Mode
    'fresh' archives any existing run and starts from epoch 1.
    'continue' resumes the latest checkpoint.

.PARAMETER TrainOnly
    Skip FID evaluation (useful for short probes).

.PARAMETER Epochs
    Override epoch count from the config. Default: use config value (100).

.PARAMETER CheckpointEvery
    Override checkpoint frequency. Default: use config value (10).

.PARAMETER MachineLabel
    Human-readable machine name for provenance tracking.

.PARAMETER DryRun
    Print what would be executed without running anything.

.EXAMPLE
    # Full 100-epoch run with tracking:
    .\scripts\windows\run_hutchinson_cv.ps1 -Mode fresh

.EXAMPLE
    # 15-epoch probe, no evaluation:
    .\scripts\windows\run_hutchinson_cv.ps1 -Mode fresh -Epochs 15 -TrainOnly

.EXAMPLE
    # Dry run to see what would execute:
    .\scripts\windows\run_hutchinson_cv.ps1 -Mode fresh -DryRun
#>

[CmdletBinding()]
param(
    [ValidateSet("continue", "fresh")]
    [string]$Mode = "fresh",

    [switch]$TrainOnly,

    [ValidateRange(1, 10000)]
    [int]$Epochs = 0,

    [ValidateRange(1, 1000)]
    [int]$CheckpointEvery = 0,

    [string]$MachineLabel = "",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Run scripts\windows\init.cmd first."
}

$Algorithm      = "mf_hutchinson"
$Config         = "config/mf_hutchinson_cv_celeba_latent.json"
$RunName        = "mf_hutchinson_cv_celeba_latent"
$ClassName      = "MeanFlowHutchinsonAlgorithm"
$Codec          = "results/codecs/celeba_vq_f4/accepted_codec.pt"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Config not found: $Config"
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $Codec)) {
    throw "Missing accepted codec: $Codec"
}

# ---------------------------------------------------------------------------
# Workflow guard: GPU lock + source identity
# ---------------------------------------------------------------------------
. "$PSScriptRoot\workflow_guard.ps1"
$identityArgs = @("--launcher", "scripts/windows/run_hutchinson_cv.ps1",
                   "--config", $Config)
Start-WorkflowGuard -Python $Python -CommandName "run_hutchinson_cv.ps1" `
    -DryRun:$DryRun -IdentityArguments $identityArgs

try {
    # -------------------------------------------------------------------
    # Log directory and file
    # -------------------------------------------------------------------
    $deviceLogRoot = Get-WorkflowDeviceLogDirectory -Category "latent"
    $deviceToken = ($deviceLogRoot.Replace('\', '/') -split '/')[1]
    if (-not $MachineLabel) {
        $MachineLabel = "windows-$($env:COMPUTERNAME)-$deviceToken".ToLowerInvariant()
    }
    $env:DIFFUSION_MACHINE_LABEL = $MachineLabel

    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $logDir = Join-Path $deviceLogRoot $Algorithm
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $LogFile = Join-Path $logDir "celeba_latent_hutchinson_cv_$($env:COMPUTERNAME)_${timestamp}.log"
    if (Test-Path -LiteralPath $LogFile) {
        throw "Refusing to overwrite existing log: $LogFile"
    }

    # -------------------------------------------------------------------
    # Provenance header
    # -------------------------------------------------------------------
    @(
        "[run] training_type=latent_diffusion"
        "[run] representation_space=latent"
        "[run] dataset=celeba_latent"
        "[run] algorithm=$Algorithm"
        "[run] config=$Config"
        "[run] estimator=control_variate_hutchinson"
        "[run] n_probes=4"
        "[run] fd_eps_range=0.01 -> 0.0001"
        "[run] mode=$Mode"
    ) | Set-Content -LiteralPath $LogFile

    $env:DIFFUSION_SELECTED_CONFIG_SHA256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $Config
    ).Hash.ToLowerInvariant()

    & $Python scripts/print_run_provenance.py --config $Config `
        --machine-label $MachineLabel --log-device-token $deviceToken `
        --log-path $LogFile | Add-Content -LiteralPath $LogFile

    # -------------------------------------------------------------------
    # Build the training command
    # -------------------------------------------------------------------
    $command = @(
        "train.py",
        "--algorithm", $Algorithm,
        "--config", $Config,
        "--mode", $Mode,
        "--machine-label", $MachineLabel
    )
    if ($Epochs -gt 0) { $command += @("--epochs", "$Epochs") }
    if ($CheckpointEvery -gt 0) { $command += @("--checkpoint-every", "$CheckpointEvery") }
    if ($TrainOnly) { $command += "--train-only" }

    # -------------------------------------------------------------------
    # Print and log the launch
    # -------------------------------------------------------------------
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  MF-Hutchinson Control-Variate Training" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Algorithm    : $Algorithm"
    Write-Host "  Config       : $Config"
    Write-Host "  Mode         : $Mode"
    Write-Host "  TrainOnly    : $TrainOnly"
    Write-Host "  Log          : $LogFile"
    Write-Host "  Command      : $Python $($command -join ' ')"
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would execute:" -ForegroundColor Yellow
        Write-Host "  $Python $($command -join ' ')" -ForegroundColor Yellow
        Write-Host "[DRY-RUN] Log would be written to: $LogFile" -ForegroundColor Yellow
        return
    }

    @(
        "[run] command=$Python $($command -join ' ')"
        "[run] started_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
    ) | Add-Content -LiteralPath $LogFile

    # -------------------------------------------------------------------
    # Run with Tee-Object for real-time console + file logging
    # -------------------------------------------------------------------
    Test-WorkflowSource -Python $Python
    & $Python @command 2>&1 | Tee-Object -FilePath $LogFile -Append
    $exitStatus = $LASTEXITCODE

    # -------------------------------------------------------------------
    # Post-run metadata
    # -------------------------------------------------------------------
    @(
        "[run] finished_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
        "[run] exit_status=$exitStatus"
    ) | Add-Content -LiteralPath $LogFile

    & $Python scripts/write_training_log_metadata.py --log $LogFile | Out-Null

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Run complete" -ForegroundColor Cyan
    Write-Host "  Exit status  : $exitStatus"
    Write-Host "  Log          : $LogFile"
    Write-Host "  Metadata     : ${LogFile}.meta.json"
    Write-Host "============================================================" -ForegroundColor Cyan

    if ($exitStatus -ne 0) {
        throw "Training failed with exit code $exitStatus. Check log: $LogFile"
    }
} finally {
    Stop-WorkflowGuard -Python $Python
}
