<#
.SYNOPSIS
    Prepare a Windows machine for every maintained training workflow.

.DESCRIPTION
    Installs Python when necessary, creates the project virtual environment,
    installs the complete dependency set and CUDA-aware PyTorch build, fetches
    CIFAR-10 and CelebA, verifies the environment, and prepares the accepted
    CelebA codec plus latent cache. The script is idempotent and never starts
    model training.

    Codec validation is intentionally not bypassed. If the repository's
    quality gate rejects the codec, rerun with -AcceptQualityFailure and a
    meaningful -AcceptanceReason after reviewing the reported measurements.

.EXAMPLE
    .\INIT_ALL.cmd

.EXAMPLE
    .\INIT_ALL.cmd -Gpu cuda128 -BatchSize 16

.EXAMPLE
    .\INIT_ALL.cmd -AcceptQualityFailure -AcceptanceReason "Approved for the documented VQ-f4 latent experiment"

.EXAMPLE
    .\INIT_ALL.cmd -DryRun
#>

[CmdletBinding()]
param(
    [ValidateSet("cuda118", "cuda121", "cuda128", "cpu", "")]
    [string]$Gpu = "",
    [switch]$SkipTorch,
    [switch]$SkipLatentAssets,
    [ValidateRange(1, 1048576)][int]$BatchSize = 32,
    [ValidateRange(0, 1024)][int]$NumWorkers = 4,
    [switch]$AcceptQualityFailure,
    [string]$AcceptanceReason = "",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$BaseSetup = Join-Path $PSScriptRoot "setup.ps1"
$LatentSetup = Join-Path $PSScriptRoot "setup_celeba_latent.ps1"
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if ($AcceptQualityFailure -and -not $AcceptanceReason.Trim()) {
    throw "-AcceptQualityFailure requires a non-empty -AcceptanceReason."
}
if (-not $AcceptQualityFailure -and $AcceptanceReason.Trim()) {
    throw "-AcceptanceReason requires -AcceptQualityFailure."
}
if (-not (Test-Path -LiteralPath $BaseSetup)) {
    throw "Missing Windows setup script: $BaseSetup"
}
if (-not $SkipLatentAssets -and -not (Test-Path -LiteralPath $LatentSetup)) {
    throw "Missing latent setup script: $LatentSetup"
}

$baseParameters = @{
    Yes = $true
    Datasets = "all"
    SkipTorch = [bool]$SkipTorch
}
if ($Gpu) { $baseParameters.Gpu = $Gpu }

$latentParameters = @{
    BatchSize = $BatchSize
    NumWorkers = $NumWorkers
    SkipPackages = $true
}
if ($AcceptQualityFailure) {
    $latentParameters.AcceptQualityFailure = $true
    $latentParameters.AcceptanceReason = $AcceptanceReason
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DiffusionProject - Complete Windows Initialization" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Root          : $ProjectRoot"
Write-Host "  GPU override  : $(if ($Gpu) { $Gpu } else { 'auto-detect' })"
Write-Host "  Datasets      : CIFAR-10 and CelebA"
Write-Host "  Latent assets : $(if ($SkipLatentAssets) { 'skip' } else { 'prepare' })"
Write-Host "  Dry run       : $DryRun"

Write-Host ""
Write-Host "[PLAN] $BaseSetup -Yes -Datasets all$(if ($SkipTorch) { ' -SkipTorch' })$(if ($Gpu) { " -Gpu $Gpu" })" -ForegroundColor DarkCyan
if (-not $SkipLatentAssets) {
    $latentPlan = "$LatentSetup -BatchSize $BatchSize -NumWorkers $NumWorkers -SkipPackages"
    if ($AcceptQualityFailure) {
        $latentPlan += " -AcceptQualityFailure -AcceptanceReason `"$AcceptanceReason`""
    }
    Write-Host "[PLAN] $latentPlan" -ForegroundColor DarkCyan
}
Write-Host "[PLAN] verify dependency imports, platform parity, and workflow configuration" -ForegroundColor DarkCyan

if ($DryRun) {
    Write-Host ""
    Write-Host "[DRY-RUN] No packages, datasets, or assets were changed." -ForegroundColor Yellow
    exit 0
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required by the provenance-aware training launchers. Install Git for Windows and rerun INIT_ALL.cmd."
}

& $BaseSetup @baseParameters
if ($LASTEXITCODE -ne 0) {
    throw "Base environment setup failed with exit code $LASTEXITCODE."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Setup completed without creating the expected interpreter: $Python"
}

$importProbe = @"
import accelerate, diffusers, gdown, huggingface_hub, matplotlib, numpy, PIL
import pytest, safetensors, scipy, torch, torch_fidelity, torchmetrics, torchvision
print(f'[OK] torch={torch.__version__} cuda={torch.cuda.is_available()}')
print(f'[OK] torchvision={torchvision.__version__}')
print(f'[OK] diffusers={diffusers.__version__} accelerate={accelerate.__version__}')
"@
& $Python -c $importProbe
if ($LASTEXITCODE -ne 0) {
    throw "The installed Python environment failed the training dependency import probe."
}

if (-not $SkipLatentAssets) {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "CelebA latent preparation requires an NVIDIA CUDA GPU and driver. Install the driver or rerun with -SkipLatentAssets."
    }
    & $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch cannot access CUDA. Fix the NVIDIA/PyTorch installation or rerun with -SkipLatentAssets."
    }

    . "$PSScriptRoot\workflow_guard.ps1"
    Start-WorkflowGuard -Python $Python -CommandName "init_all.ps1 latent setup" `
        -IdentityArguments @("--launcher", "scripts/windows/init_all.ps1")
    try {
        & $LatentSetup @latentParameters
        if ($LASTEXITCODE -ne 0) {
            throw "CelebA latent asset setup failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Stop-WorkflowGuard -Python $Python
    }
}

& $Python scripts/verify_platform_parity.py
if ($LASTEXITCODE -ne 0) {
    throw "Linux-to-Windows launcher parity verification failed."
}
& $Python scripts/verify_workflow.py --dataset none
if ($LASTEXITCODE -ne 0) {
    throw "Workflow configuration verification failed."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Windows initialization complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Preview commands (these do not train):"
Write-Host "  .\scripts\windows\train_all_datasets.ps1 -Dataset all -DryRun"
Write-Host "  .\scripts\windows\train_celeba_latent.ps1 -Only all -DryRun"
Write-Host ""
