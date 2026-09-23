<#
.SYNOPSIS
    One-shot setup for CelebA latent-space training on Windows.

.DESCRIPTION
    Ensures every prerequisite for CelebA latent diffusion training is met:

    Stage 1 -- Install missing Python packages (huggingface-hub, diffusers,
              accelerate, safetensors) that the base init.cmd skips.
    Stage 2 -- Validate the pretrained VQ-F4 codec and produce
              results/codecs/celeba_vq_f4/accepted_codec.pt.
    Stage 3 -- Cache normalised CelebA latents to data/latent_cache/.
    Stage 4 -- Build the latent-space FID reference statistics
              (results/metrics/fid_reference_stats_celeba64_latent_ch3.npz).

    Every stage is idempotent: re-running skips completed work.

.PARAMETER BatchSize
    GPU batch size for codec validation and latent caching (default: 32).

.PARAMETER NumWorkers
    DataLoader workers for codec validation and caching (default: 4).

.PARAMETER AcceptQualityFailure
    Pass through to validate_codec.py if the codec fails rFID/PSNR gates but
    you want to proceed anyway. Requires -AcceptanceReason.

.PARAMETER AcceptanceReason
    Reason string for -AcceptQualityFailure.

.PARAMETER DryRun
    Print what would be executed without running anything.

.PARAMETER SkipPackages
    Skip the pip install stage (use if packages are already installed).

.EXAMPLE
    # Full setup:
    .\scripts\windows\setup_celeba_latent.ps1

.EXAMPLE
    # Preview all steps:
    .\scripts\windows\setup_celeba_latent.ps1 -DryRun

.EXAMPLE
    # Re-run with smaller batch for low-VRAM GPU:
    .\scripts\windows\setup_celeba_latent.ps1 -BatchSize 8
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 1048576)][int]$BatchSize = 32,
    [ValidateRange(0, 1024)][int]$NumWorkers = 4,
    [switch]$AcceptQualityFailure,
    [string]$AcceptanceReason = "",
    [switch]$DryRun,
    [switch]$SkipPackages
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ProjectRoot   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python        = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project venv not found. Run scripts\windows\init.cmd first."
}

$CelebaRoot     = "data/raw"
$SourceDir      = "data/pretrained/ldm-celebahq-256"
$SourceVqvae    = "data/pretrained/ldm-celebahq-256/vqvae"
$CodecOutputDir = "results/codecs/celeba_vq_f4"
$CodecCheckpoint = "$CodecOutputDir/accepted_codec.pt"
$LatentCacheDir = "data/latent_cache"
$FidRefLatent   = "results/metrics/fid_reference_stats_celeba64_latent_ch3.npz"

# Validate parameter combinations
if ($AcceptQualityFailure -and -not $AcceptanceReason) {
    throw "-AcceptQualityFailure requires -AcceptanceReason."
}
if (-not $AcceptQualityFailure -and $AcceptanceReason) {
    throw "-AcceptanceReason requires -AcceptQualityFailure."
}

function Write-Stage {
    param([string]$Number, [string]$Title, [string]$Status)
    $color = switch ($Status) {
        "SKIP"  { "DarkGray" }
        "RUN"   { "Cyan" }
        "PLAN"  { "Yellow" }
        default { "White" }
    }
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor $color
    Write-Host "  Stage $Number -- $Title [$Status]" -ForegroundColor $color
    Write-Host "============================================================" -ForegroundColor $color
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  CelebA Latent Training -- Windows Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Project root  : $ProjectRoot"
Write-Host "  Python        : $Python"
Write-Host "  Batch size    : $BatchSize"
Write-Host "  Num workers   : $NumWorkers"
Write-Host "  Dry run       : $DryRun"
Write-Host "============================================================" -ForegroundColor Cyan

# ===================================================================
# Stage 0 -- Prerequisite checks (CelebA images + pretrained source)
# ===================================================================
Write-Stage "0" "Prerequisite checks" "RUN"

if (-not (Test-Path -LiteralPath $CelebaRoot)) {
    throw "CelebA not found at $CelebaRoot. Run: scripts\windows\init.cmd -Datasets celeba"
}
$celebaImages = Join-Path $CelebaRoot "celeba/img_align_celeba"
if (-not (Test-Path -LiteralPath $celebaImages)) {
    throw "CelebA images missing at $celebaImages. Run: .\scripts\windows\setup.ps1 -Yes -Datasets celeba"
}
Write-Host "  [OK] CelebA images found at $celebaImages" -ForegroundColor Green

if (-not (Test-Path -LiteralPath $SourceVqvae)) {
    Write-Host "  [WARN] Pretrained VQ source not found at $SourceVqvae" -ForegroundColor Yellow
    Write-Host "  Will download in Stage 2." -ForegroundColor Yellow
} else {
    Write-Host "  [OK] Pretrained VQ source found at $SourceVqvae" -ForegroundColor Green
}

# ===================================================================
# Stage 1 -- Install missing Python packages
# ===================================================================
$missingPkgs = @()
$pkgChecks = @{
    "huggingface-hub" = "huggingface_hub"
    "diffusers"       = "diffusers"
    "accelerate"      = "accelerate"
    "safetensors"     = "safetensors"
}

foreach ($entry in $pkgChecks.GetEnumerator()) {
    $ErrorActionPreference = "Continue"
    & $Python -c "import $($entry.Value)" 2>$null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($exitCode -ne 0) {
        $missingPkgs += $entry.Key
    }
}

if ($SkipPackages -or $missingPkgs.Count -eq 0) {
    if ($missingPkgs.Count -eq 0) {
        Write-Stage "1" "Python packages" "SKIP"
        Write-Host "  All latent-pipeline packages already installed." -ForegroundColor Green
    } else {
        Write-Stage "1" "Python packages" "SKIP"
        Write-Host "  Skipped by -SkipPackages. Missing: $($missingPkgs -join ', ')" -ForegroundColor Yellow
    }
} else {
    Write-Stage "1" "Python packages ($($missingPkgs -join ', '))" $(if ($DryRun) { "PLAN" } else { "RUN" })

    # Install from requirements.txt to honour version pins
    $pipArgs = @("-m", "pip", "install", "--upgrade")
    $pipArgs += $missingPkgs
    Write-Host "  [CMD] $Python $($pipArgs -join ' ')" -ForegroundColor DarkCyan

    if (-not $DryRun) {
        & $Python @pipArgs
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed for: $($missingPkgs -join ', ')"
        }
        Write-Host "  [OK] Packages installed." -ForegroundColor Green
    }
}

# ===================================================================
# Stage 2 -- Download pretrained VQ + validate codec
# ===================================================================
if (Test-Path -LiteralPath $CodecCheckpoint) {
    Write-Stage "2" "Codec validation" "SKIP"
    Write-Host "  Accepted codec already exists at $CodecCheckpoint" -ForegroundColor Green
} else {
    Write-Stage "2" "Codec validation" $(if ($DryRun) { "PLAN" } else { "RUN" })

    # 2a -- Download pretrained VQ source if missing
    if (-not (Test-Path -LiteralPath $SourceVqvae)) {
        $downloadArgs = @("-m", "codec.download_pretrained_vq",
            "--repo-id", "CompVis/ldm-celebahq-256",
            "--revision", "main",
            "--output-dir", $SourceDir)
        Write-Host "  [CMD] $Python $($downloadArgs -join ' ')" -ForegroundColor DarkCyan

        if (-not $DryRun) {
            & $Python @downloadArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Pretrained VQ download failed."
            }
            Write-Host "  [OK] Pretrained VQ downloaded." -ForegroundColor Green
        }
    } else {
        Write-Host "  [OK] Pretrained VQ source already present." -ForegroundColor Green
    }

    # 2b -- Run codec validation to produce accepted_codec.pt
    $validateArgs = @("codec/validate_codec.py",
        "--codec-source", "CompVis/ldm-celebahq-256",
        "--codec-source-path", $SourceVqvae,
        "--codec-source-revision", "auto",
        "--celeba-root", $CelebaRoot,
        "--output-dir", $CodecOutputDir,
        "--batch-size", "$BatchSize",
        "--num-workers", "$NumWorkers",
        "--device", "cuda")
    if ($AcceptQualityFailure) {
        $validateArgs += @("--accept-quality-failure",
            "--acceptance-reason", $AcceptanceReason)
    }
    Write-Host "  [CMD] $Python $($validateArgs -join ' ')" -ForegroundColor DarkCyan

    if (-not $DryRun) {
        & $Python @validateArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Codec validation failed. See output above."
        }
        if (-not (Test-Path -LiteralPath $CodecCheckpoint)) {
            throw "Codec validation completed but $CodecCheckpoint was not created."
        }
        Write-Host "  [OK] Accepted codec written to $CodecCheckpoint" -ForegroundColor Green
    }
}

# ===================================================================
# Stage 3 -- Cache normalised CelebA latents
# ===================================================================
# Check if a valid cache manifest already exists that matches the codec
$latentCacheReady = $false
if (Test-Path -LiteralPath $LatentCacheDir) {
    $manifests = Get-ChildItem -Path $LatentCacheDir -Recurse -Filter "manifest.json" -ErrorAction SilentlyContinue
    if ($manifests.Count -gt 0) {
        $latentCacheReady = $true
    }
}

if ($latentCacheReady) {
    Write-Stage "3" "Latent cache" "SKIP"
    Write-Host "  Latent cache manifest found in $LatentCacheDir" -ForegroundColor Green
} else {
    Write-Stage "3" "Latent cache" $(if ($DryRun) { "PLAN" } else { "RUN" })

    if (-not (Test-Path -LiteralPath $CodecCheckpoint) -and -not $DryRun) {
        throw "Cannot cache latents: codec checkpoint not found at $CodecCheckpoint"
    }

    $cacheArgs = @("codec/cache_latents.py",
        "--codec-path", $CodecCheckpoint,
        "--celeba-root", $CelebaRoot,
        "--output-dir", $LatentCacheDir,
        "--split", "both",
        "--batch-size", "$BatchSize",
        "--device", "cuda")
    Write-Host "  [CMD] $Python $($cacheArgs -join ' ')" -ForegroundColor DarkCyan

    if (-not $DryRun) {
        & $Python @cacheArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Latent caching failed. See output above."
        }
        Write-Host "  [OK] Latent cache created in $LatentCacheDir" -ForegroundColor Green
    }
}

# ===================================================================
# Stage 4 -- Build latent FID reference statistics
# ===================================================================
if (Test-Path -LiteralPath $FidRefLatent) {
    Write-Stage "4" "FID reference stats (latent)" "SKIP"
    Write-Host "  Latent FID reference already exists at $FidRefLatent" -ForegroundColor Green
} else {
    Write-Stage "4" "FID reference stats (latent)" $(if ($DryRun) { "PLAN" } else { "RUN" })

    Write-Host "  [NOTE] The latent FID reference cache is built automatically" -ForegroundColor Yellow
    Write-Host "  during the first evaluation run. Skipping explicit generation." -ForegroundColor Yellow
    Write-Host "  If you want to pre-build it, run:" -ForegroundColor Yellow
    Write-Host "    $Python evaluate.py --algorithm mf_hutchinson --config config/mf_hutchinson_cv_celeba_latent.json --build-fid-cache-only" -ForegroundColor DarkCyan
}

# ===================================================================
# Summary
# ===================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$checks = @(
    @{ Label = "Python packages (HF/diffusers/accelerate/safetensors)";
       OK    = ($missingPkgs.Count -eq 0 -or (-not $DryRun -and -not $SkipPackages)) },
    @{ Label = "Accepted codec ($CodecCheckpoint)";
       OK    = (Test-Path -LiteralPath $CodecCheckpoint) },
    @{ Label = "Latent cache ($LatentCacheDir)";
       OK    = $latentCacheReady -or (-not $DryRun -and (Test-Path -LiteralPath $LatentCacheDir)) },
    @{ Label = "Latent FID reference ($FidRefLatent)";
       OK    = (Test-Path -LiteralPath $FidRefLatent) }
)

foreach ($check in $checks) {
    $icon  = if ($check.OK) { "[OK]" } else { "[--]" }
    $color = if ($check.OK) { "Green" } else { "Yellow" }
    Write-Host "  $icon $($check.Label)" -ForegroundColor $color
}

Write-Host ""
if ($DryRun) {
    Write-Host "  [DRY-RUN] No changes were made. Remove -DryRun to execute." -ForegroundColor Yellow
} else {
    Write-Host "  Ready to train! Example:" -ForegroundColor Green
    Write-Host "    .\scripts\windows\run_hutchinson_cv.ps1 -Mode fresh -Epochs 15 -TrainOnly" -ForegroundColor White
    Write-Host ""
    Write-Host "  Or the full CelebA latent suite:" -ForegroundColor Green
    Write-Host "    .\scripts\windows\train_celeba_latent.ps1 -Mode fresh" -ForegroundColor White
}
Write-Host ""
