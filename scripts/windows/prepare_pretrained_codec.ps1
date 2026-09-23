[CmdletBinding()]
param(
    [string]$DataRoot = "data/raw",
    [string]$SourceDir = "data/pretrained/ldm-celebahq-256",
    [string]$OutputDir = "results/codecs/celeba_vq_f4",
    [string]$Revision = "main",
    [ValidateRange(1, 1048576)][int]$BatchSize = 32,
    [ValidateRange(0, 1024)][int]$NumWorkers = 4,
    [switch]$AcceptQualityFailure,
    [string]$AcceptanceReason = "",
    [switch]$SkipDownload,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Run scripts\windows\init.cmd first." }
if ($AcceptQualityFailure -and -not $AcceptanceReason) {
    throw "-AcceptQualityFailure requires -AcceptanceReason."
}
if (-not $AcceptQualityFailure -and $AcceptanceReason) {
    throw "-AcceptanceReason requires -AcceptQualityFailure."
}
$download = @("-m", "codec.download_pretrained_vq", "--repo-id", "CompVis/ldm-celebahq-256",
    "--revision", $Revision, "--output-dir", $SourceDir)
$validate = @("codec/validate_codec.py", "--codec-source", "CompVis/ldm-celebahq-256",
    "--codec-source-path", "$SourceDir/vqvae", "--codec-source-revision", "auto",
    "--celeba-root", $DataRoot, "--output-dir", $OutputDir, "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers", "--device", "cuda")
if ($AcceptQualityFailure) {
    $validate += @("--accept-quality-failure", "--acceptance-reason", $AcceptanceReason)
}
if (-not $SkipDownload) { Write-Host "[PLAN] $Python $($download -join ' ')" }
Write-Host "[PLAN] $Python $($validate -join ' ')"
if ($DryRun) { exit 0 }
. "$PSScriptRoot\workflow_guard.ps1"
Start-WorkflowGuard -Python $Python -CommandName "prepare_pretrained_codec.ps1" `
    -IdentityArguments @("--launcher", "scripts/windows/prepare_pretrained_codec.ps1")
try {
    $logDir = Get-WorkflowDeviceLogDirectory -Category "codec"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir "celeba_codec_pretrained_vq_f4_$((Get-Date).ToString('yyyyMMdd_HHmmss'))_pid$PID.log"
    if (-not $SkipDownload) {
        & $Python @download 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw "Codec download failed." }
    }
    @(
        "[run] training_type=codec_validation"
        "[run] representation_space=codec"
        "[run] dataset=celeba"
        "[run] algorithm=pretrained_vq_f4"
        "[run] started_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
        "[run] command=$Python $($validate -join ' ')"
    ) | Add-Content -LiteralPath $log
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @validate 2>&1 | Tee-Object -FilePath $log -Append
        $status = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    @(
        "[run] finished_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
        "[run] exit_status=$status"
    ) | Add-Content -LiteralPath $log
    if ($status -ne 0) { throw "Codec validation failed with exit code $status." }
} finally {
    Stop-WorkflowGuard -Python $Python
}
