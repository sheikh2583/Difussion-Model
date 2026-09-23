[CmdletBinding()]
param(
    [ValidateSet("continue", "fresh")][string]$Mode = "continue",
    [string]$DataRoot = "data/raw",
    [string]$WorkDir = "results/scratch_vae",
    [string]$Output = "results/codecs/accepted_scratch_kl_vae.pt",
    [ValidateRange(1, 1048576)][int]$BatchSize = 128,
    [ValidateRange(1, 1000000)][int]$Epochs = 60,
    [ValidateRange(0, 1024)][int]$NumWorkers = 4,
    [ValidateRange(0, 2147483647)][int]$Seed = 0,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Run scripts\windows\init.cmd first." }
$checkpointDir = Join-Path $WorkDir "checkpoints"
$checkpoints = @(Get-ChildItem -LiteralPath $checkpointDir -Filter "scratch_vae_epoch*.pt" -File -ErrorAction SilentlyContinue)
$resume = @()
if ($Mode -eq "continue" -and $checkpoints.Count) { $resume = @("--resume", "auto") }
if ($Mode -eq "fresh" -and $checkpoints.Count) {
    throw "Fresh mode refuses to overwrite $checkpointDir; choose a new -WorkDir."
}
$command = @("-m", "codec.train_scratch_vae", "--data-root", $DataRoot, "--work-dir", $WorkDir,
    "--output", $Output, "--device", "cuda", "--batch-size", "$BatchSize", "--epochs", "$Epochs",
    "--learning-rate", "1e-4", "--weight-decay", "1e-4", "--kl-start", "1e-5", "--kl-end", "1e-4",
    "--kl-warmup-epochs", "20", "--gradient-clip-norm", "1.0", "--validate-every", "5",
    "--validation-samples", "5000", "--num-workers", "$NumWorkers", "--seed", "$Seed", "--amp") + $resume
Write-Host "[PLAN] $Python $($command -join ' ')"
if ($DryRun) { exit 0 }
. "$PSScriptRoot\workflow_guard.ps1"
Start-WorkflowGuard -Python $Python -CommandName "train_scratch_codec.ps1" `
    -IdentityArguments @("--launcher", "scripts/windows/train_scratch_codec.ps1")
try {
    $logDir = Get-WorkflowDeviceLogDirectory -Category "codec"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir "celeba_codec_scratch_kl_vae_$((Get-Date).ToString('yyyyMMdd_HHmmss'))_pid$PID.log"
    @(
        "[run] training_type=scratch_codec"
        "[run] representation_space=codec"
        "[run] dataset=celeba"
        "[run] algorithm=scratch_kl_vae"
        "[run] started_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
        "[run] command=$Python $($command -join ' ')"
    ) | Set-Content -LiteralPath $log
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @command 2>&1 | Tee-Object -FilePath $log -Append
        $status = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    @(
        "[run] finished_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
        "[run] exit_status=$status"
    ) | Add-Content -LiteralPath $log
    if ($status -ne 0) { throw "Scratch codec training failed with exit code $status." }
} finally {
    Stop-WorkflowGuard -Python $Python
}
