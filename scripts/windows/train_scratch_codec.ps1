[CmdletBinding()]
param(
    [ValidateSet("", "A", "B", "C", "D")][string]$Stage = "",
    [ValidateSet("continue", "fresh")][string]$Mode = "continue",
    [string]$DataRoot = "data/raw",
    [string]$WorkDir = "results/scratch_vae",
    [string]$Output = "results/codecs/accepted_scratch_kl_vae.pt",
    [ValidateRange(1, 1048576)][int]$BatchSize = 128,
    [ValidateRange(1, 1000000)][int]$Epochs = 60,
    [ValidateRange(0, 1024)][int]$NumWorkers = 4,
    [ValidateRange(0, 2147483647)][int]$Seed = 0,
    [double]$LearningRate = 1e-4,
    [double]$WeightDecay = 1e-4,
    [double]$KlStart = 1e-5,
    [double]$KlEnd = 1e-4,
    [ValidateRange(1, 1000000)][int]$KlWarmupEpochs = 20,
    [double]$GradientClipNorm = 1.0,
    [ValidateRange(1, 1000000)][int]$ValidateEvery = 5,
    [ValidateRange(1, 1000000)][int]$ValidationSamples = 5000,
    [double]$LambdaKl = 1e-6,
    [double]$LambdaPerc = 0.1,
    [double]$LambdaAdv = 0.1,
    [ValidateRange(1, 1000000)][int]$AdvStartEpoch = 5,
    [double]$DiscLr = 4e-4,
    [switch]$SweepMode,
    [ValidateRange(1, 1000000)][int]$SweepEpochs = 10,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Run scripts\windows\init.cmd first." }
switch ($Stage) {
    "A" { $LambdaPerc = 0.0; $LambdaAdv = 0.0; $SweepMode = $true; $SweepEpochs = 5; $Epochs = 5; $ValidateEvery = 9999 }
    "B" { $LambdaAdv = 0.0; $AdvStartEpoch = 9999; $SweepMode = $true; $SweepEpochs = 10; $Epochs = 10; $ValidateEvery = 9999 }
    "C" { $AdvStartEpoch = 5; $SweepMode = $false; $Epochs = 20; $ValidateEvery = 5 }
    "D" { $AdvStartEpoch = 5; $SweepMode = $false; $Epochs = 150; $ValidateEvery = 20 }
}
if ($Stage) {
    if (-not $PSBoundParameters.ContainsKey("WorkDir")) { $WorkDir = "results/codecs/scratch_kl_vae_stage_${Stage}_kl$LambdaKl" }
    if (-not $PSBoundParameters.ContainsKey("Output")) { $Output = "$WorkDir/accepted_codec.pt" }
}
$checkpointDir = Join-Path $WorkDir "checkpoints"
$checkpoints = @(Get-ChildItem -LiteralPath $checkpointDir -Filter "scratch_vae_epoch*.pt" -File -ErrorAction SilentlyContinue)
$resume = @()
if ($Mode -eq "continue" -and $checkpoints.Count) { $resume = @("--resume", "auto") }
if ($Mode -eq "fresh" -and $checkpoints.Count) {
    throw "Fresh mode refuses to overwrite $checkpointDir; choose a new -WorkDir."
}
$command = @("-m", "codec.train_scratch_vae", "--data-root", $DataRoot, "--work-dir", $WorkDir,
    "--output", $Output, "--device", "cuda", "--batch-size", "$BatchSize", "--epochs", "$Epochs",
    "--learning-rate", "$LearningRate", "--weight-decay", "$WeightDecay", "--kl-start", "$KlStart", "--kl-end", "$KlEnd",
    "--kl-warmup-epochs", "$KlWarmupEpochs", "--gradient-clip-norm", "$GradientClipNorm", "--validate-every", "$ValidateEvery",
    "--validation-samples", "$ValidationSamples", "--num-workers", "$NumWorkers", "--seed", "$Seed", "--amp",
    "--lambda-kl", "$LambdaKl", "--lambda-perc", "$LambdaPerc", "--lambda-adv", "$LambdaAdv",
    "--adv-start-epoch", "$AdvStartEpoch", "--disc-lr", "$DiscLr") + $resume
if ($SweepMode) { $command += @("--sweep-mode", "--sweep-epochs", "$SweepEpochs") }
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
