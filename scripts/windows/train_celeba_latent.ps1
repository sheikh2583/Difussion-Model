# Train the CelebA latent suite serially with dependency checks and per-job logs.
[CmdletBinding()]
param(
    [ValidateSet("all", "fm", "fm_lognorm", "mf", "mf_hutchinson", "mf_distill", "consistency", "reflow")]
    [string]$Only = "all",
    [ValidateSet("continue", "fresh")][string]$Mode = "continue",
    [string]$MachineLabel = "",
    [ValidateRange(0, 1000000)][int]$CheckpointEvery = 0,
    [switch]$TrainOnly,
    [string]$LogDir = "",
    [switch]$AllowTeacherSourceMismatch,
    [switch]$DryRun,
    [switch]$List
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Run scripts\windows\init.cmd first." }

$algorithms = @("fm", "fm_lognorm", "mf", "mf_hutchinson", "mf_distill", "consistency", "reflow")
if ($List) { $algorithms | ForEach-Object { Write-Output $_ }; exit 0 }
$configs = @{
    fm = "config/fm_celeba_latent.json"
    fm_lognorm = "config/fm_lognorm_celeba_latent.json"
    mf = "config/mf_celeba_latent.json"
    mf_hutchinson = "config/mf_hutchinson_cv_celeba_latent.json"
    mf_distill = "config/mf_distill_celeba_latent.json"
    consistency = "config/consistency_celeba_latent.json"
    reflow = "config/reflow_celeba_latent.json"
}
$classes = @{
    fm = "FlowMatchingAlgorithm"; fm_lognorm = "FlowMatchingLognormAlgorithm"
    mf = "MeanFlowAlgorithm"; mf_hutchinson = "MeanFlowHutchinsonAlgorithm"
    mf_distill = "MeanFlowDistillAlgorithm"; consistency = "ConsistencyAlgorithm"
    reflow = "ReflowAlgorithm"
}
$jobs = if ($Only -eq "all") { $algorithms } else { @($Only) }
$codec = "results/codecs/celeba_vq_f4/accepted_codec.pt"
$reflowPairs = "data/reflow_pairs_celeba_latent.pt"

. "$PSScriptRoot\workflow_guard.ps1"
$identityArgs = @("--launcher", "scripts/windows/train_celeba_latent.ps1")
foreach ($name in $algorithms) { $identityArgs += @("--config", $configs[$name]) }
Start-WorkflowGuard -Python $Python -CommandName "train_celeba_latent.ps1" `
    -DryRun:$DryRun -IdentityArguments $identityArgs
try {
    if (-not $DryRun -and -not (Test-Path -LiteralPath $codec)) {
        throw "Missing accepted codec: $codec"
    }
    if ($DryRun) { Write-Host "[PLANNED] accepted codec: $codec" }
    $deviceLogRoot = Get-WorkflowDeviceLogDirectory -Category "latent"
    $deviceToken = ($deviceLogRoot.Replace('\', '/') -split '/')[1]
    if (-not $LogDir) { $LogDir = $deviceLogRoot }
    if (-not $MachineLabel) {
        $MachineLabel = "windows-$($env:COMPUTERNAME)-$deviceToken".ToLowerInvariant()
    }
    $env:DIFFUSION_MACHINE_LABEL = $MachineLabel
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $plannedTeacher = (& $Python scripts/checkpoint_path.py --run-dir results/fm_celeba_latent `
        --class-name FlowMatchingAlgorithm --epoch 100 --planned-mode $Mode).Trim()

    function Invoke-LatentCommand {
        param([string]$Job, [string]$ConfigPath, [string[]]$Command)
        Write-Host "[PLAN] job=$Job"
        Write-Host "[PLAN] $Python $($Command -join ' ')"
        if ($DryRun) { return }
        Test-WorkflowSource -Python $Python
        $jobDir = Join-Path $LogDir $Job
        New-Item -ItemType Directory -Force -Path $jobDir | Out-Null
        $log = Join-Path $jobDir "celeba_latent_${Job}_$($env:COMPUTERNAME)_${timestamp}.log"
        if (Test-Path -LiteralPath $log) { throw "Refusing to overwrite $log" }
        @(
            "[run] training_type=latent_diffusion"
            "[run] representation_space=latent"
            "[run] dataset=celeba_latent"
            "[run] algorithm=$Job"
        ) | Set-Content -LiteralPath $log
        $env:DIFFUSION_SELECTED_CONFIG_SHA256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ConfigPath).Hash.ToLowerInvariant()
        & $Python scripts/print_run_provenance.py --config $ConfigPath `
            --machine-label $MachineLabel --log-device-token $deviceToken `
            --log-path $log | Add-Content -LiteralPath $log
        @(
            "[run] command=$Python $($Command -join ' ')"
            "[run] started_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
        ) | Add-Content -LiteralPath $log
        & $Python @Command 2>&1 | Tee-Object -FilePath $log -Append
        $status = $LASTEXITCODE
        @(
            "[run] finished_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
            "[run] exit_status=$status"
        ) | Add-Content -LiteralPath $log
        & $Python scripts/write_training_log_metadata.py --log $log | Out-Null
        Write-Host "[LOG] $log"
        Write-Host "[METADATA] ${log}.meta.json"
        if ($status -ne 0) { throw "$Job failed with exit code $status" }
    }

    function Get-TeacherCheckpoint {
        if ($Only -eq "all") { return $plannedTeacher }
        return ((& $Python scripts/checkpoint_path.py --run-dir results/fm_celeba_latent `
            --class-name FlowMatchingAlgorithm --epoch 100).Trim())
    }

    foreach ($algorithm in $jobs) {
        $effectiveConfig = $configs[$algorithm]
        if ($algorithm -in @("mf_distill", "consistency")) {
            $teacher = Get-TeacherCheckpoint
            if (-not $DryRun -and -not (Test-Path -LiteralPath $teacher)) {
                throw "Missing latent FM teacher checkpoint: $teacher"
            }
            $effectiveConfig = "results/launcher_configs/$timestamp/${algorithm}_celeba_latent.json"
            Write-Host "[PLAN] bind $($configs[$algorithm]) to teacher $teacher"
            if (-not $DryRun) {
                & $Python scripts/prepare_latent_dependency_config.py --source $configs[$algorithm] `
                    --output $effectiveConfig --teacher-checkpoint $teacher | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "Could not prepare $algorithm config." }
            }
        }
        if ($algorithm -eq "reflow" -and -not (Test-Path -LiteralPath $reflowPairs)) {
            $teacher = Get-TeacherCheckpoint
            $pairCommand = @("scripts/generate_reflow_pairs_latent.py", "--checkpoint", $teacher,
                "--config", $configs.fm, "--output", $reflowPairs, "--n-pairs", "50000", "--nfe", "50")
            if ($AllowTeacherSourceMismatch) {
                $pairCommand += "--allow-source-identity-mismatch"
            }
            $env:DIFFUSION_CHECKPOINT_SERIES = Split-Path -Leaf (Split-Path -Parent $teacher)
            Invoke-LatentCommand -Job "reflow_pairs" -ConfigPath $configs.fm -Command $pairCommand
        }
        $command = @("train.py", "--algorithm", $algorithm, "--config", $effectiveConfig,
            "--mode", $Mode, "--machine-label", $MachineLabel)
        if ($CheckpointEvery -gt 0) { $command += @("--checkpoint-every", "$CheckpointEvery") }
        if ($TrainOnly) { $command += "--train-only" }
        $resolvedCheckpoint = (& $Python scripts/checkpoint_path.py `
            --run-dir "results/${algorithm}_celeba_latent" `
            --class-name $classes[$algorithm] --epoch 100 --planned-mode $Mode).Trim()
        $env:DIFFUSION_CHECKPOINT_SERIES = Split-Path -Leaf (Split-Path -Parent $resolvedCheckpoint)
        Invoke-LatentCommand -Job $algorithm -ConfigPath $effectiveConfig -Command $command
    }
    if ($DryRun) { Write-Host "Dry-run complete; no model commands were executed." -ForegroundColor Green }
    else { Write-Host "CelebA latent workflow complete." -ForegroundColor Green }
} finally {
    Stop-WorkflowGuard -Python $Python
}
