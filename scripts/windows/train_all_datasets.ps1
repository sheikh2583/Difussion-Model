# Unattended dependency-aware CIFAR-10 and CelebA pixel training.
[CmdletBinding()]
param(
    [ValidateSet("all", "cifar10", "celeba")][string]$Dataset = "all",
    [ValidateSet("continue", "fresh")][string]$Mode = "continue",
    [ValidateRange(1, 1000000)][int]$CheckpointEvery = 10,
    [switch]$TrainOnly,
    [ValidateSet("current", "legacy")][string]$CifarBackbone = "current",
    [string]$MachineLabel = "",
    [string]$LogDir = "",
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Run scripts\windows\init.cmd first." }
. "$PSScriptRoot\workflow_guard.ps1"

$identityArgs = @("--launcher", "scripts/windows/train_all_datasets.ps1",
    "--launcher", "scripts/windows/train_all.ps1")
Get-ChildItem config -Filter "*_full.json" -File | ForEach-Object {
    $identityArgs += @("--config", $_.FullName)
}
Get-ChildItem config -Filter "*_celeba64.json" -File | ForEach-Object {
    $identityArgs += @("--config", $_.FullName)
}
if (Test-Path -LiteralPath "config/mf_v3_exact_jvp_b128.json") {
    $identityArgs += @("--config", "config/mf_v3_exact_jvp_b128.json")
}
Get-ChildItem config/cifar_legacy -Filter "*.json" -File | ForEach-Object {
    $identityArgs += @("--config", $_.FullName)
}
Start-WorkflowGuard -Python $Python -CommandName "train_all_datasets.ps1" `
    -DryRun:$DryRun -IdentityArguments $identityArgs
try {
    $datasets = if ($Dataset -eq "all") { @("cifar10", "celeba") } else { @($Dataset) }
    $algorithms = @("fm", "fm_lognorm", "mf", "mf_distill", "consistency", "reflow")
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $hostToken = ($env:COMPUTERNAME.ToLowerInvariant() -replace '[^a-z0-9]+', '-').Trim('-')
    $logRoot = if ($LogDir) { $LogDir } else { Get-WorkflowDeviceLogDirectory -Category "pixel" }
    if (-not $MachineLabel) {
        $deviceToken = ($logRoot.Replace('\', '/') -split '/')[1]
        $MachineLabel = "windows-$($env:COMPUTERNAME)-$deviceToken".ToLowerInvariant()
    }
    $env:DIFFUSION_MACHINE_LABEL = $MachineLabel
    $failures = [System.Collections.Generic.List[string]]::new()

    foreach ($datasetName in $datasets) {
        foreach ($algorithm in $algorithms) {
            $parameters = @{
                Dataset = $datasetName
                Only = $algorithm
                Mode = $Mode
                CheckpointEvery = $CheckpointEvery
                TrainOnly = [bool]$TrainOnly
                DryRun = [bool]$DryRun
                MachineLabel = $MachineLabel
            }
            if ($datasetName -eq "cifar10") { $parameters.CifarBackbone = $CifarBackbone }
            if ($datasetName -eq "cifar10" -and $algorithm -eq "mf" -and $CifarBackbone -eq "current") {
                $parameters.MfConfig = "config/mf_v3_exact_jvp_b128.json"
            }
            $display = $parameters.GetEnumerator() | Sort-Object Key | ForEach-Object {
                if ($_.Value -is [bool]) { if ($_.Value) { "-$($_.Key)" } }
                else { "-$($_.Key) $($_.Value)" }
            }
            Write-Host "[PLAN] $PSScriptRoot\train_all.ps1 $($display -join ' ')"
            if ($DryRun) {
                & "$PSScriptRoot\train_all.ps1" @parameters
                continue
            }

            Test-WorkflowSource -Python $Python
            New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
            $logPath = Join-Path $logRoot "${datasetName}_pixel_${algorithm}_${hostToken}_${timestamp}.log"
            $status = 0
            try {
                @(
                    "[run] training_type=pixel_diffusion"
                    "[run] representation_space=pixel"
                    "[run] dataset=$datasetName"
                    "[run] algorithm=$algorithm"
                    "[run] source_identity_sha256=$($env:DIFFUSION_SOURCE_IDENTITY)"
                    "[run] lifecycle_mode=$Mode"
                    "[run] parent_suite_timestamp=$($env:DIFFUSION_PARENT_SUITE_TIMESTAMP)"
                    "[run] checkpoint_series=resolved-by-train.py"
                    "[run] started_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
                    "[run] command=$PSScriptRoot\train_all.ps1 $($display -join ' ')"
                ) | Set-Content -LiteralPath $logPath
                & "$PSScriptRoot\train_all.ps1" @parameters 2>&1 | Tee-Object -FilePath $logPath -Append
                if ($LASTEXITCODE -ne 0) { $status = $LASTEXITCODE }
            } catch {
                $status = if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { $LASTEXITCODE } else { 1 }
                "[run] error=$($_.Exception.Message)" | Add-Content -LiteralPath $logPath
            }
            @(
                "[run] finished_utc=$((Get-Date).ToUniversalTime().ToString('o'))"
                "[run] exit_status=$status"
            ) | Add-Content -LiteralPath $logPath
            if ($status -ne 0) {
                $failures.Add("${datasetName}:${algorithm}")
                Write-Warning "$datasetName/$algorithm failed with exit code $status; continuing."
            }
        }
    }
    if ($failures.Count) { throw "Completed with failed jobs: $($failures -join ', ')" }
    if ($DryRun) { Write-Host "Dry-run complete; no model commands were executed." -ForegroundColor Green }
    else { Write-Host "All requested pixel jobs completed successfully." -ForegroundColor Green }
} finally {
    Stop-WorkflowGuard -Python $Python
}