<# Generate separate CelebA pixel/latent animations and checkpoint sample grids. #>
[CmdletBinding()]
param([switch]$AllowRunning, [switch]$DryRun)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment not found. Run scripts\windows\init.cmd first." }
$PixelArguments = @("scripts/generate_result_gifs.py", "--dataset", "celeba")
$LatentArguments = @("scripts/generate_result_gifs.py", "--dataset", "celeba_latent")
$SampleArguments = @("scripts/generate_checkpoint_samples.py", "--dataset-family", "celeba")
Write-Host "Command: $Python $($PixelArguments -join ' ')"
Write-Host "Command: $Python $($LatentArguments -join ' ')"
Write-Host "Command: $Python $($SampleArguments -join ' ')"
if ($DryRun) { exit 0 }
$Active = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($ProjectRoot) -and $_.CommandLine -match 'train\.py(?:\s|$)'
}
if (-not $AllowRunning) {
    if ($Active) { Write-Error "Training is active; outputs were not changed. Use -AllowRunning for metric-only snapshots." }
}
Set-Location $ProjectRoot
& $Python @PixelArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python @LatentArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Active) {
    Write-Warning "Training is active; checkpoint sampling was skipped."
    exit 0
}
& $Python @SampleArguments
exit $LASTEXITCODE
