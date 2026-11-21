<#
.SYNOPSIS
    Re-score retained latent checkpoints against the raw-pixel CelebA FID reference.
.EXAMPLE
    .\scripts\windows\rescore_latent_fid.ps1 -DryRun
    .\scripts\windows\rescore_latent_fid.ps1 -Experiments fm_celeba_latent
#>
[CmdletBinding()]
param(
    [string[]]$Experiments = @(),
    [string]$OutputDir = "",
    [string]$Device = "",
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Run scripts\windows\init.cmd first." }
$command = @("scripts/rescore_latent_fid.py")
if ($Experiments.Count) { $command += @("--experiments") + $Experiments }
if ($OutputDir) { $command += @("--output-dir", $OutputDir) }
if ($Device) { $command += @("--device", $Device) }
if ($DryRun) { $command += "--dry-run" }
& $Python @command
exit $LASTEXITCODE
