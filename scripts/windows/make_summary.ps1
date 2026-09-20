<# Build aggregate tables and plots without controlling training processes. #>
[CmdletBinding()]
param(
    [string]$ResultsRoot = "results",
    [string]$OutputDir = "results/aggregate",
    [switch]$AllowRunning,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment not found. Run scripts\windows\init.cmd first." }
$Arguments = @("scripts/aggregate_results.py", "--results-root", $ResultsRoot, "--output-dir", $OutputDir)
Write-Host "Command: $Python $($Arguments -join ' ')"
if ($DryRun) { exit 0 }

if (-not $AllowRunning) {
    $Active = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine.Contains($ProjectRoot) -and $_.CommandLine -match 'train\.py(?:\s|$)'
    }
    if ($Active) {
        Write-Error "Training is active; aggregate outputs were not changed. Use -AllowRunning for a read-only snapshot."
    }
}
Set-Location $ProjectRoot
& $Python @Arguments
exit $LASTEXITCODE
