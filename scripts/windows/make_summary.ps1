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
$SummaryArguments = @("scripts/aggregate_results.py", "--results-root", $ResultsRoot, "--output-dir", $OutputDir)
$CatalogArguments = @("scripts/catalog_training_logs.py", "--results-root", $ResultsRoot, "--output-dir", $OutputDir)
$VerifyArguments = @("scripts/package_thesis_context.py", "--verify-inputs")
Write-Host "Summary command: $Python $($SummaryArguments -join ' ')"
Write-Host "Catalog command: $Python $($CatalogArguments -join ' ')"
if ($ResultsRoot -eq "results" -and $OutputDir -eq "results/aggregate") {
    Write-Host "Verification command: $Python $($VerifyArguments -join ' ')"
}
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
& $Python @SummaryArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python @CatalogArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($ResultsRoot -eq "results" -and $OutputDir -eq "results/aggregate") {
    & $Python @VerifyArguments
    exit $LASTEXITCODE
}
Write-Host "Custom aggregate paths built; thesis-context verification applies only to canonical paths."
exit 0
