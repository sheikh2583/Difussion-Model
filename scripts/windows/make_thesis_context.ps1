<# Build the compact thesis discussion package. #>
[CmdletBinding()]
param(
    [switch]$AllowRunning,
    [switch]$DryRun,
    [switch]$SkipSummary,
    [string]$Output = "",
    [double]$MaxMiB = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment not found. Run scripts\windows\init.cmd first." }
$Arguments = @("scripts/package_thesis_context.py")
if ($AllowRunning) { $Arguments += "--allow-running" }
if ($DryRun) { $Arguments += "--dry-run" }
if ($Output) { $Arguments += @("--output", $Output) }
if ($MaxMiB -ne 100) { $Arguments += @("--max-mib", [string]$MaxMiB) }
Set-Location $ProjectRoot
if (-not $SkipSummary) {
    $SummaryArguments = @()
    if ($AllowRunning) { $SummaryArguments += "-AllowRunning" }
    if ($DryRun) { $SummaryArguments += "-DryRun" }
    & (Join-Path $PSScriptRoot "make_summary.ps1") @SummaryArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $Python @Arguments
exit $LASTEXITCODE
