<# Build the compact thesis discussion package. #>
[CmdletBinding()]
param([switch]$AllowRunning, [switch]$DryRun, [string]$Output = "")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Project environment not found. Run scripts\windows\init.cmd first." }
$Arguments = @("scripts/package_thesis_context.py")
if ($AllowRunning) { $Arguments += "--allow-running" }
if ($DryRun) { $Arguments += "--dry-run" }
if ($Output) { $Arguments += @("--output", $Output) }
Set-Location $ProjectRoot
& $Python @Arguments
exit $LASTEXITCODE
