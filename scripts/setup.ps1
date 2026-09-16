# setup.ps1 — thin wrapper around bootstrap.py
# Usage: .\scripts\setup.ps1 [-Yes] [-Gpu cuda118|cuda121|rocm|cpu]
#                              [-Datasets cifar10|celeba|all|none]
[CmdletBinding()]
param(
    [switch]$Yes,
    [ValidateSet("cuda118","cuda121","rocm","cpu","")]
    [string]$Gpu = "",
    [ValidateSet("cifar10","celeba","all","none")]
    [string]$Datasets = "cifar10"
)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$extra = @()
if ($Yes)         { $extra += "--yes" }
if ($Gpu -ne "")  { $extra += @("--gpu", $Gpu) }
if ($Datasets)     { $extra += @("--datasets", $Datasets) }

# Prefer a normal PATH installation; fall back to the Windows Python launcher.
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python bootstrap.py @extra
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 bootstrap.py @extra
} else {
    throw "Python 3.9+ is required. Install it from https://www.python.org/downloads/ and rerun this script."
}
