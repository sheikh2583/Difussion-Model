# setup.ps1 — thin wrapper around bootstrap.py
# Usage: .\scripts\setup.ps1 [-Yes] [-Gpu cuda118|cuda121|rocm|cpu]
[CmdletBinding()]
param(
    [switch]$Yes,
    [ValidateSet("cuda118","cuda121","rocm","cpu","")]
    [string]$Gpu = ""
)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$extra = @()
if ($Yes)         { $extra += "--yes" }
if ($Gpu -ne "")  { $extra += @("--gpu", $Gpu) }
python bootstrap.py @extra
