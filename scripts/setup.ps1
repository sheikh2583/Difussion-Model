# setup.ps1 — thin wrapper around bootstrap.py
# Usage: .\scripts\setup.ps1 [-Yes] [-SkipTorch]
#                              [-Gpu cuda118|cuda121|cuda128|rocm|cpu]
#                              [-Datasets cifar10|celeba|all|none]
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$SkipTorch,
    [ValidateSet("cuda118","cuda121","cuda128","rocm","cpu","")]
    [string]$Gpu = "",
    [ValidateSet("cifar10","celeba","all","none")]
    [string]$Datasets = "cifar10"
)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$extra = @()
if ($Yes)         { $extra += "--yes" }
if ($SkipTorch)   { $extra += "--skip-torch" }
if ($Gpu -ne "")  { $extra += @("--gpu", $Gpu) }
if ($Datasets)     { $extra += @("--datasets", $Datasets) }

# Prefer a healthy Python 3.9+ PATH installation; fall back to the Windows
# launcher. This avoids Microsoft Store aliases and stale installations.
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; raise SystemExit(sys.version_info < (3, 9))" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & python bootstrap.py @extra
        exit $LASTEXITCODE
    }
}
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 9))" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "The Python launcher did not find Python 3.9 or newer."
    }
    & py -3 bootstrap.py @extra
    exit $LASTEXITCODE
}
throw "Python 3.9+ is required. Install it from https://www.python.org/downloads/ and rerun this script."
