# setup.ps1 — thin wrapper around bootstrap.py
# Usage: .\scripts\windows\setup.ps1 [-Yes] [-SkipTorch]
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
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$extra = @()
if ($Yes)         { $extra += "--yes" }
if ($SkipTorch)   { $extra += "--skip-torch" }
if ($Gpu -ne "")  { $extra += @("--gpu", $Gpu) }
if ($Datasets)     { $extra += @("--datasets", $Datasets) }

# Reuse an existing project interpreter first. This is useful on machines where
# Python was later removed from PATH but the checked project environment remains.
if (Test-Path -LiteralPath $VenvPython) {
    & $VenvPython -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $VenvPython bootstrap.py @extra
        exit $LASTEXITCODE
    }
}

# Prefer a healthy Python 3.10+ PATH installation; fall back to the Windows
# launcher. This avoids Microsoft Store aliases and stale installations.
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & python bootstrap.py @extra
        exit $LASTEXITCODE
    }
}
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "The Python launcher did not find Python 3.10 or newer."
    }
    & py -3 bootstrap.py @extra
    exit $LASTEXITCODE
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ is missing and winget is unavailable. Install Python from https://www.python.org/downloads/ and rerun this script."
}

Write-Host "Python 3.10+ was not found; installing Python 3.12 with winget..."
& winget install --exact --id Python.Python.3.12 --scope user `
    --accept-package-agreements --accept-source-agreements --silent
if ($LASTEXITCODE -ne 0) {
    throw "winget could not install Python (exit code $LASTEXITCODE)."
}

$InstalledPython = $null
$PythonCandidates = Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" `
    -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
foreach ($Candidate in $PythonCandidates) {
    & $Candidate.FullName -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $InstalledPython = $Candidate
        break
    }
}
if (-not $InstalledPython) {
    throw "Python was installed but python.exe could not be located. Open a new PowerShell window and rerun this script."
}
& $InstalledPython.FullName bootstrap.py @extra
exit $LASTEXITCODE
