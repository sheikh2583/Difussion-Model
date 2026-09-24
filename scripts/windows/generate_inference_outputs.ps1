<# Pre-render real checkpoint generations for the browser-only thesis demo. #>
[CmdletBinding()]
param(
    [string]$Experiments = "fm_lognorm_celeba",
    [string]$Nfe = "1,5,10,20,50",
    [string]$Seeds = "0,1,2,3",
    [int]$NSamples = 64,
    [switch]$AllCeleba,
    [switch]$Wait
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project interpreter not found at '$Python'."
}
if ($NSamples -lt 1) { throw "NSamples must be positive." }

Set-Location $ProjectRoot
if ($Wait) {
    Write-Host "Waiting for the project GPU workflow lock to clear..."
    while (Test-Path -LiteralPath "results\.lock") { Start-Sleep -Seconds 15 }
}

$Arguments = @(
    "scripts/generate_checkpoint_samples.py",
    "--nfe", $Nfe,
    "--seeds", $Seeds,
    "--n-samples", $NSamples
)
if ($AllCeleba) {
    $Arguments += @("--dataset-family", "celeba")
} else {
    $Arguments += @("--experiments", $Experiments)
}
Write-Host "Command: $Python $($Arguments -join ' ')"
& $Python @Arguments
exit $LASTEXITCODE
