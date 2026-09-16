# train_all.ps1 — Windows PowerShell equivalent of train_all.sh
# Usage: .\train_all.ps1
#        .\train_all.ps1 -Only mf_distill
#        .\train_all.ps1 -SkipReflow

param(
    [switch]$SkipFm, [switch]$SkipFmLognorm, [switch]$SkipMf,
    [switch]$SkipMfDistill, [switch]$SkipConsistency, [switch]$SkipReflow,
    [string]$Only = ""
)
$ErrorActionPreference = "Stop"

function Run-Training($Algo, $Cfg) {
    if ($Only -and $Only -ne $Algo) { return }
    Write-Host "`n=== Training: $Algo ===" -ForegroundColor Cyan
    python train.py --algorithm $Algo --config $Cfg
    if ($LASTEXITCODE -ne 0) { throw "Failed: $Algo" }
}

$FmCkpt = "results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt"

if (-not $SkipFm)        { Run-Training fm          config/fm_full.json }
if (-not $SkipFmLognorm) { Run-Training fm_lognorm  config/fm_lognorm_full.json }
if (-not $SkipMf)        { Run-Training mf           config/mf_full.json }

if (-not $SkipMfDistill) {
    if (-not (Test-Path $FmCkpt) -and -not $Only) { throw "FM checkpoint missing: $FmCkpt" }
    Run-Training mf_distill config/mf_distill_full.json
}
if (-not $SkipConsistency) {
    if (-not (Test-Path $FmCkpt) -and -not $Only) { throw "FM checkpoint missing: $FmCkpt" }
    Run-Training consistency config/consistency_full.json
}
if (-not $SkipReflow) {
    $pairs = "data/reflow_pairs_cifar10.pt"
    if (-not (Test-Path $pairs) -and -not $Only) {
        Write-Error "Reflow pairs missing. Run scripts/generate_reflow_pairs.py first."
    }
    Run-Training reflow config/reflow_full.json
}

Write-Host "`nAll training complete." -ForegroundColor Green
