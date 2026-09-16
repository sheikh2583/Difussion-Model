# scripts/export_for_claude.ps1
# Creates a zip of all essential source files for sharing with Claude web.
# Output: x:\Thesis\thesis_for_claude.zip  (next to the project folder)
#
# Usage:
#   .\scripts\export_for_claude.ps1
#   .\scripts\export_for_claude.ps1 -Out "C:\somewhere\snapshot.zip"

param(
    [string]$Out = (Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) "thesis_for_claude.zip")
)

$root = Split-Path $PSScriptRoot -Parent

$exclude = @(
    "venv", "__pycache__", ".git", "results", "data\raw", "archive",
    "*.pyc", "*.pyo", "*.pt", "*.zip", "*.npz",
    "*.gif", "*.png", "*.jpg", "*.jpeg",
    "*.pptx", "*.pdf"
)

Write-Host "`n  Scanning source files..." -ForegroundColor Cyan

$files = Get-ChildItem $root -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($root.Length + 1)
    $skip = $false
    foreach ($ex in $exclude) {
        if ($rel -like "*$ex*") { $skip = $true; break }
    }
    -not $skip
}

if (Test-Path $Out) { Remove-Item $Out -Force }
Compress-Archive -Path $files.FullName -DestinationPath $Out -Force

$size  = [math]::Round((Get-Item $Out).Length / 1KB, 1)
$count = $files.Count

Write-Host "  [OK] $Out" -ForegroundColor Green
Write-Host "       $count files | ${size} KB`n"
