[CmdletBinding()]
param(
    [ValidateRange(10, 2147483647)][int]$Interval = 300,
    [ValidateRange(0, 2147483647)][int]$Iterations = 0,
    [switch]$Once,
    [switch]$AllowRunning,
    [switch]$DryRun,
    [string]$Output = "",
    [ValidateRange(0, 2147483647)][int]$MaxMiB = 0
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot
if ($Once -or $DryRun) { $Iterations = 1 }
$mutexName = "Global\DiffusionThesisContext_$(([Math]::Abs($ProjectRoot.GetHashCode())))"
$mutex = [Threading.Mutex]::new($false, $mutexName)
if (-not $mutex.WaitOne(0)) { throw "Another thesis-context refresh watcher is active." }
try {
    $attempt = 0
    $lastStatus = 0
    while ($Iterations -eq 0 -or $attempt -lt $Iterations) {
        $attempt++
        Write-Host "[$((Get-Date).ToUniversalTime().ToString('o'))] thesis-context refresh attempt $attempt"
        $arguments = @()
        if ($AllowRunning) { $arguments += "-AllowRunning" }
        if ($DryRun) { $arguments += "-DryRun" }
        if ($Output) { $arguments += @("-Output", $Output) }
        if ($MaxMiB -gt 0) { $arguments += @("-MaxMiB", "$MaxMiB") }
        try {
            & "$PSScriptRoot\make_thesis_context.ps1" @arguments
            if (-not $? -or $LASTEXITCODE -ne 0) {
                throw "make_thesis_context.ps1 exited with status $LASTEXITCODE"
            }
            $lastStatus = 0
            Write-Host "Refresh verified."
        } catch {
            $lastStatus = 1
            Write-Warning "Refresh failed safely; existing ZIP was preserved: $($_.Exception.Message)"
        }
        if ($Iterations -ne 0 -and $attempt -ge $Iterations) { break }
        Start-Sleep -Seconds $Interval
    }
    exit $lastStatus
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
