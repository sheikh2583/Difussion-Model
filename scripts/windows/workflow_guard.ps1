Set-StrictMode -Version Latest

$script:WorkflowLockFile = "results/.lock"
$script:WorkflowLockToken = ""
$script:WorkflowLockOwned = $false

function Start-WorkflowGuard {
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$CommandName,
        [switch]$DryRun,
        [string[]]$IdentityArguments = @()
    )
    if (-not $env:DIFFUSION_PARENT_SUITE_TIMESTAMP) {
        $env:DIFFUSION_PARENT_SUITE_TIMESTAMP = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    }
    if ($DryRun) {
        Write-Host "[DRY-RUN] GPU lock: $script:WorkflowLockFile"
        Write-Host "[DRY-RUN] training source identity will be frozen before execution"
        return
    }

    $lockArgs = @("scripts/workflow_guard.py", "lock-acquire", "--lock-file",
        $script:WorkflowLockFile, "--command", $CommandName, "--owner-pid", "$PID")
    if ($env:DIFFUSION_GPU_LOCK_TOKEN) {
        $lockArgs += @("--token", $env:DIFFUSION_GPU_LOCK_TOKEN)
    }
    $parts = ((& $Python @lockArgs) -join "`n").Trim() -split "`t"
    if ($LASTEXITCODE -ne 0 -or $parts.Count -lt 2) {
        throw "Could not acquire the workflow GPU lock."
    }
    $script:WorkflowLockToken = $parts[0]
    $script:WorkflowLockOwned = $parts[1].Trim().ToLowerInvariant() -eq "true"
    $env:DIFFUSION_GPU_LOCK_TOKEN = $script:WorkflowLockToken

    if ($env:DIFFUSION_SOURCE_MANIFEST) {
        & $Python scripts/workflow_guard.py source-verify --manifest $env:DIFFUSION_SOURCE_MANIFEST | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Source identity verification failed." }
    } else {
        $manifest = "results/source_manifests/$($env:DIFFUSION_PARENT_SUITE_TIMESTAMP)_pid$PID.json"
        $identity = (& $Python scripts/workflow_guard.py source-freeze --manifest $manifest @IdentityArguments) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "Could not freeze the training source identity." }
        $env:DIFFUSION_SOURCE_IDENTITY = $identity.Trim()
        $env:DIFFUSION_SOURCE_MANIFEST = $manifest
    }
}

function Test-WorkflowSource {
    param([Parameter(Mandatory)][string]$Python, [switch]$DryRun)
    if (-not $DryRun) {
        & $Python scripts/workflow_guard.py source-verify --manifest $env:DIFFUSION_SOURCE_MANIFEST | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Training source changed after it was frozen." }
    }
}

function Stop-WorkflowGuard {
    param([Parameter(Mandatory)][string]$Python)
    if ($script:WorkflowLockOwned -and $script:WorkflowLockToken) {
        & $Python scripts/workflow_guard.py lock-release --lock-file $script:WorkflowLockFile `
            --token $script:WorkflowLockToken | Out-Null
    }
    $script:WorkflowLockOwned = $false
}

function Get-WorkflowDeviceLogDirectory {
    param([Parameter(Mandatory)][string]$Category)
    $gpuName = ""
    $gpuMemory = ""
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $gpuMemory = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
    }
    if (-not $gpuName) { return "training_logs/cpu-or-unknown/$Category" }
    $token = ($gpuName.ToLowerInvariant() -replace '[^a-z0-9]+', '-').Trim('-')
    if ($gpuMemory -match '^\d+$') {
        $token += "-$([math]::Ceiling([int]$gpuMemory / 1024))gb"
    }
    return "training_logs/$token/$Category"
}
