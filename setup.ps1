param(
    [switch]$DownloadAssets = $false
)

Write-Host "Setting up project environment..." -ForegroundColor Cyan

# 1. Create Virtual Environment
if (-Not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

# 2. Activate Virtual Environment & Install Dependencies
Write-Host "Activating virtual environment and installing dependencies..."
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    & ".\venv\Scripts\pip.exe" install -r requirements.txt
} else {
    Write-Host "requirements.txt not found. Skipping dependencies." -ForegroundColor Yellow
}

# 3. Create Local Directories
Write-Host "Creating local workspace directories..."
$dirs = @("data", "models", "results", "experiments")
foreach ($dir in $dirs) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  -> Created $dir/"
    }
}

# 4. Optional: Download large files
if ($DownloadAssets) {
    Write-Host "Downloading large assets (models, datasets)..." -ForegroundColor Cyan
    # TODO: Replace the URLs below with the actual hosted locations of your large files
    # Invoke-WebRequest -Uri "https://huggingface.co/your-model/resolve/main/model.safetensors" -OutFile "models/model.safetensors"
    # Invoke-WebRequest -Uri "https://your-data-source.com/dataset.zip" -OutFile "data/dataset.zip"
    Write-Host "Assets downloaded!"
} else {
    Write-Host "Skipping asset downloads. Run with '-DownloadAssets' to download large files."
}

Write-Host "`nSetup complete! To activate the environment in this shell, run:" -ForegroundColor Green
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
