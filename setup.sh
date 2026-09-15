#!/bin/bash
echo -e "\033[0;36mSetting up project environment...\033[0m"

# 1. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate Virtual Environment & Install Dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "\033[0;33mrequirements.txt not found. Skipping dependencies.\033[0m"
fi

# 3. Create Local Directories
echo "Creating local workspace directories..."
for dir in data models results experiments; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  -> Created $dir/"
    fi
done

# 4. Optional: Download large files
if [ "$1" == "--download-assets" ]; then
    echo -e "\033[0;36mDownloading large assets (models, datasets)...\033[0m"
    # TODO: Replace the URLs below with the actual hosted locations of your large files
    # curl -L -o models/model.safetensors "https://huggingface.co/your-model/resolve/main/model.safetensors"
    # curl -L -o data/dataset.zip "https://your-data-source.com/dataset.zip"
    echo "Assets downloaded!"
else
    echo "Skipping asset downloads. Run with '--download-assets' to download large files."
fi

echo -e "\n\033[0;32mSetup complete! To activate the environment in this shell, run:\033[0m"
echo -e "\033[0;33msource venv/bin/activate\033[0m"
