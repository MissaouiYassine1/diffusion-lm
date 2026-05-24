# ==========================================
# Project Structure Setup Script (PRO)
# ==========================================

# Stop execution on error
$ErrorActionPreference = "Stop"

try {
    # 📍 Get script directory (independent of where you run it)
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

    # 📍 Target = parent directory
    $basePath = Resolve-Path (Join-Path $scriptPath "..")

    Write-Host "Base path: $basePath"

    # 📂 Directories
    $dirs = @(
        "backend/core", "backend/api", "backend/services",
        "frontend/components", "frontend/assets",
        "data/raw", "data/processed",
        "scripts", "notebooks", "tests", "docs",
        ".github/workflows"
    )

    # 📄 Files
    $files = @(
        "backend/__init__.py", "backend/main.py", "backend/config.py",
        "backend/core/__init__.py", "backend/core/diffusion_model.py",
        "frontend/__init__.py", "frontend/app.py",
        "requirements.txt", "docker-compose.yml", ".env", ".gitignore"
    )

    # ==========================================
    # Create directories
    # ==========================================
    foreach ($dir in $dirs) {
        $fullPath = Join-Path $basePath $dir

        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-Host "[DIR CREATED] $dir"
        } else {
            Write-Host "[DIR EXISTS ] $dir"
        }
    }

    # ==========================================
    # Create files
    # ==========================================
    foreach ($file in $files) {
        $fullPath = Join-Path $basePath $file

        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType File -Path $fullPath -Force | Out-Null
            Write-Host "[FILE CREATED] $file"
        } else {
            Write-Host "[FILE EXISTS ] $file"
        }
    }

    # ==========================================
    # Optional: create .gitignore content
    # ==========================================
    $gitignorePath = Join-Path $basePath ".gitignore"

    if (-not (Test-Path $gitignorePath)) {
        @"
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
venv/
.env

# Node
node_modules/

# Data
data/raw/*
data/processed/*

# Logs
*.log

# OS
.DS_Store
Thumbs.db
"@ | Out-File -Encoding UTF8 $gitignorePath

        Write-Host "[FILE CREATED] .gitignore (with content)"
    }

    Write-Host "`n[OK] Project structure created successfully!"

} catch {
    Write-Host "[ERROR] $($_.Exception.Message)"
}