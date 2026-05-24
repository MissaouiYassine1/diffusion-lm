#!/usr/bin/env bash

# ==========================================
# Project Structure Setup Script (PRO - Bash)
# ==========================================

set -e  # stop on error

# 📍 Get script directory (independent of where you run it)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 📍 Target = parent directory
BASE_PATH="$(cd "$SCRIPT_PATH/.." && pwd)"

echo "Base path: $BASE_PATH"

# 📂 Directories
DIRS=(
"backend/core" "backend/api" "backend/services"
"frontend/components" "frontend/assets"
"data/raw" "data/processed"
"scripts" "notebooks" "tests" "docs"
".github/workflows"
)

# 📄 Files
FILES=(
"backend/__init__.py" "backend/main.py" "backend/config.py"
"backend/core/__init__.py" "backend/core/diffusion_model.py"
"frontend/__init__.py" "frontend/app.py"
"requirements.txt" "docker-compose.yml" ".env" ".gitignore"
)

# ==========================================
# Create directories
# ==========================================
for dir in "${DIRS[@]}"; do
    full_path="$BASE_PATH/$dir"

    if [ ! -d "$full_path" ]; then
        mkdir -p "$full_path"
        echo "[DIR CREATED] $dir"
    else
        echo "[DIR EXISTS ] $dir"
    fi
done

# ==========================================
# Create files
# ==========================================
for file in "${FILES[@]}"; do
    full_path="$BASE_PATH/$file"

    if [ ! -f "$full_path" ]; then
        mkdir -p "$(dirname "$full_path")"  # ensure parent dir exists
        touch "$full_path"
        echo "[FILE CREATED] $file"
    else
        echo "[FILE EXISTS ] $file"
    fi
done

# ==========================================
# Optional: create .gitignore content
# ==========================================
GITIGNORE_PATH="$BASE_PATH/.gitignore"

if [ ! -f "$GITIGNORE_PATH" ]; then
cat <<EOF > "$GITIGNORE_PATH"
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
EOF

    echo "[FILE CREATED] .gitignore (with content)"
fi

echo ""
echo "[OK] Project structure created successfully!"