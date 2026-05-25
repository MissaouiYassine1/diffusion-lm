#!/bin/bash
# Script de test pour vérifier que tout fonctionne

set -e

echo "🧪 TEST DE DÉPLOIEMENT - DIFFUSION LM"
echo "====================================="

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Vérifier Python
echo -n "🔍 Vérification Python... "
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}OK${NC} ($(python3 --version))"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# 2. Vérifier pip
echo -n "🔍 Vérification pip... "
if command -v pip &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# 3. Vérifier Docker (optionnel)
echo -n "🔍 Vérification Docker... "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}Non disponible${NC}"
    DOCKER_AVAILABLE=false
fi

# 4. Vérifier les dépendances Python
echo "📦 Vérification des dépendances Python..."
pip list | grep -q fastapi && echo -e "  ${GREEN}✓ FastAPI${NC}" || echo -e "  ${RED}✗ FastAPI manquant${NC}"
pip list | grep -q streamlit && echo -e "  ${GREEN}✓ Streamlit${NC}" || echo -e "  ${RED}✗ Streamlit manquant${NC}"
pip list | grep -q torch && echo -e "  ${GREEN}✓ PyTorch${NC}" || echo -e "  ${RED}✗ PyTorch manquant${NC}"

# 5. Vérifier la structure des dossiers
echo "📁 Vérification de la structure..."
for dir in backend frontend models scripts; do
    if [ -d "$dir" ]; then
        echo -e "  ${GREEN}✓ $dir${NC}"
    else
        echo -e "  ${RED}✗ $dir manquant${NC}"
    fi
done

# 6. Vérifier les fichiers clés
echo "📄 Vérification des fichiers..."
for file in backend/main.py frontend/app.py docker-compose.yml; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓ $file${NC}"
    else
        echo -e "  ${RED}✗ $file manquant${NC}"
    fi
done

# 7. Tester l'import des modules
echo "🐍 Test des imports Python..."
python3 -c "
import sys
sys.path.append('.')
try:
    from backend.core.diffusion_model import DiffusionLM
    print('  ✓ DiffusionLM importé')
except Exception as e:
    print(f'  ✗ Erreur: {e}')
" 2>/dev/null

# 8. Docker Compose test
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "🐳 Test Docker Compose..."
    docker-compose config &>/dev/null && echo -e "  ${GREEN}✓ docker-compose.yml valide${NC}" || echo -e "  ${RED}✗ docker-compose.yml invalide${NC}"
fi

echo ""
echo "✅ Test terminé!"