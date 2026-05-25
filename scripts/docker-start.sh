#!/bin/bash
# Script pour démarrer l'application avec Docker Compose

echo "🚀 Démarrage de Diffusion LM avec Docker Compose..."

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

# Construire les images
echo "📦 Construction des images Docker..."
docker-compose build

# Démarrer les services
echo "▶️ Démarrage des services..."
docker-compose up -d

# Attendre que l'API soit prête
echo "⏳ Attente de l'API..."
sleep 5

# Vérifier que tout fonctionne
echo "🔍 Vérification des services..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API OK"
else
    echo "⚠️ API non disponible"
fi

curl -s http://localhost:8501 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Frontend OK"
else
    echo "⚠️ Frontend non disponible"
fi

echo ""
echo "🎉 Application démarrée!"
echo "📱 Frontend: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📚 Documentation API: http://localhost:8000/docs"
echo ""
echo "Pour arrêter: docker-compose down"