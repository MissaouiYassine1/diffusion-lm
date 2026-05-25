# diffusion-lm
Academic project: Diffusion Language Models for long-text generation. Bidirectional transformer trained with mask-based diffusion. REST API + interactive dashboard. Code, docs, and demo included.
# 🔄 Diffusion Language Model

> Génération de texte par diffusion progressive — du bruit au texte cohérent, étape par étape.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![License: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg?style=for-the-badge)

---

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Technologies Utilisées](#-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Architecture du Projet](#-architecture-du-projet)
- [API Endpoints](#-api-endpoints)
- [Roadmap](#-roadmap)
- [Contribution](#-contribution)
- [Licence](#-licence)
- [Contact](#-contact)

---

## 🚀 À Propos

**Diffusion Language Model** est une implémentation complète d'un modèle de diffusion appliqué à la génération de texte en langage naturel.

Contrairement aux modèles autoregressifs classiques comme GPT qui génèrent mot par mot de gauche à droite, ce modèle utilise un **processus de diffusion progressive** : on part d'une séquence totalement bruitée (texte aléatoire) et on la débruite étape par étape pour obtenir un texte cohérent.

**Pourquoi ce projet ?** Il résout plusieurs limitations des modèles traditionnels :

- ♻️ **Correction possible** — capacité à revenir en arrière pendant la génération
- 📝 **Cohérence globale** — meilleure cohérence sur les longs textes
- ⚡ **Génération parallèle** — traitement simultané plutôt que séquentiel

Le projet inclut une **API REST** (FastAPI), une **interface utilisateur interactive** (Streamlit) et une **containerisation Docker** pour un déploiement facile.

---

## 🛠 Technologies Utilisées

| Technologie | Rôle |
|---|---|
| [Python 3.10+](https://www.python.org/) | Langage principal |
| [PyTorch 2.0+](https://pytorch.org/) | Framework deep learning — Transformer bidirectionnel |
| [FastAPI](https://fastapi.tiangolo.com/) | API REST avec validation Pydantic et documentation Swagger |
| [Streamlit](https://streamlit.io/) | Interface utilisateur avec visualisations Plotly |
| [Uvicorn](https://www.uvicorn.org/) | Serveur ASGI haute performance |
| [Docker & Compose](https://www.docker.com/) | Containerisation et orchestration multi-services |
| [Nginx](https://nginx.org/) | Reverse proxy / load balancing |
| [Redis](https://redis.io/) | Cache des résultats de génération (optionnel) |
| [Plotly](https://plotly.com/) | Visualisations interactives du dashboard |
| [Pandas](https://pandas.pydata.org/) | Manipulation des données d'historique |

---

## ⚙️ Installation

### Prérequis

- Python **3.10+**
- Docker & Docker Compose *(optionnel mais recommandé)*
- Git

### 🐳 Avec Docker Compose *(recommandé)*

```bash
# Cloner le dépôt
git clone https://github.com/MissaouiYassine1/diffusion-lm.git
cd diffusion-lm-scrum

# Configurer les variables d'environnement
cp .env.example .env

# Lancer tous les services
docker-compose up -d --build
```

### 🐍 Manuellement (sans Docker)

```bash
# Cloner le dépôt
git clone https://github.com/MissaouiYassine1/diffusion-lm.git
cd diffusion-lm

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
```

---

## 🖥 Utilisation

### Démarrage manuel (deux terminaux)

```bash
# Terminal 1 — API FastAPI
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Interface Streamlit
streamlit run frontend/app.py
```

### Accès aux interfaces

| Service | URL |
|---|---|
| Interface Streamlit | http://localhost:8501 |
| Documentation API (Swagger) | http://localhost:8000/docs |
| Documentation API (ReDoc) | http://localhost:8000/redoc |

### Exemple d'utilisation via `curl`

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "le chat", "steps": 30, "temperature": 0.8}'
```

### Workflow dans l'interface

1. Ouvrir l'interface sur **http://localhost:8501**
2. Entrer un prompt *(ex : "le chat", "il était une fois")*
3. Ajuster les paramètres dans la barre latérale :
   - **Pas de diffusion** — qualité vs rapidité
   - **Température** — niveau de créativité
   - **Longueur maximale** — taille de la sortie
4. Cliquer sur **"Générer du texte"**
5. Activer **"Mode étape par étape"** pour visualiser le processus de débrutage

---

## 🏗 Architecture du Projet

```
diffusion-lm-scrum/
├── backend/
│   ├── main.py                  # Point d'entrée FastAPI
│   ├── config.py                # Configuration & variables d'environnement
│   ├── api/routes/
│   │   ├── health.py            # GET /health
│   │   └── generation.py        # POST /generate, /batch_generate
│   ├── services/
│   │   └── generation_service.py
│   └── middleware/
│       └── rate_limiter.py
├── frontend/
│   ├── app.py                   # Application Streamlit principale
│   ├── components/
│   │   └── diffusion_viewer.py  # Visualisation des étapes
│   ├── assets/
│   │   └── animations.py        # Animations Lottie
│   └── utils/
│       ├── api_client.py
│       └── state_manager.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

### Composants clés

**Modèle Diffusion LM**
- Tokenizer au niveau caractère avec tokens spéciaux (`PAD`, `MASK`, `UNK`)
- Transformer à **attention bidirectionnelle** (non-causale)
- **Forward process** — ajout progressif de bruit par masquage aléatoire
- **Backward process** — débrutage itératif par prédiction des tokens masqués

**Infrastructure**
- Docker multi-conteneurs (API · Streamlit · Nginx · Redis)
- Rate limiting via SlowAPI *(10 requêtes/minute)*
- CORS configuré pour la communication frontend ↔ backend

---

## 📡 API Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Racine — liens vers la documentation |
| `GET` | `/health` | État de l'API et du modèle chargé |
| `POST` | `/generate` | Génération synchrone à partir d'un prompt |
| `POST` | `/batch_generate` | Génération par lots (1–10 prompts) |
| `GET` | `/docs` | Interface Swagger interactive |

### `POST /generate` — Paramètres

```json
{
  "prompt": "le chat",
  "steps": 30,
  "temperature": 0.8,
  "max_length": 100,
  "verbose": false
}
```

### Réponse

```json
{
  "generated_text": "le chat noir dormait...",
  "inference_time_ms": 342,
  "steps_used": 30,
  "length": 12,
  "diffusion_steps": []
}
```

---

## 📍 Roadmap

- [x] **Phase 1** — Modèle de diffusion basique *(tokenizer, attention bidirectionnelle, forward/backward)*
- [x] **Phase 2** — API FastAPI *(endpoints, Pydantic, rate limiting, Swagger)*
- [x] **Phase 3** — Interface Streamlit *(Plotly, historique, dashboard, animations)*
- [x] **Phase 4** — Dockerisation complète *(docker-compose, variables d'environnement)*
- [ ] **Phase 5** — Génération de contenu IA, présentation finale, dépôt Classroom

---

## 🤝 Contribution

Ce projet suit le **GitHub Flow** pour la collaboration :

1. Créez votre branche de fonctionnalité
   ```bash
   git checkout -b feature/nom-fonctionnalite
   ```
2. Committez avec des messages clairs et atomiques
   ```bash
   git commit -m "feat: description de la fonctionnalité"
   ```
   Préfixes standards : `feat` · `fix` · `docs` · `style` · `refactor` · `test` · `chore`

3. Poussez sur GitHub
   ```bash
   git push origin feature/nom-fonctionnalite
   ```
4. Ouvrez une **Pull Request** et effectuez une revue de code
5. Mergez après validation — la branche `main` doit **toujours rester déployable**
6. Supprimez la branche après merge

> ⚠️ Chaque commit doit être **atomique** (une seule chose par commit).

---

## 📄 Licence

Distribué sous la licence **MIT**. Voir le fichier [`LICENSE`](./LICENSE) pour plus d'informations.

---

## ✉️ Contact

**Votre Nom** — [yassine.missaoui.email@enis.tn](mailto:yassine.missaoui.email@enis.tn)

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/votre-pseudo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/votre-profil)

🔗 **Lien du projet** : [https://github.com/MissaouiYassine1/diffusion-lm](https://github.com/votre-pseudo/diffusion-lm-scrum)