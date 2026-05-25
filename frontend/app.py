"""
Application Streamlit pour le Diffusion Language Model.
Interface professionnelle avec visualisations interactives.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime

# Import du client API
import sys
sys.path.append('.')
from frontend.utils.api_client import get_api_client

# Importer le state manager
from frontend.utils.state_manager import SessionState

# Initialiser au démarrage
SessionState.init()

# Configuration de la page (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Diffusion Language Model",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #4CAF50, #2196F3, #9C27B0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 1rem;
    color: white;
    text-align: center;
}
.step-box {
    background: #e8f4f8;
    border-left: 4px solid #2196F3;
    padding: 0.5rem;
    margin: 0.25rem 0;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# Initialiser le client API
api_client = get_api_client()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    
    # Paramètres du modèle
    st.subheader("🎛️ Paramètres de génération")
    steps = st.slider("Pas de diffusion", 10, 100, 30, 
                      help="Plus de pas = meilleure qualité mais plus lent")
    temperature = st.slider("Température", 0.1, 2.0, 0.8, 0.05,
                            help="Plus élevé = plus créatif")
    max_length = st.number_input("Longueur max (tokens)", 20, 200, 80)
    
    # Options d'affichage
    st.subheader("📊 Affichage")
    verbose_mode = st.checkbox("Mode étape par étape", value=False,
                               help="Afficher chaque étape du débrutage")
    show_metrics = st.checkbox("Afficher métriques", value=True)
    
    # API Status
    st.divider()
    st.subheader("🔌 Connexion API")
    
    # Tester la connexion API
    health = api_client.health_check()
    if health["success"]:
        st.success("✅ API connectée")
        st.caption(f"Status: {health['data'].get('status', 'ok')}")
    else:
        st.error("❌ API indisponible")
        st.caption("Lancez: uvicorn backend.main:app --reload")

# Header principal
st.markdown('<p class="main-header">🔄 Diffusion Language Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Génération de texte par diffusion progressive (bruit → débruiter)</p>', unsafe_allow_html=True)

# Organisation en colonnes
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de saisie
    prompt = st.text_area(
        "📝 **Votre prompt**",
        placeholder="Exemple: Le chat noir...",
        height=120
    )
    
    # Bouton de génération
    generate_btn = st.button("🚀 **Générer**", type="primary", use_container_width=True)

with col2:
    # Exemples rapides
    st.markdown("### 💡 Exemples")
    examples = [
        "le chat",
        "il était une fois",
        "dans un monde lointain",
        "la révolution de l'IA"
    ]
    
    for ex in examples:
        if st.button(f"📌 {ex}", key=ex, use_container_width=True):
            prompt = ex
            st.rerun()

# Zone de résultats
if generate_btn and prompt.strip():
    st.divider()
    st.markdown("## ✨ Résultat")
    
    with st.spinner("🔄 Génération en cours (diffusion étape par étape)..."):
        start_time = time.time()
        
        # Appel API via le client
        result = api_client.generate(
            prompt=prompt.strip(),
            steps=steps,
            temperature=temperature,
            max_length=max_length,
            verbose=verbose_mode
        )
        
        if result["success"]:
            data = result["data"]
            inference_time = (time.time() - start_time) * 1000
            
            # Afficher le texte généré
            st.markdown("### 📄 Texte généré")
            st.success(data["generated_text"])
            
            # Métriques si demandé
            if show_metrics:
                st.markdown("### 📊 Métriques")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Temps d'inférence", f"{data['inference_time_ms']:.0f} ms")
                with mcol2:
                    st.metric("Étapes utilisées", data["steps_used"])
                with mcol3:
                    st.metric("Longueur (mots)", data["length"])
                with mcol4:
                    st.metric("API + UI", f"{inference_time:.0f} ms")
            
            # Mode étape par étape
            if verbose_mode and data.get("diffusion_steps"):
                st.markdown("### 🔄 Processus de diffusion inversé")
                st.caption("Chaque étape montre la progression du débrutage")
                
                # Créer un expander pour chaque étape (limité aux 10 premières)
                for step in data["diffusion_steps"][:10]:
                    with st.expander(f"📌 Étape {step['step']} (bruit: {step['noise_ratio']:.0%})"):
                        st.code(step['partial_text'][:200])
                        st.progress(1 - step['noise_ratio'])
                
                if len(data["diffusion_steps"]) > 10:
                    st.info(f"... et {len(data['diffusion_steps']) - 10} étapes supplémentaires")
            
            # Visualisation graphique (optionnel)
            if verbose_mode and data.get("diffusion_steps"):
                st.markdown("### 📈 Progression du débrutage")
                df = pd.DataFrame(data["diffusion_steps"])
                fig = px.line(df, x="step", y="noise_ratio", 
                              title="Niveau de bruit par étape",
                              labels={"step": "Étape", "noise_ratio": "Ratio de bruit"})
                st.plotly_chart(fig, use_container_width=True)

            # Ajouter à l'historique
            SessionState.add_to_history(
                prompt=prompt.strip(),
                generated_text=data["generated_text"],
                metadata={
                    'time_ms': data['inference_time_ms'],
                    'steps': data['steps_used'],
                    'temperature': temperature,
                    'api_time': data['inference_time_ms']
                }
            )

        else:
            st.error(f"❌ Erreur: {result['error']}")

elif generate_btn and not prompt.strip():
    st.warning("⚠️ Veuillez entrer un prompt avant de générer.")

# Footer
st.divider()
st.caption(f"🤖 Diffusion Language Model API v1.0 | {datetime.now().strftime('%Y-%m-%d')} | "
           f"[GitHub](https://github.com/votre-username/diffusion-lm-scrum)")