"""
Application Streamlit pour le Diffusion Language Model.
Interface professionnelle avec visualisations interactives.
Version complète pour Issue #19 - Dashboard métriques intégré.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# ============================================================================
# CONFIGURATION DE LA PAGE (doit être la première commande Streamlit)
# ============================================================================

st.set_page_config(
    page_title="Diffusion Language Model",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# IMPORTS DES COMPOSANTS (avec gestion d'erreurs)
# ============================================================================

# Tentative d'import des modules optionnels
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

try:
    from streamlit_extras.metric_cards import style_metric_cards
    METRIC_CARDS_AVAILABLE = True
except ImportError:
    METRIC_CARDS_AVAILABLE = False


# ============================================================================
# CONFIGURATION API
# ============================================================================

API_URL = st.secrets.get("API_URL", "http://localhost:8000")


# ============================================================================
# CSS PERSONNALISÉ (Design professionnel)
# ============================================================================

st.markdown("""
<style>
    /* Variables et animations */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header principal */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #10B981, #F59E0B, #EF4444, #4F46E5);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        animation: fadeIn 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 30px -10px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Box des étapes de diffusion */
    .step-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        transition: all 0.2s;
    }
    
    .step-box:hover {
        background: linear-gradient(135deg, #667eea25 0%, #764ba225 100%);
        transform: translateX(5px);
    }
    
    /* Boutons personnalisés */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px -5px rgba(102,126,234,0.4);
    }
    
    /* Sidebar stylisée */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #F3F4F6;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-radius: 0.5rem;
        font-weight: 500;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    
    /* Success/Info/Warning */
    .stAlert {
        border-radius: 0.5rem;
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6B7280;
        font-size: 0.8rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# GESTION DE L'ÉTAT DE SESSION (Session State)
# ============================================================================

def init_session_state():
    """Initialiser toutes les variables de session"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'theme': 'light',
            'auto_save': True,
            'max_history': 50
        }
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'reuse_prompt' not in st.session_state:
        st.session_state.reuse_prompt = None

init_session_state()


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def add_to_history(prompt: str, generated_text: str, metadata: Dict[str, Any]):
    """Ajouter une entrée à l'historique"""
    entry = {
        'id': len(st.session_state.history),
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'generated_text': generated_text,
        **metadata
    }
    st.session_state.history.insert(0, entry)
    
    # Limiter la taille
    max_size = st.session_state.settings.get('max_history', 50)
    if len(st.session_state.history) > max_size:
        st.session_state.history = st.session_state.history[:max_size]


def get_api_client():
    """Retourner un client API simple"""
    class APIClient:
        def __init__(self, base_url):
            self.base_url = base_url
        
        def health_check(self):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return {"status": "ok", "data": response.json()}
                return {"status": "error", "error": f"HTTP {response.status_code}"}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        def generate(self, prompt, steps=30, temperature=0.8, max_length=100, verbose=False):
            payload = {
                "prompt": prompt,
                "steps": steps,
                "temperature": temperature,
                "max_length": max_length,
                "verbose": verbose
            }
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    timeout=120
                )
                if response.status_code == 200:
                    return {"success": True, "data": response.json()}
                return {"success": False, "error": f"HTTP {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    return APIClient(API_URL)


# ============================================================================
# FONCTIONS DE VISUALISATION
# ============================================================================

def plot_noise_progression(diffusion_steps):
    """Graphique de progression du bruit"""
    if not diffusion_steps:
        return None
    
    df = pd.DataFrame(diffusion_steps)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['noise_ratio'],
        mode='lines+markers',
        name='Niveau de bruit',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.2)'
    ))
    
    fig.update_layout(
        title="📉 Progression du débrutage",
        xaxis_title="Étape de diffusion",
        yaxis_title="Ratio de bruit",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def render_diffusion_steps(diffusion_steps):
    """Afficher les étapes de diffusion"""
    if not diffusion_steps:
        st.info("Aucune étape de diffusion à afficher")
        return
    
    st.markdown("## 🔬 Analyse du processus de diffusion")
    
    tab1, tab2 = st.tabs(["📉 Progression du bruit", "📝 Détail des étapes"])
    
    with tab1:
        fig = plot_noise_progression(diffusion_steps)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        for step in diffusion_steps[:8]:
            with st.expander(f"📌 Étape {step['step']} (bruit: {step['noise_ratio']:.0%})"):
                st.code(step.get('partial_text', '')[:200])
                st.progress(1 - step['noise_ratio'])


# ============================================================================
# PAGE PRINCIPALE
# ============================================================================

# Header
st.markdown('<p class="main-header">🔄 Diffusion Language Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Génération de texte par diffusion progressive (bruit → débruiter)</p>', unsafe_allow_html=True)

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
    verbose_mode = st.checkbox("Mode étape par étape", value=False)
    show_metrics = st.checkbox("Afficher métriques", value=True)
    
    # État de l'API
    st.divider()
    st.subheader("🔌 Connexion API")
    api_client = get_api_client()
    health = api_client.health_check()
    
    if health["status"] == "ok":
        st.success("✅ API connectée")
    else:
        st.error("❌ API indisponible")
        st.caption("Lancez: uvicorn backend.main:app --reload")
    
    # Navigation
    st.divider()
    st.subheader("📱 Navigation")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("🏠 Accueil", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    with col_nav2:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = 'metrics'
            st.rerun()

# ============================================================================
# AFFICHAGE DE LA PAGE SELON LA NAVIGATION
# ============================================================================

if st.session_state.page == 'metrics':
    # ========================================================================
    # DASHBOARD MÉTRIQUES (PAGE 2)
    # ========================================================================
    st.markdown("## 📊 Dashboard des métriques")
    
    if not st.session_state.history:
        st.info("📭 Aucune donnée. Générez du texte pour voir les métriques!")
    else:
        # Convertir en DataFrame
        df = pd.DataFrame(st.session_state.history)
        
        # Nettoyer les données
        for col in ['time_ms', 'steps_used', 'temperature']:
            if col not in df.columns:
                df[col] = 0
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df['timestamp'] = datetime.now()
        
        df['word_count'] = df['generated_text'].str.split().str.len() if 'generated_text' in df.columns else 0
        df['prompt_len'] = df['prompt'].str.len() if 'prompt' in df.columns else 0
        
        # KPIs
        st.markdown("### 📈 Indicateurs clés")
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            if METRIC_CARDS_AVAILABLE:
                style_metric_cards()
            st.metric("Total générations", len(df))
        with kpi_cols[1]:
            st.metric("Temps moyen", f"{df['time_ms'].mean():.0f} ms")
        with kpi_cols[2]:
            st.metric("Longueur moyenne", f"{df['word_count'].mean():.0f} mots")
        with kpi_cols[3]:
            if 'timestamp' in df.columns:
                last_7d = df[df['timestamp'] > datetime.now() - timedelta(days=7)]
                st.metric("Activité (7j)", len(last_7d))
            else:
                st.metric("Activité (7j)", len(df))
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            if 'timestamp' in df.columns and len(df) > 1:
                df_daily = df.set_index('timestamp').resample('D').size()
                fig = px.line(x=df_daily.index, y=df_daily.values, title="📅 Générations par jour")
                fig.update_layout(xaxis_title="Date", yaxis_title="Nombre")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Plus de données nécessaires pour ce graphique")
        
        with col2:
            fig = px.histogram(df, x='time_ms', nbins=20, title="⏱️ Distribution du temps d'inférence")
            fig.update_layout(xaxis_title="Temps (ms)", yaxis_title="Fréquence")
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig = px.scatter(df, x='prompt_len', y='word_count', title="📏 Longueur prompt vs génération",
                           labels={'prompt_len': 'Longueur prompt', 'word_count': 'Mots générés'},
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            if 'temperature' in df.columns:
                fig = px.box(df, x='temperature', y='time_ms', title="🌡️ Température vs temps")
                st.plotly_chart(fig, use_container_width=True)
        
        # Top générations
        st.markdown("### 🏆 Meilleures générations")
        if 'word_count' in df.columns and len(df) > 0:
            top_cols = ['prompt', 'generated_text', 'word_count', 'time_ms']
            available_cols = [c for c in top_cols if c in df.columns]
            top = df.nlargest(5, 'word_count')[available_cols]
            st.dataframe(top, use_container_width=True)
        
        # Export
        st.markdown("### 📥 Export des données")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Télécharger les métriques (CSV)",
            data=csv,
            file_name=f"diffusion_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Bouton retour
    if st.button("← Retour à l'accueil"):
        st.session_state.page = 'home'
        st.rerun()

else:
    # ========================================================================
    # PAGE PRINCIPALE - GÉNÉRATION
    # ========================================================================
    
    # Vérifier s'il y a un prompt à réutiliser
    if st.session_state.reuse_prompt:
        prompt_default = st.session_state.reuse_prompt
        st.session_state.reuse_prompt = None
    else:
        prompt_default = ""
    
    # Zone de saisie
    prompt = st.text_area(
        "📝 **Votre prompt**",
        value=prompt_default,
        placeholder="Exemple: Le chat noir...\nil était une fois...\ndans un monde lointain...",
        height=100
    )
    
    # Organisation en colonnes
    col_gen1, col_gen2 = st.columns([3, 1])
    
    with col_gen1:
        generate_btn = st.button("🚀 **Générer du texte**", type="primary", use_container_width=True)
    
    with col_gen2:
        # Exemples rapides
        with st.popover("💡 Exemples"):
            examples = [
                "le chat",
                "il était une fois",
                "dans un monde lointain",
                "la révolution de l'IA",
                "Le soleil se levait à peine quand..."
            ]
            for ex in examples:
                if st.button(f"📌 {ex}", key=ex):
                    prompt = ex
                    st.rerun()
    
    # Zone de résultats
    if generate_btn and prompt.strip():
        with st.spinner("🔄 Génération en cours (processus de diffusion progressive)..."):
            start_time = time.time()
            
            result = api_client.generate(
                prompt=prompt.strip(),
                steps=steps,
                temperature=temperature,
                max_length=max_length,
                verbose=verbose_mode
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            if result["success"]:
                data = result["data"]
                
                st.markdown("### ✨ Texte généré")
                st.success(data["generated_text"])
                
                # Métriques
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
                        st.metric("Temps total", f"{inference_time:.0f} ms")
                
                # Ajouter à l'historique
                add_to_history(
                    prompt=prompt.strip(),
                    generated_text=data["generated_text"],
                    metadata={
                        'time_ms': data['inference_time_ms'],
                        'steps_used': data['steps_used'],
                        'temperature': temperature,
                        'length': data['length']
                    }
                )
                
                # Mode étape par étape
                if verbose_mode and data.get("diffusion_steps"):
                    render_diffusion_steps(data["diffusion_steps"])
                
                # Animation de succès (si disponible)
                if LOTTIE_AVAILABLE:
                    st.balloons()
                
            else:
                st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                st.info("💡 Vérifiez que l'API est démarrée: `uvicorn backend.main:app --reload`")
    
    elif generate_btn and not prompt.strip():
        st.warning("⚠️ Veuillez entrer un prompt avant de générer.")
    
    # ========================================================================
    # SECTION D'HISTORIQUE RÉCENT (en bas de page)
    # ========================================================================
    
    if st.session_state.history:
        st.divider()
        st.markdown("## 📜 Générations récentes")
        
        recent = st.session_state.history[:5]
        for entry in recent:
            with st.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**📝 {entry.get('prompt', '')[:80]}...**")
                    st.caption(f"✨ {entry.get('generated_text', '')[:120]}...")
                    st.caption(f"🕐 {entry.get('timestamp', '')[:19]} | ⏱️ {entry.get('time_ms', 0):.0f} ms")
                with cols[1]:
                    if st.button("📋 Utiliser", key=f"reuse_{entry.get('id', 0)}"):
                        st.session_state.reuse_prompt = entry.get('prompt', '')
                        st.rerun()
                st.divider()
        
        if len(st.session_state.history) > 5:
            if st.button("📜 Voir tout l'historique"):
                st.session_state.page = 'metrics'
                st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f'<div class="footer">'
    f'🤖 <strong>Diffusion Language Model</strong> v1.0 | '
    f'Génération par diffusion progressive | '
    f'{datetime.now().strftime("%Y-%m-%d")}<br>'
    f'📊 {len(st.session_state.history)} générations enregistrées | '
    f'⭐ {len(st.session_state.favorites)} favoris'
    f'</div>',
    unsafe_allow_html=True
)