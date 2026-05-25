"""
Animations Lottie pour Streamlit.
"""

import json
import requests
from streamlit_lottie import st_lottie

def load_lottie_url(url: str):
    """Charger une animation Lottie depuis une URL"""
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def load_lottie_file(path: str):
    """Charger une animation Lottie depuis un fichier local"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

# URLs d'animations gratuites (LottieFiles)
DIFFUSION_ANIMATION_URL = "https://assets10.lottiefiles.com/packages/lf20_k9s2yf.json"
LOADING_ANIMATION_URL = "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
SUCCESS_ANIMATION_URL = "https://assets5.lottiefiles.com/packages/lf20_tszdug7y.json"

def show_diffusion_animation():
    """Afficher l'animation du processus de diffusion"""
    anim = load_lottie_url(DIFFUSION_ANIMATION_URL)
    if anim:
        st_lottie(anim, height=200, key="diffusion_anim")

def show_loading_animation():
    """Afficher une animation de chargement"""
    anim = load_lottie_url(LOADING_ANIMATION_URL)
    if anim:
        st_lottie(anim, height=150, key="loading_anim")

def show_success_animation():
    """Afficher une animation de succès"""
    anim = load_lottie_url(SUCCESS_ANIMATION_URL)
    if anim:
        st_lottie(anim, height=100, key="success_anim")
