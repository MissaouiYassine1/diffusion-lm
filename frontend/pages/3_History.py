"""
Page d'historique des générations.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from frontend.utils.state_manager import SessionState

st.set_page_config(
    page_title="Historique - Diffusion LM",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Historique des générations")

# Initialiser l'état
SessionState.init()

# Sidebar avec contrôles
with st.sidebar:
    st.subheader("⚙️ Contrôles")
    
    if st.button("🗑️ Effacer tout l'historique", type="secondary"):
        SessionState.clear_history()
        st.rerun()
    
    st.divider()
    
    st.subheader("📊 Statistiques")
    st.metric("Nombre de générations", len(st.session_state.history))
    
    if st.session_state.history:
        # Temps moyen
        avg_time = sum(h['metadata'].get('time_ms', 0) for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Temps moyen", f"{avg_time:.0f} ms")
        
        # Dernière génération
        last = st.session_state.history[0]
        st.caption(f"Dernière: {last['timestamp'][:16]}")

# Afficher l'historique
if not st.session_state.history:
    st.info("📭 Aucune génération dans l'historique. Générez du texte depuis la page principale!")
else:
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("🔍 Rechercher dans l'historique", placeholder="Mot-clé...")
    with col2:
        sort_by = st.selectbox("Trier par", ["Date (récent)", "Date (ancien)", "Longueur"])
    
    # Filtrer
    history = st.session_state.history.copy()
    if search:
        history = [h for h in history if search.lower() in h['prompt'].lower() or search.lower() in h['generated_text'].lower()]
    
    # Trier
    if sort_by == "Date (récent)":
        pass  # Déjà trié
    elif sort_by == "Date (ancien)":
        history.reverse()
    elif sort_by == "Longueur":
        history.sort(key=lambda x: len(x['generated_text']), reverse=True)
    
    # Afficher
    for i, entry in enumerate(history[:20]):
        with st.container():
            cols = st.columns([0.1, 0.7, 0.2])
            
            with cols[0]:
                # Favori
                is_fav = entry['id'] in st.session_state.favorites
                fav_icon = "⭐" if is_fav else "☆"
                if st.button(fav_icon, key=f"fav_{entry['id']}"):
                    SessionState.toggle_favorite(entry['id'])
                    st.rerun()
            
            with cols[1]:
                st.markdown(f"**📝 {entry['prompt'][:100]}**")
                st.caption(f"✨ {entry['generated_text'][:150]}...")
                st.caption(f"🕐 {entry['timestamp'][:19]} | ⏱️ {entry['metadata'].get('time_ms', 0):.0f} ms")
            
            with cols[2]:
                if st.button("📋 Utiliser", key=f"use_{entry['id']}"):
                    st.session_state.reuse_prompt = entry['prompt']
                    st.switch_page("app.py")
            
            st.divider()