"""
Page d'historique des générations.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="Historique - Diffusion LM",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Historique des générations")

# Initialiser l'état
if 'history' not in st.session_state:
    st.session_state.history = []

if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Sidebar avec contrôles
with st.sidebar:
    st.subheader("⚙️ Contrôles")
    
    if st.button("🗑️ Effacer tout l'historique", type="secondary"):
        st.session_state.history = []
        st.session_state.favorites = []
        st.rerun()
    
    st.divider()
    
    st.subheader("📊 Statistiques")
    st.metric("Nombre de générations", len(st.session_state.history))
    
    if st.session_state.history:
        # Temps moyen
        avg_time = sum(h.get('time_ms', 0) for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Temps moyen", f"{avg_time:.0f} ms")

# Afficher l'historique
if not st.session_state.history:
    st.info("📭 Aucune génération dans l'historique. Générez du texte depuis la page principale!")
else:
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("🔍 Rechercher", placeholder="Mot-clé...")
    with col2:
        sort_by = st.selectbox("Trier par", ["Date (récent)", "Date (ancien)", "Longueur"])
    
    # Filtrer
    history = st.session_state.history.copy()
    if search:
        history = [h for h in history if search.lower() in h.get('prompt', '').lower() or search.lower() in h.get('generated_text', '').lower()]
    
    # Trier
    if sort_by == "Date (récent)":
        pass
    elif sort_by == "Date (ancien)":
        history.reverse()
    elif sort_by == "Longueur":
        history.sort(key=lambda x: len(x.get('generated_text', '')), reverse=True)
    
    # Afficher
    for i, entry in enumerate(history[:20]):
        with st.container():
            cols = st.columns([0.7, 0.2, 0.1])
            
            with cols[0]:
                st.markdown(f"**📝 {entry.get('prompt', '')[:100]}**")
                st.caption(f"✨ {entry.get('generated_text', '')[:150]}...")
                st.caption(f"🕐 {entry.get('timestamp', '')[:19]} | ⏱️ {entry.get('time_ms', 0):.0f} ms")
            
            with cols[1]:
                if st.button("📋 Utiliser", key=f"use_{i}"):
                    st.session_state.reuse_prompt = entry.get('prompt', '')
                    st.switch_page("app.py")
            
            with cols[2]:
                is_fav = entry.get('id', i) in st.session_state.favorites
                fav_icon = "⭐" if is_fav else "☆"
                if st.button(fav_icon, key=f"fav_{i}"):
                    fav_id = entry.get('id', i)
                    if fav_id in st.session_state.favorites:
                        st.session_state.favorites.remove(fav_id)
                    else:
                        st.session_state.favorites.append(fav_id)
                    st.rerun()
            
            st.divider()