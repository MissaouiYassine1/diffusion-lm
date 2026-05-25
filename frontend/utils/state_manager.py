"""
Gestion de l'état de session pour Streamlit.
"""

import streamlit as st
from datetime import datetime
import json
from typing import List, Dict, Any

class SessionState:
    """Gestionnaire d'état persistant dans la session"""
    
    @staticmethod
    def init():
        """Initialiser les variables de session"""
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
    
    @staticmethod
    def add_to_history(prompt: str, generated_text: str, metadata: Dict[str, Any]):
        """Ajouter une entrée à l'historique"""
        
        entry = {
            'id': len(st.session_state.history),
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'generated_text': generated_text,
            'metadata': metadata
        }
        
        st.session_state.history.insert(0, entry)
        
        # Limiter la taille
        max_size = st.session_state.settings.get('max_history', 50)
        if len(st.session_state.history) > max_size:
            st.session_state.history = st.session_state.history[:max_size]
        
        # Sauvegarder si auto_save actif
        if st.session_state.settings.get('auto_save', True):
            SessionState.save_history()
    
    @staticmethod
    def save_history():
        """Sauvegarder l'historique dans un fichier"""
        try:
            with open('.streamlit/history.json', 'w') as f:
                json.dump(st.session_state.history, f, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde historique: {e}")
    
    @staticmethod
    def load_history():
        """Charger l'historique depuis un fichier"""
        try:
            with open('.streamlit/history.json', 'r') as f:
                loaded = json.load(f)
                st.session_state.history = loaded
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Erreur chargement historique: {e}")
    
    @staticmethod
    def clear_history():
        """Vider l'historique"""
        st.session_state.history = []
        SessionState.save_history()
    
    @staticmethod
    def toggle_favorite(entry_id: int):
        """Ajouter/retirer des favoris"""
        if entry_id in st.session_state.favorites:
            st.session_state.favorites.remove(entry_id)
        else:
            st.session_state.favorites.append(entry_id)
    
    @staticmethod
    def get_history_as_dataframe():
        """Retourner l'historique comme DataFrame pandas"""
        import pandas as pd
        if not st.session_state.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['prompt_short'] = df['prompt'].str[:50] + '...'
        return df