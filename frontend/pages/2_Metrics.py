"""
Dashboard de métriques et analyses.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Métriques - Diffusion LM",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard des métriques")

# Initialiser l'état
if 'history' not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.info("📭 Aucune donnée. Générez du texte pour voir les métriques!")
    st.stop()

# Convertir en DataFrame
df = pd.DataFrame(st.session_state.history)

# Nettoyer les données
for col in ['time_ms', 'steps_used', 'temperature']:
    if col not in df.columns:
        df[col] = 0

df['timestamp'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.Timestamp.now()
df['word_count'] = df['generated_text'].str.split().str.len() if 'generated_text' in df.columns else 0

# KPIs
st.subheader("📈 Indicateurs clés")
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric("Total générations", len(df))
with kpi_cols[1]:
    st.metric("Temps moyen", f"{df['time_ms'].mean():.0f} ms")
with kpi_cols[2]:
    st.metric("Longueur moyenne", f"{df['word_count'].mean():.0f} mots")
with kpi_cols[3]:
    last_7d = df[df['timestamp'] > datetime.now() - timedelta(days=7)] if 'timestamp' in df.columns else df
    st.metric("Activité (7j)", len(last_7d))

# Graphiques
col1, col2 = st.columns(2)

with col1:
    if 'timestamp' in df.columns:
        df_daily = df.set_index('timestamp').resample('D').size()
        fig = px.line(x=df_daily.index, y=df_daily.values, title="📅 Générations par jour")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x='time_ms', nbins=20, title="⏱️ Distribution du temps")
    st.plotly_chart(fig, use_container_width=True)

# Top générations
st.subheader("🏆 Meilleures générations")
if 'word_count' in df.columns:
    top = df.nlargest(5, 'word_count')[['prompt', 'generated_text', 'word_count', 'time_ms']]
    st.dataframe(top, use_container_width=True)

st.info("📊 Plus de métriques disponibles après plus de générations!")