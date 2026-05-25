"""
Visualisation interactive du processus de diffusion.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_noise_progression(diffusion_steps):
    """Graphique de progression du bruit"""
    
    df = pd.DataFrame(diffusion_steps)
    df['step_reversed'] = df['step'].max() - df['step']
    
    fig = go.Figure()
    
    # Courbe de bruit
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

def plot_token_heatmap(diffusion_steps, tokenizer=None, max_tokens=30):
    """Heatmap des tokens au fil des étapes"""
    
    # Extraire les tokens pour chaque étape
    steps_data = []
    for step in diffusion_steps:
        tokens = step.get('tokens', [])
        if isinstance(tokens, list):
            steps_data.append(tokens[:max_tokens])
    
    if not steps_data:
        return None
    
    # Créer la heatmap
    fig = go.Figure(data=go.Heatmap(
        z=steps_data,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title="Token ID")
    ))
    
    fig.update_layout(
        title="🔥 Évolution des tokens par étape",
        xaxis_title="Position dans la séquence",
        yaxis_title="Étape de diffusion",
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_step_animation(diffusion_steps):
    """Créer une animation des étapes (Slider)"""
    
    if not diffusion_steps:
        return None
    
    steps_to_show = diffusion_steps[::max(1, len(diffusion_steps)//10)]
    
    fig = go.Figure()
    
    frames = []
    for step in steps_to_show:
        frame_data = [go.Scatter(
            x=list(range(len(step['partial_text']))),
            y=[1] * len(step['partial_text']),
            mode='text',
            text=list(step['partial_text']),
            textfont=dict(size=14),
            showlegend=False
        )]
        frames.append(go.Frame(data=frame_data, name=f"step_{step['step']}"))
    
    fig.frames = frames
    
    # Données initiales
    initial_step = steps_to_show[0]
    fig.add_trace(go.Scatter(
        x=list(range(len(initial_step['partial_text']))),
        y=[1] * len(initial_step['partial_text']),
        mode='text',
        text=list(initial_step['partial_text']),
        textfont=dict(size=14),
        showlegend=False
    ))
    
    fig.update_layout(
        title="🎬 Animation du processus de diffusion",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])]
        )],
        height=200
    )
    
    return fig

def render_diffusion_steps(diffusion_steps):
    """Afficher toutes les visualisations"""
    
    if not diffusion_steps:
        st.info("Aucune étape de diffusion à afficher")
        return
    
    st.markdown("## 🔬 Analyse du processus de diffusion")
    
    # Onglets pour différentes visualisations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Progression", "📝 Détail des étapes", "🔥 Heatmap", "🎬 Animation"
    ])
    
    with tab1:
        fig = plot_noise_progression(diffusion_steps)
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques additionnelles
        col1, col2, col3 = st.columns(3)
        with col1:
            first_step = diffusion_steps[0]
            st.metric("Bruit initial", f"{first_step['noise_ratio']:.0%}")
        with col2:
            last_step = diffusion_steps[-1]
            st.metric("Bruit final", f"{last_step['noise_ratio']:.0%}")
        with col3:
            st.metric("Nombre d'étapes", len(diffusion_steps))
    
    with tab2:
        # Tableau des étapes
        df = pd.DataFrame(diffusion_steps)
        df['partial_text_short'] = df['partial_text'].str[:50] + '...'
        st.dataframe(
            df[['step', 'noise_ratio', 'num_masked', 'partial_text_short']],
            use_container_width=True,
            column_config={
                'step': 'Étape',
                'noise_ratio': st.column_config.NumberColumn('Bruit', format='%.0f%%'),
                'num_masked': 'Masqués',
                'partial_text_short': 'Texte partiel'
            }
        )
    
    with tab3:
        fig = plot_token_heatmap(diffusion_steps)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données de heatmap non disponibles")
    
    with tab4:
        fig = create_step_animation(diffusion_steps)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Animation non disponible")