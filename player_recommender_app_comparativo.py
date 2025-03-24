
import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from mplsoccer import PyPizza

st.title("⚽ Recomendador de Jugadores por Perfil y Valor")

try:
    df = pd.read_pickle("df_recommender_ready.pkl")

    selected_features = [col for col in df.columns if ("/90" in col or "%" in col) and df[col].dtype != "O"]
    radar_features = ['npxG/90', 'xA/90', 'KeyPass/90', 'Touches/90', 'PassCmp%', 'DribPast/90', 'TklW/90']

    st.sidebar.title("Filtros")
    stat_filter = st.sidebar.selectbox("¿Querés priorizar alguna estadística?", [None] + selected_features)

    jugador_base = st.selectbox("Seleccioná un jugador para reemplazar:", df['Player'].dropna().unique())
    base_row = df[df['Player'] == jugador_base].iloc[0]
    rol = base_row['PredictedRole']
    valor_base = base_row['MarketValueEUR']
    vec_base = base_row[selected_features].values.astype(float)

    st.markdown(f"### 🧠 Perfil de {jugador_base}")
    st.markdown(f"- Rol estimado: **{rol}**")
    st.markdown(f"- Valor de mercado: **€{int(valor_base):,}**")
    st.markdown(f"- Principales estadísticas:")
    st.markdown(f"  - xG: `{base_row['npxG/90']:.2f}`")
    st.markdown(f"  - xA: `{base_row['xA/90']:.2f}`")
    st.markdown(f"  - Key Passes: `{base_row['KeyPass/90']:.2f}`")
    st.markdown(f"  - PassCmp%: `{base_row['PassCmp%']:.2f}`")

    candidatos = df[(df['PredictedRole'] == rol) & (df['Player'] != jugador_base) & (~df['MarketValueEUR'].isna())].copy()
    candidatos = candidatos.dropna(subset=selected_features)
    vectores = candidatos[selected_features].values.astype(float)
    candidatos['Distancia'] = np.linalg.norm(vectores - vec_base, axis=1)

    if stat_filter:
        peso = 2.0
        diff = np.abs(candidatos[stat_filter] - base_row[stat_filter])
        candidatos['Distancia'] -= peso * diff

    candidatos = candidatos.sort_values('Distancia')
    barato = candidatos[candidatos['MarketValueEUR'] < valor_base].head(1)
    similar = candidatos[np.isclose(candidatos['MarketValueEUR'], valor_base, rtol=0.25)].head(1)
    caro = candidatos[candidatos['MarketValueEUR'] > valor_base].head(1)
    recs = pd.concat([barato, similar, caro])

    st.subheader(f"💡 Reemplazos sugeridos para {jugador_base} ({rol})")
    for _, row in recs.iterrows():
        st.markdown(f"**{row['Player']}** (€{int(row['MarketValueEUR']):,})")
        st.markdown(f"xG: `{row['npxG/90']:.2f}`, xA: `{row['xA/90']:.2f}`, Key Passes: `{row['KeyPass/90']:.2f}`")
        st.markdown('---')

    st.info("¿Querés afinar la búsqueda? Probá seleccionando una estadística en la barra lateral.")

    st.subheader("📊 Comparación visual de rendimiento (percentiles dentro del rol)")

    df_role = df[df["PredictedRole"] == rol].dropna(subset=radar_features)
    jugadores = [base_row] + [df[df["Player"] == name].iloc[0] for name in recs["Player"]]

    colors = ["#1A78CF", "#FF7F0E", "#2CA02C", "#D62728"]
    labels = ["Base"] + ["Barato", "Similar", "Caro"]
    params = ["npxG", "xA", "Key Passes", "Touches", "Pass%", "Dribbled Past", "Tackles Won"]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})
    theta = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    theta += theta[:1]

    for i, jugador in enumerate(jugadores):
        percentiles = [int(stats.percentileofscore(df_role[stat], jugador[stat])) for stat in radar_features]
        percentiles += percentiles[:1]
        ax.plot(theta, percentiles, color=colors[i], label=f"{jugador['Player']} ({labels[i]})", linewidth=2)
        ax.fill(theta, percentiles, color=colors[i], alpha=0.1)

    ax.set_ylim(0, 100)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(params)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([20, 40, 60, 80, 100])
    ax.set_title("Radar de Percentiles Comparado", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

except Exception as e:
    st.error("💥 Ocurrió un error al cargar la aplicación.")
    st.exception(e)
