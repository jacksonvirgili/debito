import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =================================================
# CONFIGURAÇÃO
# =================================================
st.set_page_config(page_title="Acompanhamento VLRAF", layout="wide")

# =================================================
# FERIADOS NACIONAIS
# =================================================
FERIADOS_BR = pd.to_datetime([
    "2024-01-01","2025-01-01","2026-01-01",
    "2024-02-12","2024-02-13","2025-03-03","2025-03-04","2026-02-16","2026-02-17",
    "2024-03-29","2025-04-18","2026-04-03",
    "2024-04-21","2025-04-21","2026-04-21",
    "2024-05-01","2025-05-01","2026-05-01",
    "2024-06-20","2025-06-19","2026-06-04",
    "2024-09-07","2025-09-07","2026-09-07",
    "2024-10-12","2025-10-12","2026-10-12",
    "2024-11-02","2025-11-02","2026-11-02",
    "2024-11-15","2025-11-15","2026-11-15",
    "2024-12-25","2025-12-25","2026-12-25",
])

# =================================================
# DADOS
# =================================================
@st.cache_data
def carregar_dados():
    df = pd.read_parquet("Acomp.parquet")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["DATA_EFETIVACAO"] = pd.to_datetime(df["DATA_EFETIVACAO"])
    return df

df = carregar_dados()

# =================================================
# FUNÇÕES
# =================================================
def adicionar_dia_util(df):
    df = df.copy()
    df = df[df["DATA_EFETIVACAO"].dt.weekday < 5]
    df = df[~df["DATA_EFETIVACAO"].isin(FERIADOS_BR)]
    df["MES"] = df["DATA_EFETIVACAO"].dt.strftime("%Y-%m")
    df["DIA_UTIL"] = (
        df.groupby("MES")["DATA_EFETIVACAO"]
        .rank(method="dense")
        .astype(int)
    )
    return df

# =================================================
# FILTROS
# =================================================
st.sidebar.title("Filtros")

regional = st.sidebar.selectbox(
    "Regional",
    ["Todas"] + sorted(df["REGIONAIS"].dropna().unique())
)
coordenador = st.sidebar.selectbox(
    "Coordenador",
    ["Todos"] + sorted(df["COORDENADOR"].dropna().unique())
)
loja = st.sidebar.selectbox(
    "Loja",
    ["Todas"] + sorted(df["DESCRICAO_LOJA"].dropna().unique())
)

if regional != "Todas":
    df = df[df["REGIONAIS"] == regional]
if coordenador != "Todos":
    df = df[df["COORDENADOR"] == coordenador]
if loja != "Todas":
    df = df[df["DESCRICAO_LOJA"] == loja]

df = adicionar_dia_util(df)

# =================================================
# TABS
# =================================================
tab1, tab2 = st.tabs(["Análise Diária", "Comparação entre Meses"])

# =================================================
# TAB 1 — MATPLOTLIB COM ESTILO
# =================================================
with tab1:
    st.subheader("Análise diária por classificação")

    grupo_produto = st.selectbox(
        "Grupo Produto",
        ["Todos"] + sorted(df["GRUPO PRODUTO"].unique())
    )

    classificacao = st.radio(
        "Classificar por:",
        ["COMISSAO_DIFERIDA", "TIPO PRODUTO"]
    )

    mes = st.selectbox("Mês", sorted(df["MES"].unique()))

    base = df[df["MES"] == mes]
    if grupo_produto != "Todos":
        base = base[base["GRUPO PRODUTO"] == grupo_produto]

    g = (
        base.groupby(["DIA_UTIL", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
    )
    g_acum = g.cumsum()

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    # ---- Diário
    g.plot(kind="bar", ax=axs[0])
    axs[0].set_title("VLRAF Diário")
    axs[0].grid(axis="y", alpha=0.3)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    # ---- Acumulado
    g_acum.plot(kind="bar", ax=axs[1])
    axs[1].set_title("VLRAF Acumulado")
    axs[1].grid(axis="y", alpha=0.3)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].set_xticklabels([f"D+{d}" for d in g.index], rotation=0)

    plt.tight_layout()
    st.pyplot(fig)

# =================================================
# TAB 2 — PLOTLY COM EIXOS ALINHADOS
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    c1, c2 = st.columns(2)
    mes_a = c1.selectbox("Mês A", meses)
    mes_b = c2.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

    base = df[df["MES"].isin([mes_a, mes_b])]
    produtos = sorted(base["GRUPO PRODUTO"].unique())

    for prod in produtos:
        st.markdown(f"### {prod}")
        sub = base[base["GRUPO PRODUTO"] == prod]

        g = (
            sub.groupby(["MES", "DIA_UTIL"])["VLRAF"]
            .sum()
            .unstack("MES", fill_value=0)
        )

        eixo_x = [f"D+{d}" for d in g.index]

        # -------- Gráfico principal
        fig_main = go.Figure()
        for mes in [mes_a, mes_b]:
            fig_main.add_bar(
                x=eixo_x,
                y=g.get(mes, 0),
                name=mes,
                hovertemplate="<b>%{x}</b><br>" + mes + ": R$ %{y:,.0f}<extra></extra>"
            )

        fig_main.update_layout(
            barmode="group",
            height=400,
            yaxis_title="VLRAF"
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # -------- Desvio alinhado
        g_desvio = g[(g[mes_a] != 0) & (g[mes_b] != 0)].copy()
        g_desvio["DESVIO"] = g_desvio[mes_b] - g_desvio[mes_a]

        fig_dev = go.Figure()
        fig_dev.add_bar(
            x=[f"D+{d}" for d in g_desvio.index],
            y=g_desvio["DESVIO"],
            marker_color=["green" if v >= 0 else "red" for v in g_desvio["DESVIO"]],
            hovertemplate="<b>%{x}</b><br>Desvio: R$ %{y:,.0f}<extra></extra>"
        )

        fig_dev.add_hline(y=0)

        fig_dev.update_layout(
            height=280,  # 👈 aumentado
            yaxis_title="Δ VLRAF",
            xaxis=dict(
                categoryorder="array",
                categoryarray=eixo_x
            ),
            showlegend=False
        )

        st.plotly_chart(fig_dev, use_container_width=True)
