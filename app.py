import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# =================================================
# CONFIGURAÇÃO
# =================================================
st.set_page_config(page_title="Acompanhamento VLRAF", layout="wide")

# =================================================
# FERIADOS NACIONAIS (MANUAL)
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
# CARREGAR DADOS
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
def formatar_reais(v):
    return f"R$ {v:,.0f}".replace(",", ".")

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
# SIDEBAR – FILTROS
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
# TAB 1 — ANÁLISE DIÁRIA
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

    mes = st.selectbox(
        "Mês",
        sorted(df["MES"].unique())
    )

    base = df[df["MES"] == mes]

    if grupo_produto != "Todos":
        base = base[base["GRUPO PRODUTO"] == grupo_produto]

    g = (
        base
        .groupby(["DIA_UTIL", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
    )

    g_acum = g.cumsum()

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    bars1 = g.plot(kind="bar", ax=axs[0])
    axs[0].set_title("VLRAF Diário")
    axs[0].legend()

    mplcursors.cursor(axs[0], hover=True).connect(
        "add",
        lambda sel: sel.annotation.set_text(
            formatar_reais(sel.target[1])
        )
    )

    bars2 = g_acum.plot(kind="bar", ax=axs[1])
    axs[1].set_title("VLRAF Acumulado")
    axs[1].set_xticklabels([f"D+{d}" for d in g.index], rotation=0)
    axs[1].legend()

    mplcursors.cursor(axs[1], hover=True).connect(
        "add",
        lambda sel: sel.annotation.set_text(
            formatar_reais(sel.target[1])
        )
    )

    plt.tight_layout()
    st.pyplot(fig)

# =================================================
# TAB 2 — DESVIO ENTRE MESES (SUBPLOTS POR PRODUTO)
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    col1, col2 = st.columns(2)
    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox(
        "Mês B", meses, index=1 if len(meses) > 1 else 0
    )

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

        # garante colunas
        g[mes_a] = g.get(mes_a, 0)
        g[mes_b] = g.get(mes_b, 0)

        # Mantém somente dias com valor nos dois meses
        g_desvio = g[(g[mes_a] != 0) & (g[mes_b] != 0)].copy()
        
        g_desvio["DESVIO"] = g_desvio[mes_b] - g_desvio[mes_a]
        cores = np.where(g_desvio["DESVIO"] >= 0, "green", "red")

        # ===== FIGURA COM GRID (principal + auxiliary) =====
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

        ax_main = fig.add_subplot(gs[0])
        ax_dev = fig.add_subplot(gs[1], sharex=ax_main)

        # -------- GRÁFICO PRINCIPAL --------
        g[[mes_a, mes_b]].plot(kind="bar", ax=ax_main)
        ax_main.set_title(f"{prod} – Comparação {mes_a} x {mes_b}")
        ax_main.set_ylabel("VLRAF")
        ax_main.legend(title="Mês")

        mplcursors.cursor(ax_main, hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(
                formatar_reais(sel.target[1])
            )
        )

        # -------- SUBPLOT DESVIO (MENOR) --------
        # posições completas do eixo X (todos os dias do gráfico principal)
        dias_completos = list(g.index)
        x_all = np.arange(len(dias_completos))
        
        # posições apenas dos dias válidos para desvio
        x_desvio = [dias_completos.index(d) for d in g_desvio.index]
        
        bars = ax_dev.bar(
            x_desvio,
            g_desvio["DESVIO"],
            color=cores,
            width=0.8
        )
        
        ax_dev.axhline(0, color="black", linewidth=1)
        ax_dev.set_ylabel("Δ VLRAF")
        ax_dev.set_xlabel("Dia útil")
        
        # mantém exatamente o mesmo eixo X do gráfico principal
        ax_dev.set_xlim(-0.5, len(dias_completos) - 0.5)
        ax_dev.set_xticks(x_all)
        ax_dev.set_xticklabels([f"D+{d}" for d in dias_completos], rotation=0)

        mplcursors.cursor(bars, hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"D+{int(sel.target[0])}\n{formatar_reais(sel.target[1])}"
            )
        )

        mplcursors.cursor(bars, hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"D+{int(sel.target[0])}\n{formatar_reais(sel.target[1])}"
            )
        )

        st.pyplot(fig)
