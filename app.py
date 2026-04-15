import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Configuração da página
# ===============================
st.set_page_config(page_title="Acompanhamento VLRAF", layout="wide")

# ===============================
# Carregar dados
# ===============================
@st.cache_data
def carregar_dados():
    df = pd.read_parquet("Acomp.parquet")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df.loc[df["TIPO PRODUTO"] != "NOVO", "TIPO PRODUTO"] = "REFIN"
    df["DATA_EFETIVACAO"] = pd.to_datetime(df["DATA_EFETIVACAO"])

    return df


df = carregar_dados()

# ===============================
# Funções auxiliares
# ===============================
def formatar_reais(v):
    return f"R$ {v:,.0f}".replace(",", ".")


def anotar_barras(ax, x_pos, valores, desloc=0):
    if len(valores) == 0:
        return

    max_val = max(valores)

    for x, v in zip(x_pos, valores):
        if v <= 0:
            continue

        y = v if v < max_val * 0.12 else v * 0.5
        va = "bottom" if y == v else "center"
        color = "black" if y == v else "white"

        ax.text(
            x + desloc,
            y,
            formatar_reais(v),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=color,
            rotation=90,
        )


def dia_util_relativo(datas):
    datas = pd.to_datetime(datas)
    datas_unicas = sorted(d for d in datas.unique() if d.weekday() < 5)
    mapa = {data: i + 1 for i, data in enumerate(datas_unicas)}
    return datas.map(mapa)

# ===============================
# Sidebar — filtros globais
# ===============================
st.sidebar.title("Filtros Globais")

regional = st.sidebar.selectbox(
    "Regional",
    ["Todas"] + sorted(df["REGIONAIS"].dropna().unique())
)

if regional != "Todas":
    df = df[df["REGIONAIS"] == regional]

# ===============================
# Tabs
# ===============================
tab_analise, tab_comparacao = st.tabs([
    "Análise Diária",
    "Comparação Mensal (Grupo Produto)"
])

# ===============================
# TAB 1 — ANÁLISE DIÁRIA (SEU GRÁFICO ORIGINAL)
# ===============================
with tab_analise:
    st.subheader("VLRAF Diário e Acumulado")

    classificacao = st.radio(
        "Classificar por:",
        ["COMISSAO_DIFERIDA", "TIPO PRODUTO"]
    )

    mes = st.selectbox(
        "Mês",
        ["Todos"] + sorted(df["DATA_EFETIVACAO"].dt.strftime("%Y-%m").unique())
    )

    df_f = df.copy()

    if mes != "Todos":
        df_f = df_f[
            df_f["DATA_EFETIVACAO"].dt.strftime("%Y-%m") == mes
        ]

    if df_f.empty:
        st.warning("Sem dados.")
        st.stop()

    g = (
        df_f
        .groupby(["DATA_EFETIVACAO", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    g_acum = g.cumsum()

    classes = g.columns.tolist()
    x = np.arange(len(g))
    width = 0.8 / max(len(classes), 1)
    offsets = np.linspace(-width, width, len(classes))

    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for i, classe in enumerate(classes):
        axs[0].bar(x + offsets[i], g[classe], width, label=classe)
        anotar_barras(axs[0], x, g[classe], offsets[i])

    axs[0].set_title("VLRAF Diário")
    axs[0].legend()

    for i, classe in enumerate(classes):
        axs[1].bar(x + offsets[i], g_acum[classe], width, label=classe)
        anotar_barras(axs[1], x, g_acum[classe], offsets[i])

    axs[1].set_title("VLRAF Acumulado")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(g.index.day)
    axs[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# ===============================
# TAB 2 — COMPARAÇÃO ENTRE MESES (DIAS ÚTEIS)
# ===============================
with tab_comparacao:
    st.subheader("Comparação por Grupo Produto — Dias Úteis")

    meses = sorted(df["DATA_EFETIVACAO"].dt.strftime("%Y-%m").unique())

    col1, col2 = st.columns(2)
    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

    df_comp = df[
        df["DATA_EFETIVACAO"].dt.strftime("%Y-%m").isin([mes_a, mes_b])
    ].copy()

    df_comp["MES"] = df_comp["DATA_EFETIVACAO"].dt.strftime("%Y-%m")

    df_comp["DIA_UTIL"] = (
        df_comp
        .groupby("MES")["DATA_EFETIVACAO"]
        .transform(dia_util_relativo)
    )

    g_comp = (
        df_comp
        .groupby(["MES", "DIA_UTIL", "GRUPO PRODUTO"])["VLRAF"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(16, 6))

    for mes in [mes_a, mes_b]:
        dados = g_comp[g_comp["MES"] == mes]
        for grp in dados["GRUPO PRODUTO"].unique():
            sub = dados[dados["GRUPO PRODUTO"] == grp]
            ax.plot(
                sub["DIA_UTIL"],
                sub["VLRAF"],
                marker="o",
                label=f"{grp} — {mes}"
            )

    ax.set_xlabel("Dia Útil (D+)")
    ax.set_ylabel("VLRAF")
    ax.set_title("Comparação Mensal por Dia Útil")
    ax.legend()

    st.pyplot(fig)
