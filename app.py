import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Configuração da página
# ===============================
st.set_page_config(page_title="Acompanhamento VLRAF", layout="wide")

# ===============================
# Feriados nacionais (MANUAL)
# ===============================
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
        ax.text(x + desloc, y, formatar_reais(v),
                ha="center", va=va, fontsize=9,
                fontweight="bold", color=color, rotation=90)

def adicionar_dia_util(df):
    df = df.copy()
    df = df[df["DATA_EFETIVACAO"].dt.weekday < 5]
    df = df[~df["DATA_EFETIVACAO"].isin(FERIADOS_BR)]
    df["MES"] = df["DATA_EFETIVACAO"].dt.to_period("M")
    df["DIA_UTIL"] = (
        df.groupby("MES")["DATA_EFETIVACAO"]
        .rank(method="dense").astype(int)
    )
    return df

def agregar(df, classificacao):
    return (
        df.groupby(["DIA_UTIL", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

# ===============================
# Sidebar – filtros globais
# ===============================
st.sidebar.title("Filtros")

regional = st.sidebar.selectbox("Regional", ["Todas"] + sorted(df["REGIONAIS"].dropna().unique()))
coordenador = st.sidebar.selectbox("Coordenador", ["Todos"] + sorted(df["COORDENADOR"].dropna().unique()))
loja = st.sidebar.selectbox("Loja", ["Todas"] + sorted(df["DESCRICAO_LOJA"].dropna().unique()))
atendente = st.sidebar.selectbox("Atendente", ["Todos"] + sorted(df["NOME_ATENDENTE"].dropna().unique()))
tipo_produto = st.sidebar.selectbox("Tipo Produto", ["Todos"] + sorted(df["TIPO PRODUTO"].dropna().unique()))
classificacao = st.sidebar.radio("Classificar barras por:", ["COMISSAO_DIFERIDA", "TIPO PRODUTO"])

# ===============================
# Aplicar filtros globais
# ===============================
df_f = df.copy()
if regional != "Todas": df_f = df_f[df_f["REGIONAIS"] == regional]
if coordenador != "Todos": df_f = df_f[df_f["COORDENADOR"] == coordenador]
if loja != "Todas": df_f = df_f[df_f["DESCRICAO_LOJA"] == loja]
if atendente != "Todos": df_f = df_f[df_f["NOME_ATENDENTE"] == atendente]
if tipo_produto != "Todos": df_f = df_f[df_f["TIPO PRODUTO"] == tipo_produto]

df_f = adicionar_dia_util(df_f)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["Visão Mensal", "Comparação entre Meses"])

# ==================================================
# TAB 1 – VISÃO MENSAL
# ==================================================
with tab1:
    st.subheader("Consignado x Débito — Mês selecionado")

    mes = st.selectbox(
        "Mês",
        sorted(df_f["DATA_EFETIVACAO"].dt.strftime("%Y-%m").unique())
    )

    base = df_f[df_f["DATA_EFETIVACAO"].dt.strftime("%Y-%m") == mes]

    cons = agregar(base[base["GRUPO PRODUTO"] == "CONSIGNADO"], classificacao)
    deb = agregar(base[base["GRUPO PRODUTO"] == "DÉBITO"], classificacao)

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    def plot(ax, g, titulo):
        classes = g.columns.tolist()
        x = np.arange(len(g))
        width = 0.8 / max(len(classes), 1)
        offsets = np.linspace(-width, width, len(classes))
        for i, c in enumerate(classes):
            ax.bar(x + offsets[i], g[c], width, label=c)
            anotar_barras(ax, x, g[c], offsets[i])
        ax.set_title(titulo)
        ax.legend()

    plot(axs[0], cons, "CONSIGNADO")
    plot(axs[1], deb, "DÉBITO")

    axs[1].set_xticks(range(len(cons.index)))
    axs[1].set_xticklabels([f"D+{i}" for i in cons.index])
    axs[1].set_xlabel("Dia útil")

    plt.tight_layout()
    st.pyplot(fig)

# ==================================================
# TAB 2 – COMPARAÇÃO ENTRE MESES
# ==================================================
with tab2:
    st.subheader("Comparação entre meses por Dia Útil")

    meses = sorted(df_f["MES"].astype(str).unique())
    col1, col2 = st.columns(2)
    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox("Mês B", meses, index=1)

    base = df_f[df_f["MES"].astype(str).isin([mes_a, mes_b])]

    cons = agregar(base[base["GRUPO PRODUTO"] == "CONSIGNADO"], classificacao)
    deb = agregar(base[base["GRUPO PRODUTO"] == "DÉBITO"], classificacao)

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    plot(axs[0], cons, f"CONSIGNADO — {mes_a} x {mes_b}")
    plot(axs[1], deb, f"DÉBITO — {mes_a} x {mes_b}")

    max_dia = max(cons.index.max(), deb.index.max())
    axs[1].set_xticks(range(1, max_dia + 1))
    axs[1].set_xticklabels([f"D+{i}" for i in range(1, max_dia + 1)])
    axs[1].set_xlabel("Dia útil")

    plt.tight_layout()
    st.pyplot(fig)
