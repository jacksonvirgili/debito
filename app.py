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
    # FIXOS
    "2024-01-01", "2025-01-01", "2026-01-01",  # Confraternização Universal
    "2024-04-21", "2025-04-21", "2026-04-21",  # Tiradentes
    "2024-05-01", "2025-05-01", "2026-05-01",  # Dia do Trabalho
    "2024-09-07", "2025-09-07", "2026-09-07",  # Independência
    "2024-10-12", "2025-10-12", "2026-10-12",  # Nossa Senhora Aparecida
    "2024-11-02", "2025-11-02", "2026-11-02",  # Finados
    "2024-11-15", "2025-11-15", "2026-11-15",  # Proclamação da República
    "2024-12-25", "2025-12-25", "2026-12-25",  # Natal

    # MÓVEIS (definidos manualmente)
    "2024-02-12", "2025-03-03", "2026-02-16",  # Carnaval (segunda)
    "2024-02-13", "2025-03-04", "2026-02-17",  # Carnaval (terça)
    "2024-03-29", "2025-04-18", "2026-04-03",  # Sexta-feira Santa
    "2024-06-20", "2025-06-19", "2026-06-04",  # Corpus Christi
])

# ===============================
# Carregar dados
# ===============================
@st.cache_data
def carregar_dados():
    df = pd.read_parquet("Acomp.parquet")

    # Remover colunas lixo
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalizar tipo produto
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
            rotation=90
        )


def adicionar_dia_util(df, coluna_data):
    df = df.copy()

    df[coluna_data] = pd.to_datetime(df[coluna_data])

    # Remover sábado/domingo
    df = df[df[coluna_data].dt.weekday < 5]

    # Remover feriados
    df = df[~df[coluna_data].isin(FERIADOS_BR)]

    df["MES"] = df[coluna_data].dt.to_period("M")

    df["DIA_UTIL"] = (
        df
        .groupby("MES")[coluna_data]
        .rank(method="dense")
        .astype(int)
    )

    return df

# ===============================
# Sidebar — filtros
# ===============================
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

atendente = st.sidebar.selectbox(
    "Atendente",
    ["Todos"] + sorted(df["NOME_ATENDENTE"].dropna().unique())
)

tipo_produto = st.sidebar.selectbox(
    "Tipo Produto",
    ["Todos"] + sorted(df["TIPO PRODUTO"].dropna().unique())
)

mes = st.sidebar.selectbox(
    "Mês",
    ["Todos"] + sorted(df["DATA_EFETIVACAO"].dt.strftime("%Y-%m").unique())
)

classificacao = st.sidebar.radio(
    "Classificar barras por:",
    ["COMISSAO_DIFERIDA", "TIPO PRODUTO"]
)

# ===============================
# Aplicar filtros
# ===============================
df_f = df.copy()

if regional != "Todas":
    df_f = df_f[df_f["REGIONAIS"] == regional]

if coordenador != "Todos":
    df_f = df_f[df_f["COORDENADOR"] == coordenador]

if loja != "Todas":
    df_f = df_f[df_f["DESCRICAO_LOJA"] == loja]

if atendente != "Todos":
    df_f = df_f[df_f["NOME_ATENDENTE"] == atendente]

if tipo_produto != "Todos":
    df_f = df_f[df_f["TIPO PRODUTO"] == tipo_produto]

if mes != "Todos":
    df_f = df_f[df_f["DATA_EFETIVACAO"].dt.strftime("%Y-%m") == mes]

if df_f.empty:
    st.warning("Sem dados para os filtros selecionados.")
    st.stop()

# ===============================
# Calcular dia útil
# ===============================
df_f = adicionar_dia_util(df_f, "DATA_EFETIVACAO")

# ===============================
# Separar Consignado / Débito
# ===============================
df_consignado = df_f[df_f["GRUPO PRODUTO"] == "CONSIGNADO"]
df_debito = df_f[df_f["GRUPO PRODUTO"] == "DÉBITO"]


def agregar(df):
    return (
        df
        .groupby(["DIA_UTIL", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )


g_cons = agregar(df_consignado) if not df_consignado.empty else pd.DataFrame()
g_deb = agregar(df_debito) if not df_debito.empty else pd.DataFrame()

# ===============================
# Gráficos
# ===============================
fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

def plot_barras(ax, g, titulo):
    classes = g.columns.tolist()
    x = np.arange(len(g))
    width = 0.8 / max(len(classes), 1)

    offsets = np.linspace(
        -width * (len(classes) - 1) / 2,
        width * (len(classes) - 1) / 2,
        len(classes)
    )

    for i, classe in enumerate(classes):
        ax.bar(x + offsets[i], g[classe], width, label=str(classe))
        anotar_barras(ax, x, g[classe], offsets[i])

    ax.set_title(titulo)
    ax.legend()

if not g_cons.empty:
    plot_barras(axs[0], g_cons, "CONSIGNADO — VLRAF por Dia Útil")
else:
    axs[0].text(0.5, 0.5, "Sem dados Consignado",
                ha="center", va="center", transform=axs[0].transAxes)

if not g_deb.empty:
    plot_barras(axs[1], g_deb, "DÉBITO — VLRAF por Dia Útil")
else:
    axs[1].text(0.5, 0.5, "Sem dados Débito",
                ha="center", va="center", transform=axs[1].transAxes)

max_dia = max(
    g_cons.index.max() if not g_cons.empty else 0,
    g_deb.index.max() if not g_deb.empty else 0
)

axs[1].set_xlabel("Dia útil (D+)")
axs[1].set_xticks(range(max_dia))
axs[1].set_xticklabels([f"D+{i}" for i in range(1, max_dia + 1)])

plt.tight_layout()
st.pyplot(fig)
