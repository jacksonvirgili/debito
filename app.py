import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tseries.holiday import BrazilHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

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

    # Remover colunas lixo do Excel
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalizar tipo produto
    df.loc[df["TIPO PRODUTO"] != "NOVO", "TIPO PRODUTO"] = "REFIN"

    # Garantir datetime
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

        if max_val > 0 and v < max_val * 0.12:
            y = v
            va = "bottom"
            color = "black"
        else:
            y = v * 0.5
            va = "center"
            color = "white"

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
    cal = BrazilHolidayCalendar()
    cbd = CustomBusinessDay(calendar=cal)

    df = df.copy()
    df["MES"] = df[coluna_data].dt.to_period("M")

    def calcular_dia_util(serie):
        dias_uteis = pd.date_range(
            start=serie.min(),
            end=serie.max(),
            freq=cbd
        )
        mapa = {data: i + 1 for i, data in enumerate(dias_uteis)}
        return serie.map(mapa)

    df["DIA_UTIL"] = (
        df
        .groupby("MES")[coluna_data]
        .transform(calcular_dia_util)
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
# Dia útil (sem feriados)
# ===============================
df_f = adicionar_dia_util(df_f, "DATA_EFETIVACAO")

# ===============================
# Separar Consignado e Débito
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
    plot_barras(
        axs[0],
        g_cons,
        "CONSIGNADO — VLRAF por Dia Útil"
    )
else:
    axs[0].text(
        0.5, 0.5,
        "Sem dados de Consignado",
        ha="center", va="center",
        transform=axs[0].transAxes
    )


if not g_deb.empty:
    plot_barras(
        axs[1],
        g_deb,
        "DÉBITO — VLRAF por Dia Útil"
    )
else:
    axs[1].text(
        0.5, 0.5,
        "Sem dados de Débito",
        ha="center", va="center",
        transform=axs[1].transAxes
    )

max_dia = max(
    g_cons.index.max() if not g_cons.empty else 0,
    g_deb.index.max() if not g_deb.empty else 0,
)

axs[1].set_xlabel("Dia útil (D+)")
axs[1].set_xticks(range(max_dia))
axs[1].set_xticklabels([f"D+{i}" for i in range(1, max_dia + 1)])

plt.tight_layout()
st.pyplot(fig)
