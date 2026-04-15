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

    # Remover colunas lixo do Excel
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalizar tipo de produto
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
            rotation=90,
        )


# ===============================
# Sidebar — filtros
# ===============================
st.sidebar.title("Filtros")

# 🔹 Mapeamento CORRETO (label -> nome da coluna)
mapa_classificacao = {
    "Comissão Diferida": "COMISSAO_DIFERIDA",
    "Tipo Produto": "TIPO PRODUTO",
}

label_classificacao = st.sidebar.radio(
    "Classificar por:",
    options=list(mapa_classificacao.keys()),
)

classificacao = mapa_classificacao[label_classificacao]


regional = st.sidebar.selectbox(
    "Regional",
    ["Todas"] + sorted(df["REGIONAIS"].dropna().unique()),
)

coordenador = st.sidebar.selectbox(
    "Coordenador",
    ["Todos"] + sorted(df["COORDENADOR"].dropna().unique()),
)

loja = st.sidebar.selectbox(
    "Loja",
    ["Todas"] + sorted(df["DESCRICAO_LOJA"].dropna().unique()),
)

atendente = st.sidebar.selectbox(
    "Atendente",
    ["Todos"] + sorted(df["NOME_ATENDENTE"].dropna().unique()),
)

tipo_produto = st.sidebar.selectbox(
    "Tipo Produto",
    ["Todos"] + sorted(df["TIPO PRODUTO"].dropna().unique()),
)

grupo_produto = st.sidebar.selectbox(
    "Grupo Produto",
    ["Todos"] + sorted(df["GRUPO PRODUTO"].dropna().unique()),
)

mes = st.sidebar.selectbox(
    "Mês",
    ["Todos"]
    + sorted(df["DATA_EFETIVACAO"].dt.strftime("%Y-%m").unique()),
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

if tipo_produto != "Todos" and classificacao != "TIPO PRODUTO":
    df_f = df_f[df_f["TIPO PRODUTO"] == tipo_produto]

if grupo_produto != "Todos" and classificacao != "GRUPO PRODUTO":
    df_f = df_f[df_f["GRUPO PRODUTO"] == grupo_produto]

if mes != "Todos":
    df_f = df_f[
        df_f["DATA_EFETIVACAO"].dt.strftime("%Y-%m") == mes
    ]

if df_f.empty:
    st.warning("Sem dados para os filtros selecionados.")
    st.stop()


# ===============================
# Validação de segurança
# ===============================
colunas_necessarias = ["DATA_EFETIVACAO", classificacao, "VLRAF"]
faltando = [c for c in colunas_necessarias if c not in df_f.columns]

if faltando:
    st.error(
        f"Erro interno.\n\nColunas ausentes: {faltando}\n\n"
        f"Colunas disponíveis:\n{df_f.columns.tolist()}"
    )
    st.stop()


# ===============================
# Agregação
# ===============================
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

offsets = np.linspace(
    -width * (len(classes) - 1) / 2,
    width * (len(classes) - 1) / 2,
    len(classes),
)


# ===============================
# Gráficos
# ===============================
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Diário
for i, classe in enumerate(classes):
    axs[0].bar(x + offsets[i], g[classe], width, label=str(classe))
    anotar_barras(axs[0], x, g[classe], offsets[i])

axs[0].set_title(f"VLRAF diário — {label_classificacao}")
axs[0].legend()

# Acumulado
for i, classe in enumerate(classes):
    axs[1].bar(x + offsets[i], g_acum[classe], width, label=str(classe))
    anotar_barras(axs[1], x, g_acum[classe], offsets[i])

axs[1].set_title(f"VLRAF acumulado — {label_classificacao}")
axs[1].set_xticks(x)
axs[1].set_xticklabels(g.index.day, rotation=45)
axs[1].legend()

plt.tight_layout()
st.pyplot(fig)
