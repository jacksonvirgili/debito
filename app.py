import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# CARREGAMENTO DOS DADOS
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

def aplicar_estilo_plotly(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="group",
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
    )
    fig.update_xaxes(showline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False
    )
    return fig

def formatar_moeda(v):
    if pd.isna(v):
        return ""
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def mapa_dia_util_para_data(df, mes):
    return (
        df[df["MES"] == mes]
        .drop_duplicates("DIA_UTIL")
        .set_index("DIA_UTIL")["DATA_EFETIVACAO"]
        .dt.strftime("%d/%m/%Y")
    )

def cor_desvio(valor):
    v = float(valor.replace("R$", "").replace(".", "").replace(",", "."))
    if v < 0:
        return "color: red; font-weight: bold"
    return "color: #1f77b4; font-weight: bold"

# =================================================
# MAPAS DE CORES
# =================================================
MAPA_CORES_PAGAMENTO = {"24": "#1f77b4", "corte": "#EA9411"}
MAPA_CORES_PRODUTO = {
    "NOVO": "#1f77b4",
    "REFIN": "#EA9411",
    "PORTABILIDADE": "#6f42c1",
    "REFIN DA PORT": "#2ca02c"
}

df = adicionar_dia_util(df)

# =================================================
# FILTROS
# =================================================
st.sidebar.title("Filtros")

regional = st.sidebar.selectbox("Regional", ["Todas"] + sorted(df["REGIONAIS"].dropna().unique()))
coordenador = st.sidebar.selectbox("Coordenador", ["Todos"] + sorted(df["COORDENADOR"].dropna().unique()))
loja = st.sidebar.selectbox("Loja", ["Todas"] + sorted(df["DESCRICAO_LOJA"].dropna().unique()))

if regional != "Todas":
    df = df[df["REGIONAIS"] == regional]
if coordenador != "Todos":
    df = df[df["COORDENADOR"] == coordenador]
if loja != "Todas":
    df = df[df["DESCRICAO_LOJA"] == loja]

# =================================================
# TABS
# =================================================
tab1, tab2 = st.tabs(["Análise Diária", "Comparação entre Meses"])

# =================================================
# TAB 1 — ANÁLISE DIÁRIA
# =================================================
with tab1:
    st.subheader("Análise diária por classificação")

    grupo_produto = st.selectbox("Grupo Produto", ["Todos"] + sorted(df["GRUPO PRODUTO"].unique()))
    classificacao_label = st.radio("Classificar por:", ["TIPO PAGAMENTO", "TIPO PRODUTO"])
    classificacao = {"TIPO PAGAMENTO": "COMISSAO_DIFERIDA", "TIPO PRODUTO": "TIPO PRODUTO"}[classificacao_label]
    mes = st.selectbox("Mês", sorted(df["MES"].unique()))

    base = df[df["MES"] == mes]
    if grupo_produto != "Todos":
        base = base[base["GRUPO PRODUTO"] == grupo_produto]

    g = base.groupby(["DIA_UTIL", classificacao])["VLRAF"].sum().unstack(fill_value=0)
    g_acum = g.cumsum()

    mapa_datas = mapa_dia_util_para_data(df, mes)

    tabela_diaria = g.reset_index()
    tabela_diaria["Dia"] = tabela_diaria["DIA_UTIL"].map(mapa_datas)
    for c in tabela_diaria.columns:
        if c not in ["DIA_UTIL", "Dia"]:
            tabela_diaria[c] = tabela_diaria[c].apply(formatar_moeda)
    tabela_diaria = tabela_diaria.drop(columns="DIA_UTIL")

    tabela_acum = g_acum.reset_index()
    tabela_acum["Dia"] = tabela_acum["DIA_UTIL"].map(mapa_datas)
    for c in tabela_acum.columns:
        if c not in ["DIA_UTIL", "Dia"]:
            tabela_acum[c] = tabela_acum[c].apply(formatar_moeda)
    tabela_acum = tabela_acum.drop(columns="DIA_UTIL")

    st.markdown("### 📋 VLRAF Diário")
    st.dataframe(tabela_diaria, use_container_width=True)
    st.download_button("⬇️ CSV Diário", tabela_diaria.to_csv(index=False, sep=";", decimal=","), f"vlraf_diario_{mes}.csv")

    st.markdown("### 📋 VLRAF Acumulado")
    st.dataframe(tabela_acum, use_container_width=True)
    st.download_button("⬇️ CSV Acumulado", tabela_acum.to_csv(index=False, sep=";", decimal=","), f"vlraf_acumulado_{mes}.csv")

# =================================================
# TAB 2 — COMPARAÇÃO
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    mes_a = st.selectbox("Mês A", meses)
    mes_b = st.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

    base = df[df["MES"].isin([mes_a, mes_b])]

    for prod in sorted(base["GRUPO PRODUTO"].unique()):
        st.markdown(f"### {prod}")
        sub = base[base["GRUPO PRODUTO"] == prod]

        g = sub.groupby(["MES", "DIA_UTIL"])["VLRAF"].sum().unstack("MES", fill_value=0)

        tabela = g.reset_index()
        tabela["Dia"] = tabela["DIA_UTIL"].apply(
            lambda x: sub[sub["DIA_UTIL"] == x]["DATA_EFETIVACAO"].iloc[0].strftime("%d/%m/%Y")
        )
        tabela["Δ VLRAF"] = tabela[mes_b] - tabela[mes_a]

        tabela = tabela.rename(columns={mes_a: f"VLRAF {mes_a}", mes_b: f"VLRAF {mes_b}"})
        for c in [f"VLRAF {mes_a}", f"VLRAF {mes_b}"]:
            tabela[c] = tabela[c].apply(formatar_moeda)
        tabela["Δ VLRAF"] = tabela["Δ VLRAF"].apply(formatar_moeda)

        tabela = tabela[["Dia", f"VLRAF {mes_a}", f"VLRAF {mes_b}", "Δ VLRAF"]]

        st.dataframe(
            tabela.style.applymap(cor_desvio, subset=["Δ VLRAF"]),
            use_container_width=True
        )

        st.download_button(
            f"⬇️ CSV ({prod})",
            tabela.to_csv(index=False, sep=";", decimal=","),
            f"comparacao_{prod}_{mes_a}_vs_{mes_b}.csv"
        )

