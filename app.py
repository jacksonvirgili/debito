import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =================================================
# CONFIGURAÇÃO
# =================================================
st.set_page_config(page_title="Acompanhamento VLRAF", layout="wide")

# =================================================
# FERIADOS
# =================================================
FERIADOS_BR = pd.to_datetime([
    "2026-01-01",
    "2026-02-16","2026-02-17",
    "2026-04-03",
    "2026-04-21",
    "2026-05-01",
    "2026-06-04",
    "2026-09-07",
    "2026-10-12",
    "2026-11-02",
    "2026-11-15",
    "2026-12-25",
])

# =================================================
# DADOS
# =================================================
@st.cache_data
def carregar_dados():
    df = pd.read_parquet("Acomp.parquet")
    df["DATA_EFETIVACAO"] = pd.to_datetime(df["DATA_EFETIVACAO"])
    return df

df = carregar_dados()

# =================================================
# FUNÇÕES
# =================================================
def adicionar_dia_util(df):
    df = df[df["DATA_EFETIVACAO"].dt.weekday < 5]
    df = df[~df["DATA_EFETIVACAO"].isin(FERIADOS_BR)]
    df["MES"] = df["DATA_EFETIVACAO"].dt.strftime("%Y-%m")
    df["DIA_UTIL"] = df.groupby("MES")["DATA_EFETIVACAO"].rank("dense").astype(int)
    return df

def moeda(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def cor_desvio(v):
    v = float(v.replace("R$", "").replace(".", "").replace(",", "."))
    if v > 0:
        return "color: blue"
    if v < 0:
        return "color: red"
    return ""

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
    grupo = st.selectbox("Grupo Produto", ["Todos"] + sorted(df["GRUPO PRODUTO"].unique()))
    classificacao = st.radio("Classificar por:", ["COMISSAO_DIFERIDA", "TIPO PRODUTO"])
    mes = st.selectbox("Mês", sorted(df["MES"].unique()))

    base = df[df["MES"] == mes]
    if grupo != "Todos":
        base = base[base["GRUPO PRODUTO"] == grupo]

    g = base.groupby(["DIA_UTIL", classificacao])["VLRAF"].sum().unstack(fill_value=0)

    eje_x = [f"D+{d}" for d in g.index]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for c in g.columns:
        fig.add_bar(row=1,col=1,x=eje_x,y=g[c],name=str(c))
        fig.add_bar(row=2,col=1,x=eje_x,y=g[c].cumsum(),showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # ---------- TABELA ----------
    st.subheader("Tabela de Apoio (Download)")
    tabela = base.groupby(["DATA_EFETIVACAO","DIA_UTIL",classificacao])["VLRAF"].sum().reset_index()
    tabela["Dia Útil"] = tabela["DIA_UTIL"].apply(lambda x: f"D+{x}")
    tabela["Data"] = tabela["DATA_EFETIVACAO"].dt.strftime("%d/%m/%Y")
    tabela["VLRAF"] = tabela["VLRAF"].apply(moeda)
    tabela = tabela[["Dia Útil","Data",classificacao,"VLRAF"]]

    st.dataframe(tabela, use_container_width=True)

    tabela.to_excel("analise_diaria.xlsx", index=False)
    st.download_button("⬇️ Baixar Excel", open("analise_diaria.xlsx","rb"), "analise_diaria.xlsx")

# =================================================
# TAB 2 — COMPARAÇÃO
# =================================================
with tab2:
    meses = sorted(df["MES"].unique())
    mes_a, mes_b = st.columns(2)
    mes_a = mes_a.selectbox("Mês A", meses)
    mes_b = mes_b.selectbox("Mês B", meses, index=1)

    base = df[df["MES"].isin([mes_a, mes_b])]

    for prod in base["GRUPO PRODUTO"].unique():
        st.markdown(f"### {prod}")
        sub = base[base["GRUPO PRODUTO"] == prod]

        g = sub.groupby(["DIA_UTIL","MES"])["VLRAF"].sum().unstack(fill_value=0)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_bar(row=1,col=1,x=g.index,y=g[mes_a],name=mes_a)
        fig.add_bar(row=1,col=1,x=g.index,y=g[mes_b],name=mes_b)

        desvio = g[mes_b] - g[mes_a]
        fig.add_bar(row=2,col=1,x=g.index,y=desvio)

        st.plotly_chart(fig, use_container_width=True)

        # ---------- TABELA ----------
        tabela = pd.DataFrame({
            "Dia Útil": [f"D+{i}" for i in g.index],
            mes_a: g[mes_a].apply(moeda),
            mes_b: g[mes_b].apply(moeda),
            "Desvio": desvio.apply(moeda)
        })

        st.dataframe(
            tabela.style.applymap(cor_desvio, subset=["Desvio"]),
            use_container_width=True
        )

        nome = f"comparativo_{prod}.xlsx"
        tabela.to_excel(nome, index=False)
        st.download_button(f"⬇️ Baixar {prod}", open(nome,"rb"), nome)
