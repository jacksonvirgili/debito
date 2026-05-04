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
# FUNÇÕES UTILITÁRIAS
# =================================================
def formatar_moeda(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def estilo_desvio_coluna(col):
    return [
        "color: red; font-weight: 600" if float(v.replace("R$", "").replace(".", "").replace(",", ".")) < 0
        else "color: blue; font-weight: 600"
        for v in col
    ]

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
# DIA ÚTIL
# =================================================
df = df[df["DATA_EFETIVACAO"].dt.weekday < 5]
df = df[~df["DATA_EFETIVACAO"].isin(FERIADOS_BR)]
df["MES"] = df["DATA_EFETIVACAO"].dt.strftime("%Y-%m")
df["DIA_UTIL"] = df.groupby("MES")["DATA_EFETIVACAO"].rank(method="dense").astype(int)

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

    mes = st.selectbox("Mês", sorted(df["MES"].unique()))
    base = df[df["MES"] == mes]

    g = base.groupby("DIA_UTIL")["VLRAF"].sum()

    fig = go.Figure()
    fig.add_bar(
        x=[f"D+{d}" for d in g.index],
        y=g.values,
        hovertemplate="%{x}<br>VLRAF: R$ %{y:,.0f}<extra></extra>"
    )

    fig = aplicar_estilo_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 2 — COMPARAÇÃO ENTRE MESES (COMPLETA)
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    col1, col2 = st.columns(2)

    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

    base = df[df["MES"].isin([mes_a, mes_b])]
    produtos = sorted(base["GRUPO PRODUTO"].dropna().unique())

    for prod in produtos:
        st.markdown(f"### {prod}")

        sub = base[base["GRUPO PRODUTO"] == prod]

        g = (
            sub
            .groupby(["MES", "DIA_UTIL"])["VLRAF"]
            .sum()
            .unstack()
            .fillna(0)
        )

        for m in [mes_a, mes_b]:
            if m not in g.columns:
                g[m] = 0

        g = g[[mes_a, mes_b]]
        g["DESVIO"] = g[mes_b] - g[mes_a]

        eixo_x = [f"D+{d}" for d in g.index]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08
        )

        fig.add_bar(row=1, col=1, x=eixo_x, y=g[mes_a], name=mes_a, marker_color="gray")
        fig.add_bar(row=1, col=1, x=eixo_x, y=g[mes_b], name=mes_b, marker_color="#EA9411")

        fig.add_bar(
            row=2, col=1,
            x=eixo_x,
            y=g["DESVIO"],
            marker_color=["blue" if v >= 0 else "red" for v in g["DESVIO"]],
            showlegend=False
        )

        fig.add_hline(y=0, row=2, col=1)
        fig.update_yaxes(title_text="VLRAF", row=1, col=1)
        fig.update_yaxes(title_text="Δ VLRAF", row=2, col=1)

        fig = aplicar_estilo_plotly(fig)
        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)

        tabela = g.reset_index()
        tabela["Dia Útil"] = tabela["DIA_UTIL"].apply(lambda x: f"D+{x}")

        for col in [mes_a, mes_b, "DESVIO"]:
            tabela[col] = tabela[col].apply(formatar_moeda)

        tabela_final = tabela[["Dia Útil", mes_a, mes_b, "DESVIO"]]

        st.dataframe(
            tabela_final.style.apply(
                estilo_desvio_coluna,
                subset=["DESVIO"]
            ),
            use_container_width=True,
            hide_index=True
        )
