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
# LOAD
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
    df = df[
        (df["DATA_EFETIVACAO"].dt.weekday < 5) &
        (~df["DATA_EFETIVACAO"].isin(FERIADOS_BR))
    ]
    df = df.copy()
    df["MES"] = df["DATA_EFETIVACAO"].dt.strftime("%Y-%m")
    df["DIA_UTIL"] = df.groupby("MES")["DATA_EFETIVACAO"].rank("dense").astype(int)
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
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.15)", zeroline=False)
    return fig

def formatar_moeda(v):
    if pd.isna(v):
        return ""
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def aplicar_cor_desvio(df):
    estilos = pd.DataFrame("", index=df.index, columns=df.columns)
    for i, v in df["Δ VLRAF"].items():
        n = float(v.replace("R$", "").replace(".", "").replace(",", "."))
        estilos.loc[i, "Δ VLRAF"] = (
            "color:#1f77b4;font-weight:bold"
            if n >= 0 else
            "color:red;font-weight:bold"
        )
    return estilos

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

    g = (
        base.groupby(["DIA_UTIL", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    g_acum = g.cumsum()

    if classificacao == "COMISSAO_DIFERIDA":
        ordem = [c for c in ["24", "corte"] if c in g.columns]
        g, g_acum = g[ordem], g_acum[ordem]
        mapa_cores = MAPA_CORES_PAGAMENTO
    else:
        mapa_cores = MAPA_CORES_PRODUTO

    eixo_x = [f"D+{d}" for d in g.index]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("VLRAF Diário", "VLRAF Acumulado"))

    for col in g.columns:
        fig.add_bar(row=1, col=1, x=eixo_x, y=g[col],
                    name=col, marker_color=mapa_cores.get(col, "gray"))

    for col in g_acum.columns:
        fig.add_bar(row=2, col=1, x=eixo_x, y=g_acum[col],
                    showlegend=False, marker_color=mapa_cores.get(col, "gray"))

    fig = aplicar_estilo_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 2 — COMPARAÇÃO ENTRE MESES
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    col1, col2 = st.columns(2)
    mes_a = col1.selectbox("Mês A", meses, key="mes_a")
    mes_b = col2.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0, key="mes_b")

    base = df[df["MES"].isin([mes_a, mes_b])]

    for prod in sorted(base["GRUPO PRODUTO"].unique()):
        st.markdown(f"### {prod}")

        sub = base[base["GRUPO PRODUTO"] == prod]

        g = (
            sub.groupby(["MES", "DIA_UTIL"])["VLRAF"]
            .sum()
            .unstack()
            .reindex(columns=[mes_a, mes_b], fill_value=0)
            .sort_index()
        )

        eixo_x = [f"D+{d}" for d in g.index]
        desvio = g[mes_b] - g[mes_a]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

        fig.add_bar(row=1, col=1, x=eixo_x, y=g[mes_a], name=mes_a, marker_color="gray")
        fig.add_bar(row=1, col=1, x=eixo_x, y=g[mes_b], name=mes_b, marker_color="#EA9411")

        fig.add_bar(
            row=2, col=1,
            x=eixo_x, y=desvio,
            marker_color=["#EA9411" if v >= 0 else "gray" for v in desvio],
            showlegend=False
        )

        fig.add_hline(y=0, row=2, col=1)
        fig = aplicar_estilo_plotly(fig)
        fig.update_layout(height=550)

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"grafico_{prod}_{mes_a}_{mes_b}"
        )
