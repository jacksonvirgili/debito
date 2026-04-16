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
    
MAPA_CORES = {
    "NOVO": "#1f77b4",          # azul
    "corte": "#1f77b4",
    "REFIN": "#EA9411",         # laranja
    "24": "#EA9411",
    "PORT": "#6f42c1",          # roxo
    "REFIN DA PORT": "#2ca02c"  # verde
}

df = adicionar_dia_util(df)

# =================================================
# FILTROS
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

# =================================================
# TABS
# =================================================
tab1, tab2 = st.tabs(["Análise Diária", "Comparação entre Meses"])

# =================================================
# TAB 1 — PLOTLY (ESTILO DO TAB 2)
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

    mes = st.selectbox("Mês", sorted(df["MES"].unique()))

    base = df[df["MES"] == mes]
    if grupo_produto != "Todos":
        base = base[base["GRUPO PRODUTO"] == grupo_produto]

    g = (
        base.groupby(["DIA_UTIL", classificacao])["VLRAF"]
        .sum()
        .unstack(fill_value=0)
    )

    g_acum = g.cumsum()
    eixo_x = [f"D+{d}" for d in g.index]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("VLRAF Diário", "VLRAF Acumulado")
    )

    for col in g.columns:
        fig.add_bar(
            row=1, col=1,
            x=eixo_x,
            y=g[col],
            name=str(col),
            showlegend=False,
            marker_color=MAPA_CORES.get(col, "gray"),
            hovertemplate="<b>%{x}</b><br>" + str(col) + ": R$ %{y:,.0f}<extra></extra>"
        )

    for col in g_acum.columns:
        fig.add_bar(
            row=2, col=1,
            x=eixo_x,
            y=g_acum[col],
            showlegend=False,
            marker_color=MAPA_CORES.get(col, "gray"),
            hovertemplate="<b>%{x}</b><br>" + str(col) + ": R$ %{y:,.0f}<extra></extra>"
        )

    fig.update_yaxes(title_text="VLRAF", row=1, col=1)
    fig.update_yaxes(title_text="VLRAF Acumulado", row=2, col=1)
    fig.update_xaxes(title_text="Dia Útil", row=2, col=1)

    fig = aplicar_estilo_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# TAB 2 — COMPARAÇÃO ENTRE MESES
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    col1, col2 = st.columns(2)
    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

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

        eixo_x = [f"D+{d}" for d in g.index]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )

        cores = {mes_a: "gray", mes_b: "#EA9411"}

        for mes in [mes_a, mes_b]:
            fig.add_bar(
                row=1, col=1,
                x=eixo_x,
                y=g[mes],
                name=mes,
                marker_color=cores[mes],
                hovertemplate="<b>%{x}</b><br>" + mes + ": R$ %{y:,.0f}<extra></extra>"
            )

        g_desvio = g[(g[mes_a] != 0) & (g[mes_b] != 0)]
        g_desvio["DESVIO"] = g_desvio[mes_b] - g_desvio[mes_a]

        fig.add_bar(
            row=2, col=1,
            x=[f"D+{d}" for d in g_desvio.index],
            y=g_desvio["DESVIO"],
            marker_color=["#EA9411" if v >= 0 else "gray" for v in g_desvio["DESVIO"]],
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Desvio: R$ %{y:,.0f}<extra></extra>"
        )

        fig.add_hline(y=0, row=2, col=1)

        fig.update_yaxes(title_text="VLRAF", row=1, col=1)
        fig.update_yaxes(title_text="Δ VLRAF", row=2, col=1)
        fig.update_xaxes(title_text="Dia Útil", row=2, col=1)

        fig = aplicar_estilo_plotly(fig)
        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)
