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

# =================================================
# FORMATAÇÃO TABELAS
# =================================================
def formatar_tabela(df):
    df_fmt = df.copy()
    df_fmt.index = [f"D+{i}" for i in df_fmt.index]
    df_fmt.index.name = "DATA"

    for col in df_fmt.columns:
        df_fmt[col] = df_fmt[col].apply(
            lambda x: f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
    return df_fmt

def aplicar_cor(val):
    try:
        v = float(str(val).replace("R$", "").replace(".", "").replace(",", "."))
        return "color: red" if v < 0 else "color: blue"
    except:
        return ""

# =================================================
# MAPAS DE CORES
# =================================================
MAPA_CORES_PAGAMENTO = {"24": "#1f77b4","corte": "#EA9411"}
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
# TAB 1
# =================================================
with tab1:
    st.subheader("Análise diária por classificação")

    grupo_produto = st.selectbox("Grupo Produto", ["Todos"] + sorted(df["GRUPO PRODUTO"].unique()))
    classificacao_label = st.radio("Classificar por:", ["TIPO PAGAMENTO", "TIPO PRODUTO"])

    MAPA_CLASSIFICACAO = {
        "TIPO PAGAMENTO": "COMISSAO_DIFERIDA",
        "TIPO PRODUTO": "TIPO PRODUTO"
    }

    classificacao = MAPA_CLASSIFICACAO[classificacao_label]
    mes = st.selectbox("Mês", sorted(df["MES"].unique()))

    base = df[df["MES"] == mes]
    if grupo_produto != "Todos":
        base = base[base["GRUPO PRODUTO"] == grupo_produto]

    g = base.groupby(["DIA_UTIL", classificacao])["VLRAF"].sum().unstack(fill_value=0)
    g_acum = g.cumsum()

    if classificacao == "COMISSAO_DIFERIDA":
        ordem = [c for c in ["24", "corte"] if c in g.columns]
        g = g[ordem]
        g_acum = g_acum[ordem]

    mapa_cores = MAPA_CORES_PAGAMENTO if classificacao == "COMISSAO_DIFERIDA" else MAPA_CORES_PRODUTO
    eixo_x = [f"D+{d}" for d in g.index]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)

    for col in g.columns:
        fig.add_bar(row=1, col=1, x=eixo_x, y=g[col], name=str(col),
                    marker_color=mapa_cores.get(col, "gray"))

    for col in g_acum.columns:
        fig.add_bar(row=2, col=1, x=eixo_x, y=g_acum[col],
                    showlegend=False,
                    marker_color=mapa_cores.get(col, "gray"))

    fig = aplicar_estilo_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    # TABELAS
    st.dataframe(formatar_tabela(g).style.applymap(aplicar_cor))
    st.download_button("📥 Download Diário", g.to_csv().encode("utf-8"), f"diario_{mes}.csv")

    st.dataframe(formatar_tabela(g_acum).style.applymap(aplicar_cor))
    st.download_button("📥 Download Acumulado", g_acum.to_csv().encode("utf-8"), f"acumulado_{mes}.csv")

# =================================================
# TAB 2
# =================================================
with tab2:
    st.subheader("Comparação entre meses")

    meses = sorted(df["MES"].unique())
    mes_a = st.selectbox("Mês A", meses)
    mes_b = st.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

    base = df[df["MES"].isin([mes_a, mes_b])]
    produtos = sorted(base["GRUPO PRODUTO"].unique())

    for prod in produtos:
        st.markdown(f"### {prod}")

        sub = base[base["GRUPO PRODUTO"] == prod]
        g = sub.groupby(["MES", "DIA_UTIL"])["VLRAF"].sum().unstack("MES", fill_value=0)

        eixo_x = [f"D+{d}" for d in g.index]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        for mes in [mes_a, mes_b]:
            fig.add_bar(row=1, col=1, x=eixo_x, y=g[mes], name=mes)

        g_desvio = g.loc[(g[mes_a] != 0) & (g[mes_b] != 0)].copy()
        g_desvio["DESVIO"] = g_desvio[mes_b] - g_desvio[mes_a]

        fig.add_bar(row=2, col=1,
                    x=[f"D+{d}" for d in g_desvio.index],
                    y=g_desvio["DESVIO"])

        fig = aplicar_estilo_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

        # TABELAS
        st.dataframe(formatar_tabela(g).style.applymap(aplicar_cor))
        st.download_button(f"📥 Download {prod}", g.to_csv().encode("utf-8"))

        if not g_desvio.empty:
            st.dataframe(formatar_tabela(g_desvio[["DESVIO"]]).style.applymap(aplicar_cor))
            st.download_button(f"📥 Download Desvio {prod}", g_desvio.to_csv().encode("utf-8"))
