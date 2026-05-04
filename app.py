import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

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

def formatar_tabela(df):
    df_fmt = df.copy()

    df_fmt.index = [f"D+{i}" for i in df_fmt.index]
    df_fmt.index.name = "DIA_UTIL"

    for col in df_fmt.columns:
        if "DATA" not in str(col):
            df_fmt[col] = df_fmt[col].apply(
                lambda x: f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )

    return df_fmt

# =================================================
# EXCEL
# =================================================
def gerar_excel(dfs_dict):
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        for nome, df in dfs_dict.items():
            df.to_excel(writer, sheet_name=nome)
    return output.getvalue()

# =================================================
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

    # MAPA DATA
    mapa_data = (
        base.groupby("DIA_UTIL")["DATA_EFETIVACAO"]
        .min()
    )

    g["DATA"] = g.index.map(mapa_data)
    g_acum["DATA"] = g_acum.index.map(mapa_data)

    if classificacao == "COMISSAO_DIFERIDA":
        ordem = [c for c in ["24", "corte"] if c in g.columns]
        g = g[ordem + ["DATA"]]
        g_acum = g_acum[ordem + ["DATA"]]

    eixo_x = [f"D+{d}" for d in g.index]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    for col in g.columns:
        if col != "DATA":
            fig.add_bar(row=1, col=1, x=eixo_x, y=g[col], name=str(col))

    for col in g_acum.columns:
        if col != "DATA":
            fig.add_bar(row=2, col=1, x=eixo_x, y=g_acum[col], showlegend=False)

    fig = aplicar_estilo_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    tabela_unificada = pd.concat({"Diário": g, "Acumulado": g_acum}, axis=1)
    st.dataframe(formatar_tabela(tabela_unificada))

# =================================================
# TAB 2
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

        # ===============================
        # BASE DIÁRIA
        # ===============================
        g = (
            sub.groupby(["MES", "DIA_UTIL"])["VLRAF"]
            .sum()
            .unstack("MES", fill_value=0)
        )

        # ===============================
        # MAPA DATA POR MES
        # ===============================
        mapa_data = (
            sub.groupby(["MES", "DIA_UTIL"])["DATA_EFETIVACAO"]
            .min()
            .unstack("MES")
        )

        # ===============================
        # TRANSFORMAR EM DATA + VALOR
        # ===============================
        for mes in [mes_a, mes_b]:
            g[(mes, "VALOR")] = g[mes]
            g[(mes, "DATA")] = mapa_data[mes]
            g.drop(columns=mes, inplace=True)

        # ===============================
        # ORGANIZAR COLUNAS (ROBUSTO)
        # ===============================
        g.columns = pd.MultiIndex.from_tuples(g.columns)
        g.columns.names = ["MÊS", "TIPO"]

        cols_ordenadas = []
        for mes in [mes_a, mes_b]:
            for tipo in ["DATA", "VALOR"]:
                if (mes, tipo) in g.columns:
                    cols_ordenadas.append((mes, tipo))

        cols_restantes = [c for c in g.columns if c not in cols_ordenadas]
        g = g[cols_ordenadas + cols_restantes]

        # ===============================
        # ACUMULADO
        # ===============================
        g_acum = g.copy()

        for mes in [mes_a, mes_b]:
            if (mes, "VALOR") in g_acum.columns:
                g_acum[(mes, "VALOR")] = g[(mes, "VALOR")].cumsum()

        g_acum = g_acum[g.columns]

        # ===============================
        # DESVIO
        # ===============================
        g_desvio = pd.DataFrame(index=g.index)

        if (mes_a, "VALOR") in g.columns and (mes_b, "VALOR") in g.columns:
            g_desvio["DESVIO"] = g[(mes_b, "VALOR")] - g[(mes_a, "VALOR")]
        else:
            g_desvio["DESVIO"] = 0

        # ===============================
        # DESVIO ACUMULADO
        # ===============================
        g_desvio_acum = g_desvio.copy()
        g_desvio_acum["DESVIO_ACUM"] = g_desvio["DESVIO"].cumsum()

        # ===============================
        # GRÁFICO
        # ===============================
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )

        eixo_x = [f"D+{d}" for d in g.index]
        cores = {mes_a: "gray", mes_b: "#EA9411"}

        # barras principais
        if (mes_a, "VALOR") in g.columns:
            fig.add_bar(
                row=1, col=1,
                x=eixo_x,
                y=g[(mes_a, "VALOR")],
                name=mes_a,
                marker_color=cores[mes_a]
            )

        if (mes_b, "VALOR") in g.columns:
            fig.add_bar(
                row=1, col=1,
                x=eixo_x,
                y=g[(mes_b, "VALOR")],
                name=mes_b,
                marker_color=cores[mes_b]
            )

        # desvio
        fig.add_bar(
            row=2, col=1,
            x=eixo_x,
            y=g_desvio["DESVIO"],
            marker_color=["#EA9411" if v >= 0 else "gray" for v in g_desvio["DESVIO"]],
            showlegend=False
        )

        fig.add_hline(y=0, row=2, col=1)

        fig = aplicar_estilo_plotly(fig)
        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # TABELA UNIFICADA
        # ===============================
        tabela_unificada = pd.concat(
            {
                "Diário": g,
                "Acumulado": g_acum,
                "Desvio": g_desvio,
                "Desvio Acumulado": g_desvio_acum
            },
            axis=1
        )

        st.dataframe(formatar_tabela(tabela_unificada))

        # ===============================
        # EXCEL
        # ===============================
        excel = gerar_excel({
            "Resumo": tabela_unificada
        })

        st.download_button(
            f"📥 Baixar Excel - {prod}",
            data=excel,
            file_name=f"{prod}_{mes_a}_{mes_b}.xlsx",
            key=f"download_{prod}_{mes_a}_{mes_b}"
        )
