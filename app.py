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

    # índice como D+
    df_fmt.index = [f"D+{i}" for i in df_fmt.index]
    df_fmt.index.name = "DATA"

    # formatação moeda
    for col in df_fmt.columns:
        df_fmt[col] = df_fmt[col].apply(
            lambda x: f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )

    return df_fmt

# =================================================
# EXCEL FORMATADO
# =================================================
def gerar_excel(dfs_dict):
    output = BytesIO()

    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for nome, df in dfs_dict.items():
                df.to_excel(writer, sheet_name=nome)

                workbook  = writer.book
                worksheet = writer.sheets[nome]

                fmt_pos = workbook.add_format({'font_color': 'blue', 'num_format': 'R$ #,##0'})
                fmt_neg = workbook.add_format({'font_color': 'red', 'num_format': 'R$ #,##0'})

                for col_idx in range(1, len(df.columns)+1):
                    worksheet.set_column(col_idx, col_idx, 18)

                    worksheet.conditional_format(1, col_idx, len(df), col_idx, {
                        'type': 'cell',
                        'criteria': '>=',
                        'value': 0,
                        'format': fmt_pos
                    })

                    worksheet.conditional_format(1, col_idx, len(df), col_idx, {
                        'type': 'cell',
                        'criteria': '<',
                        'value': 0,
                        'format': fmt_neg
                    })

    except ModuleNotFoundError:
        # fallback simples (sem formatação)
        with pd.ExcelWriter(output) as writer:
            for nome, df in dfs_dict.items():
                df.to_excel(writer, sheet_name=nome)

    return output.getvalue()

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
    st.dataframe(formatar_tabela(g))
    st.dataframe(formatar_tabela(g_acum))

    excel = gerar_excel({
        "Diario": g,
        "Acumulado": g_acum
    })

    st.download_button("📥 Baixar Excel", data=excel, file_name=f"vlraf_{mes}.xlsx")

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
        st.dataframe(formatar_tabela(g))

        if not g_desvio.empty:
            st.dataframe(formatar_tabela(g_desvio[["DESVIO"]]))

        excel = gerar_excel({
            "Comparacao": g,
            "Desvio": g_desvio if not g_desvio.empty else pd.DataFrame()
        })

        st.download_button(
            f"📥 Baixar Excel - {prod}",
            data=excel,
            file_name=f"{prod}_{mes_a}_{mes_b}.xlsx"
        )
