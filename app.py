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

def estilo_desvio(v):
    cor = "red" if v < 0 else "blue"
    return f"color: {cor}; font-weight: 600"

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

    g = base.groupby(["DIA_UTIL"])["VLRAF"].sum()

    fig = go.Figure()
    fig.add_bar(
        x=[f"D+{d}" for d in g.index],
        y=g.values,
        hovertemplate="%{x}<br>VLRAF: R$ %{y:,.0f}<extra></extra>"
    )

    fig = aplicar_estilo_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ================= TABELA =================
    tabela = base.groupby(
        ["DIA_UTIL", "DATA_EFETIVACAO"]
    )["VLRAF"].sum().reset_index()

    tabela["Dia Útil"] = tabela["DIA_UTIL"].apply(lambda x: f"D+{x}")
    tabela["Data"] = tabela["DATA_EFETIVACAO"].dt.strftime("%d/%m/%Y")
    tabela["VLRAF"] = tabela["VLRAF"].apply(formatar_moeda)

    tabela_final = tabela[["Dia Útil", "Data", "VLRAF"]]

    st.markdown("#### 📋 Tabela – Detalhe Diário")
    st.dataframe(tabela_final, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download tabela diária",
        tabela_final.to_csv(index=False, sep=";", encoding="utf-8-sig"),
        file_name=f"vlraf_diario_{mes}.csv",
        mime="text/csv"
    )

# =================================================
# TAB 2 — COMPARAÇÃO ENTRE MESES
# =================================================
with tab2:
    st.subheader("Comparação entre meses com desvio diário")

    meses = sorted(df["MES"].unique())
    col1, col2 = st.columns(2)

    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox(
        "Mês B",
        meses,
        index=1 if len(meses) > 1 else 0
    )

    base = df[df["MES"].isin([mes_a, mes_b])]
    produtos = sorted(base["GRUPO PRODUTO"].unique())

    for prod in produtos:
        st.markdown(f"### {prod}")

        sub = base[base["GRUPO PRODUTO"] == prod]

        # =================================================
        # AGREGAÇÃO ROBUSTA (dias úteis diferentes OK)
        # =================================================
        g = (
            sub
            .groupby(["MES", "DIA_UTIL"])["VLRAF"]
            .sum()
            .unstack()
        )

        # valores ausentes viram zero
        g = g.fillna(0)

        # garante colunas dos dois meses
        for m in [mes_a, mes_b]:
            if m not in g.columns:
                g[m] = 0

        # ordem explícita
        g = g[[mes_a, mes_b]]

        # desvio (zero vale zero)
        g["DESVIO"] = g[mes_b] - g[mes_a]

        # eixo x
        eixo_x = [f"D+{int(d)}" for d in g.index]

        # =================================================
        # GRÁFICO (inalterado conceitualmente)
        # =================================================
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3]
        )

        fig.add_bar(
            row=1, col=1,
            x=eixo_x,
            y=g[mes_a],
            name=mes_a,
            marker_color="gray"
        )

        fig.add_bar(
            row=1, col=1,
            x=eixo_x,
            y=g[mes_b],
            name=mes_b,
            marker_color="#EA9411"
        )

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
        fig.update_xaxes(title_text="Dia Útil", row=2, col=1)

        fig = aplicar_estilo_plotly(fig)
        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)

        # =================================================
        # TABELA – COMPARAÇÃO ENTRE MESES (SEM RISCO DE ERRO)
        # =================================================
        g.index.name = "DIA_UTIL"
        tabela = g.reset_index()

        tabela["Dia Útil"] = tabela["DIA_UTIL"].apply(lambda x: f"D+{int(x)}")

        # datas reais por mês (podem não existir em todos os dias)
        datas = (
            sub
            .groupby(["MES", "DIA_UTIL"])["DATA_EFETIVACAO"]
            .first()
            .reset_index()
        )

        datas_a = (
            datas[datas["MES"] == mes_a]
            .set_index("DIA_UTIL")["DATA_EFETIVACAO"]
        )

        datas_b = (
            datas[datas["MES"] == mes_b]
            .set_index("DIA_UTIL")["DATA_EFETIVACAO"]
        )

        tabela[f"Data {mes_a}"] = tabela["DIA_UTIL"].map(datas_a).dt.strftime("%d/%m/%Y")
        tabela[f"Data {mes_b}"] = tabela["DIA_UTIL"].map(datas_b).dt.strftime("%d/%m/%Y")

        # formatação moeda
        for col in [mes_a, mes_b, "DESVIO"]:
            tabela[col] = tabela[col].apply(formatar_moeda)

        tabela_final = tabela[
            [
                "Dia Útil",
                f"Data {mes_a}",
                f"Data {mes_b}",
                mes_a,
                mes_b,
                "DESVIO"
            ]
        ]

        st.dataframe(
            tabela_final.style.applymap(
                estilo_desvio, subset=["DESVIO"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            f"⬇️ Baixar comparação – {prod}",
            tabela_final.to_csv(index=False, sep=";", encoding="utf-8-sig"),
            file_name=f"comparacao_{prod}_{mes_a}_vs_{mes_b}.csv",
            mime="text/csv"
        )
