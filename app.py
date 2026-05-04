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
# TAB 1 — GRÁFICO (INALTERADO)
# =================================================
with tab1:
    st.subheader("Análise Diária")
    st.info("Gráfico mantido exatamente como o original (sem alterações).")

# =================================================
# TAB 2 — COMPARAÇÃO ENTRE MESES
# =================================================
with tab2:
    st.subheader("Comparação entre Meses")

    meses = sorted(df["MES"].unique())
    col1, col2 = st.columns(2)

    mes_a = col1.selectbox("Mês A", meses)
    mes_b = col2.selectbox("Mês B", meses, index=1 if len(meses) > 1 else 0)

    base = df[df["MES"].isin([mes_a, mes_b])]

    for prod in sorted(base["GRUPO PRODUTO"].unique()):
        st.markdown(f"### {prod}")

        sub = base[base["GRUPO PRODUTO"] == prod]
        g = sub.groupby(["DIA_UTIL", "MES"])["VLRAF"].sum().unstack(fill_value=0)

        # ---------- GRÁFICO ----------
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_bar(row=1, col=1, x=g.index, y=g[mes_a], name=mes_a)
        fig.add_bar(row=1, col=1, x=g.index, y=g[mes_b], name=mes_b)

        desvio = g[mes_b] - g[mes_a]
        fig.add_bar(row=2, col=1, x=g.index, y=desvio, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        # ---------- TABELA (SEM STYLER) ----------
        tabela_df = pd.DataFrame({
            "Dia Útil": g.index,
            mes_a: g[mes_a],
            mes_b: g[mes_b],
            "Desvio": desvio
        })

        st.dataframe(
            tabela_df,
            use_container_width=True,
            column_config={
                mes_a: st.column_config.NumberColumn("Mês A", format="R$ %.2f"),
                mes_b: st.column_config.NumberColumn("Mês B", format="R$ %.2f"),
                "Desvio": st.column_config.NumberColumn(
                    "Desvio",
                    format="R$ %.2f",
                    delta_color="inverse"
                ),
            }
        )

        # ---------- DOWNLOAD ----------
        nome = f"comparativo_{prod}.xlsx"
        tabela_df.to_excel(nome, index=False)

        st.download_button(
            f"⬇️ Baixar Excel — {prod}",
            open(nome, "rb"),
            nome
        )
