import streamlit as st
import pandas as pd
import joblib
import numpy as np

def log1p_safe(x):
    return np.log1p(np.clip(x, 0, None))

# ===============================
# Configurações iniciais
# ===============================
st.set_page_config(
    page_title="Modelo de Regressão",
    layout="wide"
)

st.title("📊 Aplicação de Regressão com Upload de CSV")

# ===============================
# Colunas esperadas pelo modelo
# ===============================
NUM_COLS = [
    'qtd_filhos',
    'idade',
    'tempo_emprego',
    'qt_pessoas_residencia',
    'renda'
]

CAT_COLS = [
    'posse_de_veiculo'
]

EXPECTED_COLS = NUM_COLS + CAT_COLS

# ===============================
# Carregamento do modelo
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("modelo_regressao.pkl")

modelo = load_model()

# ===============================
# Upload do arquivo
# ===============================
uploaded_file = st.file_uploader(
    "📤 Faça upload do arquivo CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Dados carregados")
        st.dataframe(df)

        # ===============================
        # Validação de colunas
        # ===============================
        missing_cols = set(EXPECTED_COLS) - set(df.columns)

        if missing_cols:
            st.error(
                f"❌ O arquivo não possui as colunas obrigatórias: {missing_cols}"
            )
            st.stop()

        df_model = df[EXPECTED_COLS]

        # ===============================
        # Previsão
        # ===============================
        previsoes = modelo.predict(df_model)

        df_resultado = df.copy()
        df_resultado["previsao"] = previsoes

        st.subheader("✅ Resultado das previsões")
        st.dataframe(df_resultado)

        # ===============================
        # Download
        # ===============================
        csv = df_resultado.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Baixar CSV com previsões",
            data=csv,
            file_name="resultado_previsoes.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Erro ao processar o arquivo")
        st.exception(e)