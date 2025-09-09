import streamlit as st
import requests
from data_ingestion import DataIngestion
import os
from dotenv import load_dotenv

load_dotenv()

db_params={
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}



ingestion = DataIngestion(db_params)

@st.cache_data
def get_products():
    df = ingestion.read_products()
    return df

products_df = get_products()

st.title("Teste Elasticidade de Produtos")

# Seleciona produto
product_name = st.selectbox("Escolha o produto", products_df["name"])
price_original = products_df.loc[products_df["name"] == product_name, "price"].values[0]

st.write(f"Preço atual: {price_original:.2f}")

# Percentual de aumento
percent_increase = st.number_input("Aumentar preço (%)", min_value=0.0, value=0.0, step=1.0)

# Calcula novo preço
new_price = price_original * (1 + percent_increase / 100)
st.write(f"Novo preço: {new_price:.2f}")

# --- BOTÃO DE ENVIO ---
if st.button("Calcular elasticidade"):
    payload = {
        "promotion": False,  # você pode colocar True se quiser testar promoção
        "name": product_name,
        "price": float(new_price)
    }

    try:
        response = requests.post("http://localhost:8000/prediction/elasticidade", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.success("Predição realizada com sucesso!")
            st.json(data)
        else:
            st.error(f"Erro na API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Falha ao conectar na API: {e}")
