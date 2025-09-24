import streamlit as st
import psycopg2
import pandas as pd
import joblib
import pickle
import numpy as np

# ============================
# Credenciales Supabase / Postgres
# ============================
USER = "postgres.aedpfifnkhudsoecnimt"
PASSWORD = "supabase123"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# ============================
# Conexión a la base de datos
# ============================
def get_connection():
    return psycopg2.connect(
        host=HOST,
        dbname=DBNAME,
        user=USER,
        password=PASSWORD,
        port=PORT
    )

# ============================
# Insertar registro evitando duplicados
# ============================
def insert_prediction(longsepalo, anchosepalo, longpetalo, anchopetalo, prediction):
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
    INSERT INTO table_iris (longsepalo, anchosepalo, longpetalo, anchopetalo, prediction, created_at)
    SELECT %s, %s, %s, %s, %s, NOW()
    WHERE NOT EXISTS (
        SELECT 1 FROM table_iris
        WHERE longsepalo = %s AND anchosepalo = %s AND longpetalo = %s AND anchopetalo = %s AND prediction = %s
    );
    """
    
    cur.execute(query, (
        longsepalo, anchosepalo, longpetalo, anchopetalo, prediction,
        longsepalo, anchosepalo, longpetalo, anchopetalo, prediction
    ))
    
    conn.commit()
    cur.close()
    conn.close()

# ============================
# Obtener historial
# ============================
def get_history():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM table_iris ORDER BY created_at DESC", conn)
    conn.close()
    return df

# ============================
# Cargar modelos
# ============================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

# ============================
# Interfaz Streamlit
# ============================
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelo
model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        # Predicción
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_species = model_info['target_names'][prediction_idx]

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

        # Insertar en la BD evitando duplicados
        insert_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species)

        # Mostrar historial
        st.subheader("📊 Historial de predicciones")
        history = get_history()
        st.dataframe(history)

        # Mostrar probabilidades
        st.subheader("Probabilidades:")
        for species, prob in zip(model_info['target_names'], probabilities):
            st.write(f"- {species}: {prob:.1%}")


