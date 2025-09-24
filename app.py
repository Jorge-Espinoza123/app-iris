import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

# Variables de conexión
USER = "postgres.aedpfifnkhudsoecnimt"
PASSWORD = "supabase123"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para probar conexión inicial
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    cursor = connection.cursor()
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Connection successful! Current Time:", result)
    cursor.close()
    connection.close()
except Exception as e:
    st.write(str(e))

# Función para cargar los modelos
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

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

        # Guardar en la base de datos e imprimir historial
        try:
            connection = psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            )
            cursor = connection.cursor()
            
            # Insertar registro
            insert_query = """
                INSERT INTO table_iris (longpetalo, longsepalo, anchopetalo, anchosepalo, prediction, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW());
            """
            cursor.execute(insert_query, (
                petal_length,   # longpetalo
                sepal_length,   # longsepalo
                petal_width,    # anchopetalo
                sepal_width,    # anchosepalo
                predicted_species
            ))
            connection.commit()

            # Consultar historial completo
            cursor.execute("""
                SELECT created_at, longsepalo, anchosepalo, longpetalo, anchopetalo, prediction
                FROM table_iris
                ORDER BY created_at DESC;
            """)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]

            df = pd.DataFrame(rows, columns=cols)

            cursor.close()
            connection.close()

            # Mostrar historial en tabla
            st.subheader("Historial de predicciones")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Error al insertar o consultar la base de datos: {e}")

        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")


