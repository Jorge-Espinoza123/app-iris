import streamlit as st
import pandas as pd
import pickle
import psycopg2
from psycopg2.extras import RealDictCursor

# =============================
# Cargar modelo y etiquetas
# =============================
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ["setosa", "versicolor", "virginica"]

# =============================
# Conexión a la base de datos
# =============================
def get_connection():
    return psycopg2.connect(
        host="your_host",
        dbname="your_db",
        user="your_user",
        password="your_password",
        port="5432"
    )

# =============================
# Guardar predicción en la BD
# =============================
def save_prediction(sepal_length, sepal_width, petal_length, petal_width, prediction):
    conn = get_connection()
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO table_iris (created_at, longsepalo, anchosepalo, longpetalo, anchopetalo, prediction)
    VALUES (NOW(), %s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, (
        sepal_length,    # longsepalo
        sepal_width,     # anchosepalo
        petal_length,    # longpetalo
        petal_width,     # anchopetalo
        prediction       # prediction
    ))
    conn.commit()
    cursor.close()
    conn.close()

# =============================
# Obtener historial
# =============================
def load_history():
    conn = get_connection()
    query = "SELECT created_at, longsepalo, anchosepalo, longpetalo, anchopetalo, prediction FROM table_iris ORDER BY created_at DESC LIMIT 10;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# =============================
# Interfaz Streamlit
# =============================
st.title("🌸 Clasificador de Iris")

# Entradas de usuario
sepal_length = st.number_input("Largo del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Largo del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0)

if st.button("Predecir Especie"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_species = class_names[prediction]
    confidence = round(probabilities[prediction] * 100, 2)

    # Mostrar resultado
    st.success(f"🌿 Especie predicha: **{predicted_species}**")
    st.write(f"Confianza: **{confidence}%**")

    st.write("Probabilidades:")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name}: {probabilities[i]*100:.1f}%")

    # Guardar en la BD
    save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species)
    st.success("✅ Registro guardado en la base de datos")

# =============================
# Mostrar historial
# =============================
st.subheader("Historial de Predicciones")
history_df = load_history()
st.dataframe(history_df)

