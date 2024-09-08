import streamlit as st
import joblib
import pandas as pd

# Definir la función de preprocesamiento
def preprocesar_datos(df_humedades):
    df_humedades['chanel'] = df_humedades['chanel'].map({
        'Landing Search': 0, 'Meta Ads': 1, 'Habitissimo': 2, 'Landing-display': 3
    }).astype('Int64')

    df_humedades['ubicacion_geografica'] = df_humedades['ubicacion_geografica'].map({
        'Comunidad de Madrid': 0, 'Barcelona': 1, 'Tarragona': 2, 'Girona': 3, 'Lleida': 4
    }).astype('Int64')

    df_humedades['tipo_problema'] = df_humedades['tipo_problema'].map({
       'Capilaridad': 1, 'Filtración': 2, 'Condensación': 3
    }).astype('Int64')

    return df_humedades

# Cargar el modelo entrenado
model = joblib.load('rf_humedad_model.pkl')

# Título de la aplicación
st.title('Predicción de Cerrado Ganado para Humedades')

# Sidebar con la información de la aplicación
st.sidebar.title("Información de la Aplicación")
st.sidebar.info(
    """
    Esta aplicación predice si una oferta de servicios de tratamiento de humedades resultará en un "Cerrado Ganado".
    
    **Cómo usar la aplicación:**
    
    1. Selecciona el **Canal de origen** a través del cual se ha generado el cliente.
    2. Elige la **Ubicación Geográfica** donde se encuentra la propiedad afectada.
    3. Selecciona el **Tipo de Problema** de humedad.
    
    Luego, haz clic en el botón **Realizar Predicción** para ver si la oferta será "Cerrada Ganada" y la probabilidad asociada.
    """
)

# Input del usuario para cada característica
chanel = st.selectbox('Canal de origen', ['Landing Search', 'Meta Ads', 'Habitissimo', 'Landing-display'])
ubicacion_geografica = st.selectbox('Ubicación Geográfica', ['Comunidad de Madrid', 'Barcelona', 'Tarragona', 'Girona', 'Lleida'])
tipo_problema = st.selectbox('Tipo de Problema', ['Capilaridad', 'Filtración', 'Condensación'])

# Crear un DataFrame con los inputs del usuario
input_data = pd.DataFrame({
    'chanel': [chanel],
    'ubicacion_geografica': [ubicacion_geografica],
    'tipo_problema': [tipo_problema]
})

# Botón para ejecutar la predicción
if st.button('Realizar Predicción'):
    # Aplicar el preprocesamiento usando la función
    input_data_preprocessed = preprocesar_datos(input_data)

    # Realizar la predicción
    prediction = model.predict(input_data_preprocessed)
    prediction_proba = model.predict_proba(input_data_preprocessed)

    # Mostrar el resultado de la predicción
    st.write(f'**Predicción:** {"Cerrado Ganado" if prediction[0] == 1 else "No Cerrado"}')
    st.write(f'**Probabilidad de Cerrado Ganado:** {prediction_proba[0][1]:.2f}')


