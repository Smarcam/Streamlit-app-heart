import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Heart Attack Predict",
    initial_sidebar_state="collapsed",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: white;'>Predicción de Infartos al Corazón</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Proyecto Machine Learning</h2>", unsafe_allow_html=True)

image = 'img/ataque-cardaco.jpg'
col1, col2, col3 = st.columns(3)

with col1:
    st.write("")

with col2:
    st.image(image,  use_column_width= 'auto')

with col3:
    st.write("")

@st.cache(allow_output_mutation=True)
def model():
    return joblib.load("model/heart.pkl")

heart_predict = model()

nombre = st.text_input("Introduce el nombre del paciente", max_chars=10, value="Pepe", help= 'Introduce el nombre del paciente')
apellidos = st.text_input("Introduce los apellidos del paciente", max_chars=30, value="Viruela Sarampión", help= 'Introduce los apellidos del paciente')

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("edad",step=1, value=20, help= 'Introduce la edad del paciente')
    sT_depresion = st.number_input(" Depresión ST inducida por el ejercicio en relación", value=1.5, help= 'Introduce el valor de la depresion sistolica del paciente')
    max_fecuencia_cardiaca = st.number_input("Frecuencia cardíaca máxima alcanzada", step=1, value=110, help= 'Introduce la frcuencia cardiaca maxima del paciente')
    presion_arterial_reposo = st.number_input("Presión arterial en reposo (en mm Hg)", step=1, value=120, help= 'Introduce la presion arterial maxima en reposo')
    colesterol = st.number_input("Colesterol en mg/dl", step=1, value=160, help= '3')
    sT_pendiente = st.slider(" La pendiente del segmento ST de ejercicio máximo", 0, 2, value=0, help= 'Introduce la pendiente del segmento sistolico maximo del paciente')

with col2:
    sexo = st.selectbox("sexo", ("Hombre", "Mujer"), help= 'Introduce el sexo del paciente')
    ecg_reposo = st.selectbox("Resultados electrocardiográficos en reposo", ["Normal", "Tener anomalía de la onda ST-T", "Mostrando una hipertrofia ventricular izquierda probable"], help= 'Introduce los resultados del electrocardigrama en reposp del paciente')
    tipo_defecto = st.selectbox("Talasemia", ["nulo", "Defecto fijo", "Normal", "Defecto reversible"], help= 'Introduce la talasemia del paciente')
    tipo_dolor_pecho = st.selectbox("Tipo de angina", ["Angina tipica", "angina atipica", "Dolor no Anginal", "Asintomatico"], help= 'Introduce el tipo de angina del paciente')
    angina_ind_ejercicio = st.selectbox(" Angina inducida por el ejercicio", ["Si", "No"], help= 'Introduce si la angina ha sido producida por el ejercicio')
    velocidad_azucar_sangre = st.selectbox(" Azúcar en sangre en ayunas > 120 mg/dl", ["Si", "No"], help= 'Introduce si tiene un valor de azucar en sangre superior a 120 mg/dl')
    vasos_principales = st.slider(" Número de vasos principales", 0, 3, value=3, help= 'introduce el numero de vasos principales')

if st.button("Calcular la salida"):

    input_data_num = [edad, presion_arterial_reposo, colesterol, max_fecuencia_cardiaca, sT_depresion]
    input_data_cat = [sexo, tipo_dolor_pecho, velocidad_azucar_sangre, ecg_reposo, angina_ind_ejercicio, sT_pendiente, vasos_principales, tipo_defecto]

    input_data_num_array = np.asarray(input_data_num)
    input_data_cat_array = np.asarray(input_data_cat)

    input_data_reshape_num = input_data_num_array.reshape(1,-1)
    input_data_reshape_cat = input_data_cat_array.reshape(1,-1)

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(input_data_reshape_num)

    X = np.concatenate((scaled_array, input_data_reshape_cat), axis = 1)
    
    X = pd.DataFrame([[edad, presion_arterial_reposo, colesterol, max_fecuencia_cardiaca, sT_depresion, sexo, tipo_dolor_pecho, velocidad_azucar_sangre, ecg_reposo, angina_ind_ejercicio, sT_pendiente, vasos_principales, tipo_defecto]], 
    columns = ["Edad", "Presion_arterial_reposo", "Colesterol", "Max_fecuencia_cardiaca", "ST_depresion", "Sexo", "Tipo_dolor_pecho", "Velocidad_azucar_sangre", "ECG_reposo", "Angina_ind._ejercicio", "ST_pendiente", "Vasos_principales", "Tipo_defecto"])

    X = X.replace(["Mujer", "Hombre"], [0, 1])
    X = X.replace(["Normal", "Tener anomalía de la onda ST-T", "Mostrando una hipertrofia ventricular izquierda probable"], [0, 1, 2])
    X = X.replace(["Angina tipica", "angina atipica", "Dolor no Anginal", "Asintomatico"], [0, 1, 2, 3])
    X = X.replace(["nulo", "Defecto fijo", "Normal", "Defecto reversible"], [0, 1, 2, 3])
    X = X.replace(["No", "Si"], [0, 1])
    X = X.replace(["No", "Si"], [0, 1]) 

    prediction = heart_predict.predict(X)[0]
    
    if prediction == 1:
        prediction = "muy alto"
    else:
        prediction = "poco probable"

    st.text(f"el paciente {nombre} {apellidos} tiene un riesgo {prediction} de sufrir un infarto")