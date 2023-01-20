import streamlit as st
from PIL import Image
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScaler


st.title("Predicción de Infartos al Corazón")
st.subheader("Proyecto Machine Learning")

image = 'src/img/ataque-cardaco.jpg'
st.image(image, caption='Ataque_Cardiaco')

nombre = st.text_input("Introduce el nombre del paciente", max_chars=10)
apellidos = st.text_input("Introduce los apellidos del paciente", max_chars=30)

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("edad",step=1)
    sT_depresion = st.number_input(" Depresión ST inducida por el ejercicio en relación")
    sT_pendiente = st.number_input("La pendiente del segmento ST de ejercicio máximo",step=1)
    max_fecuencia_cardiaca = st.number_input("Frecuencia cardíaca máxima alcanzada", step=1)
    presion_arterial_reposo = st.number_input("Presión arterial en reposo (en mm Hg)", step=1)
    colesterol = st.number_input("Colesterol en mg/dl", step=1)
    sT_pendiente = st.slider(" La pendiente del segmento ST de ejercicio máximo", 0, 2)

with col2:
    sexo = st.selectbox("sexo", ("Hombre", "Mujer"))
    ecg_reposo = st.selectbox("Resultados electrocardiográficos en reposo", ["Normal", " Tener anomalía de la onda ST-T", "Mostrando una hipertrofia ventricular izquierda probable"])
    tipo_defecto = st.selectbox("Talasemia", ["nulo", "Defecto fijo", "Normal", "Defecto reversible"])
    tipo_dolor_pecho = st.selectbox("Tipo de angina", ["Angina tipica", "angina atipica", "Dolor no Anginal", "Asintomatico"])
    angina_ind_ejercicio = st.selectbox(" Angina inducida por el ejercicio", ["Si", "No"])
    velocidad_azucar_sangre = st.selectbox(" Azúcar en sangre en ayunas > 120 mg/dl", ["Si", "No"])
    vasos_principales = st.slider(" Número de vasos principales", 0, 3)

if st.button("Calcular la salida"):

    heart = joblib.load("heart.pkl")

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
    X = X.replace(["Normal", " Tener anomalía de la onda ST-T", "Mostrando una hipertrofia ventricular izquierda probable"], [0, 1, 2])
    X = X.replace(["Angina tipica", "angina atipica", "Dolor no Anginal", "Asintomatico"], [0, 1, 2, 3])
    X = X.replace(["nulo", "Defecto fijo", "Normal", "Defecto reversible"], [0, 1, 2, 3])
    X = X.replace(["No", "Si"], [0, 1])
    X = X.replace(["No", "Si"], [0, 1]) 

    prediction = heart.predict(X)[0]
    
    if prediction == 1:
        prediction = "muy alto"
    else:
        prediction = "poco probable"


    st.text(f"el paciente {nombre} {apellidos} tiene un riesgo {prediction} de sufrir un infarto")