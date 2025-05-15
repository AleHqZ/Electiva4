import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import pandas as pd
import os
import gdown

# Parámetros
width_shape = 224
height_shape = 224
batch_size = 16

# Lista de nombres de las aves
names = [
    'Arrocero Americano','Cerquero Cabecinegro','Cerquero Negrilistado','Cerquero Picofino',
    'Chinchinero Común','Chingolo Pajonalero','Clorospingo Bigotudo','Clorospingo Gorjigrís',
    'Hemispingo Capirotado','Hemispingo Cejudo','Hemispingo Oleaginoso','Hemispingo Orejinegro',
    'Picogordo Azulnegro','Rascador Pico Naranja','Tangara Carinegra','Tangara Coroniblanca',
    'Tangara Crestirrufa','Tangara Encapuchada',
]

# Funciones
@st.cache_resource
def load_model_cached(path):
    return load_model(path)

@st.cache_data
def load_bird_info(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()  # Quitar espacios
    return df

def preprocess_image(img, target_size=(width_shape, height_shape)):
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Descargar modelo automáticamente desde Google Drive
model_path = "modelo.keras"
file_id = "1TCEGlUGMV5sqUolo9UFK4PaD6IeSs--W"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    with st.spinner("Descargando modelo desde Google Drive, espera..."):
        gdown.download(url, model_path, quiet=False)

# Interfaz
st.title("🦜 Clasificador y Evaluador de Aves")

model = None

# -- OPCIÓN 1 para cargar modelo: subir archivo --
uploaded_model = st.file_uploader("Sube el modelo .keras", type=["keras"])
if uploaded_model is not None:
    with open("temp_model.keras", "wb") as f:
        f.write(uploaded_model.getbuffer())
    model = load_model_cached("temp_model.keras")
    st.success("Modelo cargado correctamente desde archivo subido.")

# -- OPCIÓN 2 para cargar modelo: poner ruta manual --
if model is None:
    model_path_manual = st.text_input("O escribe la ruta del modelo .keras", value="")
    if model_path_manual:
        if os.path.exists(model_path_manual):
            model = load_model_cached(model_path_manual)
            st.success("Modelo cargado correctamente desde ruta.")
        else:
            st.warning("No se encontró el archivo en la ruta especificada.")

# -- OPCIÓN 3: cargar modelo descargado automáticamente --
if model is None:
    if os.path.exists(model_path):
        model = load_model_cached(model_path)
        st.success("Modelo cargado correctamente desde Google Drive.")

if model is None:
    st.warning("Por favor sube un modelo o escribe la ruta correcta para continuar.")
    st.stop()

# Cargar Excel con info de aves
excel_path = st.text_input("Ruta del archivo Excel con información de aves", value=r"aves_info_completo.xlsx")
if not os.path.exists(excel_path):
    st.warning("No se encontró el archivo Excel con la información de las aves.")
    st.stop()

bird_info_df = load_bird_info(excel_path)

# Opciones
option = st.radio("Selecciona opción:", ("Clasificar imagen individual", "Evaluar conjunto de test"))

# Opción 1: Clasificar imagen
if option == "Clasificar imagen individual":
    uploaded_file = st.file_uploader("Sube una imagen (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen subida", use_container_width=True)
        img_array = preprocess_image(img)

        preds = model.predict(img_array)[0]
        pred_class = np.argmax(preds)
        pred_prob = preds[pred_class]
        ave_predicha = names[pred_class]

        st.markdown(f"### 🐦 Predicción: **{ave_predicha}** ({pred_prob*100:.2f}%)")

        # Buscar info del ave
        info_ave = bird_info_df[bird_info_df["Nombre"] == ave_predicha]
        if not info_ave.empty:
            ave_info = info_ave.iloc[0]

            st.markdown(f"**📝 Descripción:** {ave_info['Descripción']}")
            st.markdown(f"**🌳 Hábitat:** {ave_info['Hábitat']}")
            st.markdown(f"**🍽️ Alimentación:** {ave_info['Alimentación']}")
            st.markdown(f"**⚠️ Estado de conservación:** {ave_info['Estado']}")

            if pd.notna(ave_info['Busqueda_Google']):
                st.markdown(f"[🔍 Buscar en Google]({ave_info['Busqueda_Google']})")

            col1, col2 = st.columns(2)
            with col1:
                if pd.notna(ave_info['Imagen_1']):
                    st.image(ave_info['Imagen_1'], caption="Imagen 1", use_container_width=True)
            with col2:
                if pd.notna(ave_info['Imagen_2']):
                    st.image(ave_info['Imagen_2'], caption="Imagen 2", use_container_width=True)
        else:
            st.warning("No se encontró información en el Excel para esta especie.")

# Opción 2: Evaluar conjunto de test
elif option == "Evaluar conjunto de test":
    test_data_dir = st.text_input("Ruta del directorio de test (con subcarpetas por clase)", value=r"C:\Users\ronal\Desktop\dataset\Test")
    if test_data_dir and os.path.exists(test_data_dir):
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(width_shape, height_shape),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        with st.spinner("Evaluando conjunto de test..."):
            preds = model.predict(test_generator)
            y_pred = np.argmax(preds, axis=1)
            y_true = test_generator.classes

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            report = classification_report(y_true, y_pred, target_names=names, zero_division=0)

            st.markdown("### Métricas de evaluación")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write(f"**Precision:** {precision:.4f}")
            st.write(f"**Recall:** {recall:.4f}")
            st.write(f"**F1-Score:** {f1:.4f}")

            st.markdown("### Reporte de clasificación detallado")
            st.text(report)

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=ax)
            plt.xlabel("Predicho")
            plt.ylabel("Verdadero")
            plt.title("Matriz de Confusión")
            st.pyplot(fig)
    else:
        st.warning("Por favor ingresa una ruta válida al directorio de test.")
