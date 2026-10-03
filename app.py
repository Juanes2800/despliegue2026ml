import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación", layout="wide")

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación permite realizar predicciones utilizando un modelo de Bagging pre-entrenado, ya sea de forma individual o de manera masiva cargando un archivo.")

# Menú de navegación principal
opcion = st.sidebar.radio("Selecciona el modo de predicción:", ["Predicción Individual (Manual)", "Predicción Masiva (Subir Archivo)"])

# Cargar los recursos comunes
try:
    columnas_one_hot = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
except Exception as e:
    st.error(f"Error al cargar los modelos o recursos: {e}")
    st.stop()

# --- MODO 1: PREDICCIÓN INDIVIDUAL ---
if opcion == "Predicción Individual (Manual)":
    st.header("Predicción Individual")
    st.write("Ingresa los valores manualmente para estimar la nota final del estudiante.")
    
    # Opciones de Felder basadas en las columnas One-Hot
    si_columnas_one_hot = [col for col in columnas_one_hot if 'Felder_' in col]
    opciones_felder = [col.replace('Felder_', '') for col in si_columnas_one_hot]
    if not opciones_felder:
        opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    col1, col2 = st.columns(2)
    with col1:
        felder_input = st.selectbox("Estilo de aprendizaje (Felder):", opciones_felder)
    with col2:
        examen_input = st.number_input("Calificación Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)
    
    if st.button("Calcular Predicción Individual"):
        try:
            # Crear DataFrame de un registro
            df_single = pd.DataFrame([{'Felder': felder_input, 'Examen_admisión': examen_input}])
            
            # Procesamiento One-Hot
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_single[col_name] = (df_single['Felder'] == valor_esperado).astype(float)
            
            df_single = df_single.drop(columns=['Felder'], errors='ignore')
            
            # Normalizar Examen_admisión
            df_single['Examen_admision_scaled'] = scaler.transform(df_single[['Examen_admisión']])[0][0]
            df_single = df_single.drop(columns=['Examen_admisión'], errors='ignore')
            
            # Alinear columnas
            df_single = df_single[columnas_one_hot]
            
            # Predecir
            prediccion = model.predict(df_single)[0]
            
            st.success(f"### La Nota Final Estimada para el estudiante es: **{prediccion:.4f}**")
            
        except Exception as e:
            st.error(f"Error al procesar la predicción individual: {e}")

# --- MODO 2: PREDICCIÓN MASIVA ---
elif opcion == "Predicción Masiva (Subir Archivo)":
    st.header("Predicción Masiva por Lote")
    st.write("Sube un archivo de datos para obtener las estimaciones de múltiples estudiantes simultáneamente.")
    
    file_uploaded = st.file_uploader("Sube tu archivo (Excel o CSV):", type=["xlsx", "xls", "csv"])
    
    if file_uploaded is not None:
        try:
            if file_uploaded.name.endswith('.csv'):
                df_input = pd.read_csv(file_uploaded)
            else:
                df_input = pd.read_excel(file_uploaded)
            
            st.subheader("Vista Previa de los Datos Cargados")
            st.dataframe(df_input.head())
            
            if st.button("Procesar y Predicir Lote"):
                df_procesado = df_input.copy()
                
                # Eliminar columnas irrelevantes si existen
                columnas_a_eliminar = ['ID', 'Año - Semestre']
                df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
                
                # Procesar One-Hot para Felder
                si_columnas_one_hot = [col for col in columnas_one_hot if 'Felder_' in col]
                if 'Felder' in df_procesado.columns:
                    for col_name in si_columnas_one_hot:
                        valor_esperado = col_name.replace('Felder_', '')
                        df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
                    df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
                else:
                    st.warning("La columna 'Felder' no fue encontrada. Se rellenará con 0.0.")
                    for col_name in si_columnas_one_hot:
                        df_procesado[col_name] = 0.0
                
                # Normalizar Examen_admisión
                col_examen = None
                for col in ['Examen_admisión', 'Examen_admision', 'Examen_admisión_scaled']:
                    if col in df_procesado.columns:
                        col_examen = col
                        break
                
                if col_examen is not None:
                    scaled_values = scaler.transform(df_procesado[[col_examen]])
                    df_procesado['Examen_admision_scaled'] = scaled_values
                    if col_examen != 'Examen_admision_scaled':
                        df_procesado = df_procesado.drop(columns=[col_examen], errors='ignore')
                else:
                    st.error("No se encontró la columna de calificación del Examen de Admisión.")
                    st.stop()
                
                # Alinear columnas para el modelo
                df_procesado = df_procesado[columnas_one_hot]
                
                # Predecir
                predicciones = model.predict(df_procesado)
                
                # Mostrar Resultados
                df_resultado = df_input.copy()
                df_resultado['Nota_final_estimada'] = predicciones
                
                st.success("¡Predicciones por lote completadas con éxito!")
                st.subheader("Resultados Generados")
                st.dataframe(df_resultado)
                
                # Botón de Descarga
                csv = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Resultados en CSV",
                    data=csv,
                    file_name="predicciones_lote.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
    else:
        st.info("Por favor, carga un archivo compatible en formato Excel o CSV para realizar las predicciones.")
