import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso (Carga de Archivos)")
st.write("Esta aplicación permite cargar un archivo de datos, procesar sus variables y realizar predicciones por lote utilizando un modelo de Bagging pre-entrenado.")

# 1. Sección para cargar archivos
st.header("1. Cargar Archivo de Datos")
file_uploaded = st.file_uploader("Sube tu archivo (Excel o CSV):", type=["xlsx", "xls", "csv"])

if file_uploaded is not None:
    try:
        # Leer el archivo dependiendo de la extensión
        if file_uploaded.name.endswith('.csv'):
            df_input = pd.read_csv(file_uploaded)
        else:
            df_input = pd.read_excel(file_uploaded)
        
        st.subheader("Datos Originales Cargados")
        st.dataframe(df_input.head())
        
        # Botón para procesar y predecir
        if st.button("Procesar y Realizar Predicción"):
            df_procesado = df_input.copy()
            
            # Eliminar columnas ID y Año - Semestre si existen
            columnas_a_eliminar = ['ID', 'Año - Semestre']
            df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
            
            # Cargar la lista de columnas One-Hot desde el archivo
            columnas_one_hot = joblib.load('one_hot_columns.joblib')
            
            # Separar las columnas One-Hot que corresponden a Felder
            si_columnas_one_hot = [col for col in columnas_one_hot if 'Felder_' in col]
            
            # Aplicar One-Hot encoding de forma manual para cada fila basada en la columna 'Felder'
            if 'Felder' in df_procesado.columns:
                for col_name in si_columnas_one_hot:
                    valor_esperado = col_name.replace('Felder_', '')
                    df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
                # Eliminar la variable original Felder
                df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
            else:
                st.warning("La columna 'Felder' no fue encontrada en el archivo. Se asumirán valores en 0.")
                for col_name in si_columnas_one_hot:
                    df_procesado[col_name] = 0.0
            
            # Normalizar la variable 'Examen_admisión' con 'min_max_scaler.joblib'
            scaler = joblib.load('min_max_scaler.joblib')
            
            # Verificar la presencia de la columna Examen_admisión
            col_examen = None
            for col in ['Examen_admisión', 'Examen_admision', 'Examen_admisión_scaled']:
                if col in df_procesado.columns:
                    col_examen = col
                    break
            
            if col_examen is not None:
                # El escalador espera una estructura 2D
                scaled_values = scaler.transform(df_procesado[[col_examen]])
                df_procesado['Examen_admision_scaled'] = scaled_values
                if col_examen != 'Examen_admision_scaled':
                    df_procesado = df_procesado.drop(columns=[col_examen], errors='ignore')
            else:
                st.error("No se encontró la columna 'Examen_admisión' en el archivo cargado.")
                st.stop()
            
            # Reordenar las columnas exactamente para que coincidan con 'columnas_one_hot'
            columnas_finales = [col for col in columnas_one_hot if col in df_procesado.columns]
            df_procesado = df_procesado[columnas_finales]
            
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_procesado.head())
            
            # Cargar el modelo
            model = joblib.load('bagging_optimizado.joblib')
            
            # Realizar las predicciones
            predicciones = model.predict(df_procesado)
            
            # Agregar predicciones al DataFrame original para mostrar los resultados de forma amigable
            df_resultado = df_input.copy()
            df_resultado['Nota_final_estimada'] = predicciones
            
            st.success("¡Predicciones realizadas con éxito!")
            st.subheader("Resultados Finales")
            st.dataframe(df_resultado)
            
            # Opción para descargar los resultados
            csv = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Resultados en CSV",
                data=csv,
                file_name="predicciones_aprobacion.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo para comenzar.")
