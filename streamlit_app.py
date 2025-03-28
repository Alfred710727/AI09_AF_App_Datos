import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# Configuración inicial
st.set_page_config(
    page_title="Sistema de Soporte a Decisiones",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carga de datos
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Formato no soportado")
        return None

st.sidebar.title("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu dataset", type=['csv', 'xls', 'xlsx', 'json'])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Sección de Análisis Exploratorio
    st.sidebar.header("Análisis Exploratorio")
    show_eda = st.sidebar.checkbox("Mostrar Análisis Exploratorio")

    if show_eda:
        st.header("Análisis Exploratorio de Datos")

        # Resumen estadístico
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Primeras filas")
            st.write(df.head())
        with col2:
            st.subheader("Tipos de datos")
            st.write(df.dtypes.astype(str))

        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.write(df.describe(include='all'))

        # Valores faltantes
        st.subheader("Valores Faltantes")
        missing_values = df.isnull().sum()
        st.bar_chart(missing_values[missing_values > 0])

        # Correlación entre variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Matriz de Correlación")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # Sección de Visualización
    st.sidebar.header("Visualización Interactiva")

    # Filtrar opciones de gráfico según estado de EDA
    available_plots = ["Histograma", "Scatter Plot", "Box Plot", "Bar Plot", "Pairplot"]
    if not show_eda:
        available_plots.append("Heatmap")

    plot_type = st.sidebar.selectbox("Tipo de Gráfico", available_plots)

    if plot_type:
        st.header(f"{plot_type} Interactivo")

        # Selección de variables
        if plot_type in ["Histograma", "Box Plot"]:
            selected_col = st.selectbox("Selecciona una variable", df.columns)
        elif plot_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("Variable X", df.columns)
            y_col = col2.selectbox("Variable Y", df.columns)

            # Selector de colores primarios
            color_options = ["Azul", "Rojo", "Verde", "Amarillo", "Morado"]
            color_col = st.selectbox("Color de los puntos", color_options)

            # Validación de columnas numéricas
            if not is_numeric_dtype(df[x_col]):
                st.warning(f"La variable X '{x_col}' no es numérica")
            if not is_numeric_dtype(df[y_col]):
                st.warning(f"La variable Y '{y_col}' no es numérica")

        elif plot_type == "Bar Plot":
            # Validar existencia de columnas categóricas y numéricas
            cat_cols_available = list(df.select_dtypes(include=['object']).columns)
            num_cols_available = list(df.select_dtypes(include=np.number).columns)

            if not cat_cols_available or not num_cols_available:
                st.warning("El dataset no contiene columnas categóricas y/o numéricas necesarias")
            else:
                cat_col = st.selectbox("Variable Categórica", cat_cols_available)
                num_col = st.selectbox("Variable Numérica", num_cols_available)

        elif plot_type == "Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns

        elif plot_type == "Pairplot":
            cols = st.multiselect("Selecciona variables", df.columns)

        # Generación de gráficos
        if plot_type == "Histograma":
            if is_numeric_dtype(df[selected_col]):
                fig = px.histogram(df, x=selected_col, marginal="box", nbins=30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("La variable seleccionada no es numérica")

        elif plot_type == "Scatter Plot":
            # Validar columnas numéricas
            if not is_numeric_dtype(df[x_col]) or not is_numeric_dtype(df[y_col]):
                st.warning("Ambas variables deben ser numéricas")
            else:
                # Configurar color
                color_map = {
                    "Rojo": "#FF0000",
                    "Verde": "#00FF00",
                    "Amarillo": "#FFFF00",
                    "Morado": "#800080"
                }

                if color_col == "Azul":
                    fig = px.scatter(df, x=x_col, y=y_col)
                else:
                    selected_color = color_map[color_col]
                    fig = px.scatter(df, x=x_col, y=y_col)
                    fig.update_traces(marker=dict(color=selected_color))

                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box Plot":
            fig = px.box(df, y=selected_col)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Bar Plot":
            # Validación doble
            if 'cat_col' not in locals() or 'num_col' not in locals():
                st.warning("Seleccione variables válidas")
            elif cat_col not in df.columns or num_col not in df.columns:
                st.warning("Las variables seleccionadas no existen en el dataset")
            else:
                counts = df.groupby(cat_col)[num_col].mean().reset_index()
                fig = px.bar(counts, x=cat_col, y=num_col)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Heatmap" and not show_eda:
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay suficientes columnas numéricas para generar el heatmap")

        elif plot_type == "Pairplot":
            if cols:
                pair_df = df[cols]
                fig = sns.pairplot(pair_df)
                st.pyplot(fig)

    # Exportar informe
    if 'mostrar_resumen' not in st.session_state:
        st.session_state.mostrar_resumen = False

    texto_boton = "**Ocultar Resumen**" if st.session_state.mostrar_resumen else "**Resumen Dataset**"

    if st.sidebar.button(texto_boton):
        st.session_state.mostrar_resumen = not st.session_state.mostrar_resumen

    if st.session_state.mostrar_resumen:
        st.sidebar.write(f"Dataset: **{uploaded_file.name}**")
        st.sidebar.write(f"Número de filas: **{len(df)}**")
        st.sidebar.write(f"Número de columnas: **{len(df.columns)}**")

else:
    st.info("Por favor, sube un dataset para comenzar")
