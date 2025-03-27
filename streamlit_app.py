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
    plot_type = st.sidebar.selectbox("Tipo de Gráfico", 
                                    ["Histograma", "Scatter Plot", 
                                     "Box Plot", "Bar Plot", 
                                     "Heatmap", "Pairplot"])
    
    if plot_type:
        st.header(f"{plot_type} Interactivo")
        
        # Selección de variables
        if plot_type in ["Histograma", "Box Plot"]:
            selected_col = st.selectbox("Selecciona una variable", df.columns)
        elif plot_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("Variable X", df.columns)
            y_col = col2.selectbox("Variable Y", df.columns)
            color_col = st.selectbox("Color por variable categórica", 
                                    ["Ninguna"] + list(df.select_dtypes(include=['object']).columns))
        elif plot_type == "Bar Plot":
            cat_col = st.selectbox("Variable Categórica", df.select_dtypes(include=['object']).columns)
            num_col = st.selectbox("Variable Numérica", df.select_dtypes(include=np.number).columns)
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
            if color_col == "Ninguna":
                fig = px.scatter(df, x=x_col, y=y_col)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            fig = px.box(df, y=selected_col)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Bar Plot":
            counts = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(counts, x=cat_col, y=num_col)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Heatmap":
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Pairplot":
            if cols:
                pair_df = df[cols]
                fig = sns.pairplot(pair_df)
                st.pyplot(fig)
        
    # Exportar informe
    # st.sidebar.header("Resumen")
    if st.sidebar.button("Resumen Dataset"):
        # st.sidebar.markdown("### Resumen del Análisis")
        st.sidebar.write(f"Dataset: {uploaded_file.name}")
        st.sidebar.write(f"Número de filas: {len(df)}")
        st.sidebar.write(f"Número de columnas: {len(df.columns)}")

else:
    st.info("Por favor, sube un dataset para comenzar")
