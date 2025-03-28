import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import base64
import pdfkit
from io import BytesIO

# Configuración inicial
st.set_page_config(
    page_title="Sistema de Soporte a Decisiones",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variables de estado
if 'report_html' not in st.session_state:
    st.session_state.report_html = []
if 'show_readme' not in st.session_state:
    st.session_state.show_readme = False
if 'previous_state' not in st.session_state:
    st.session_state.previous_state = {}
if 'current_plot' not in st.session_state:
    st.session_state.current_plot = None

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

# Funciones auxiliares para generación de reportes
def df_to_html(df):
    return df.to_html(classes='dataframe', border=0, index=False)

def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Barra lateral
st.sidebar.title("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu dataset", type=['csv', 'xls', 'xlsx', 'json'])

# Menú con nuevos botones
with st.sidebar:
    st.header("Menú")
    readme_btn = st.button("README")
    exit_btn = st.button("Salir")
    
    if readme_btn:
        st.session_state.previous_state = {
            'show_eda': st.session_state.get('show_eda', False),
            'plot_type': st.session_state.get('plot_type', None)
        }
        st.session_state.show_readme = True
        st.experimental_rerun()
        
    if exit_btn:
        if uploaded_file is None:
            st.session_state.clear()
        else:
            st.session_state.show_readme = False
            st.session_state.show_eda = st.session_state.previous_state.get('show_eda', False)
            st.session_state.plot_type = st.session_state.previous_state.get('plot_type', None)
        st.experimental_rerun()

# Control de estado para README
if st.session_state.show_readme:
    st.header("README")
    st.markdown("""
    **Pasos de la funcionalidad de la aplicación:**
    1. Carga tu dataset en la sección lateral
    2. Activa el Análisis Exploratorio para ver estadísticas básicas
    3. Selecciona tipo de gráfico en Visualización Interactiva
    4. Configura variables y opciones del gráfico
    5. Visualiza resultados y descarga informes
    """)
    st.stop()

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Sección de Análisis Exploratorio
    st.sidebar.header("Análisis Exploratorio")
    show_eda = st.sidebar.checkbox("Mostrar Análisis Exploratorio")

    if show_eda:
        st.header("Análisis Exploratorio de Datos")
        st.session_state.report_html.append("<h1>Análisis Exploratorio de Datos</h1>")

        # Resumen estadístico
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Primeras filas")
            st.write(df.head())
            st.session_state.report_html.append(f"<h2>Primeras filas</h2>{df_to_html(df.head())}")
        with col2:
            st.subheader("Tipos de datos")
            st.write(df.dtypes.astype(str))
            st.session_state.report_html.append(f"<h2>Tipos de datos</h2>{df_to_html(df.dtypes.astype(str).reset_index())}")

        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        stats = df.describe(include='all').T
        st.write(stats)
        st.session_state.report_html.append(f"<h2>Estadísticas Descriptivas</h2>{df_to_html(stats)}")

        # Valores faltantes
        st.subheader("Valores Faltantes")
        missing_values = df.isnull().sum()
        st.bar_chart(missing_values[missing_values > 0])
        fig, ax = plt.subplots()
        ax.bar(missing_values.index, missing_values.values)
        ax.set_title("Valores Faltantes")
        st.session_state.report_html.append(f"<h2>Valores Faltantes</h2><img src='data:image/png;base64,{fig_to_base64(fig)}'/>")

        # Correlación entre variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("Matriz de Correlación")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
            img_bytes = fig.to_image(format="png")
            st.session_state.report_html.append(f"<h2>Matriz de Correlación</h2><img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}'>")
        else:
            st.warning("No hay suficientes columnas numéricas")

    # Sección de Visualización
    st.sidebar.header("Visualización Interactiva")

    # Filtrar opciones de gráfico según estado de EDA
    available_plots = ["Histograma", "Scatter Plot", "Box Plot", "Bar Plot", "Pairplot"]
    if not show_eda:
        available_plots.append("Heatmap")

    plot_type = st.sidebar.selectbox("Tipo de Gráfico", available_plots)

    if plot_type:
        st.header(f"{plot_type} Interactivo")
        st.session_state.report_html.append(f"<h1>{plot_type} Interactivo</h1>")

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
                img_bytes = fig.to_image(format="png")
                st.session_state.report_html.append(f"<img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}'>")
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
                img_bytes = fig.to_image(format="png")
                st.session_state.report_html.append(f"<img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}'>")

        elif plot_type == "Box Plot":
            fig = px.box(df, y=selected_col)
            st.plotly_chart(fig, use_container_width=True)
            img_bytes = fig.to_image(format="png")
            st.session_state.report_html.append(f"<img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}'>")

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
                img_bytes = fig.to_image(format="png")
                st.session_state.report_html.append(f"<img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}'>")

        elif plot_type == "Heatmap" and not show_eda:
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                img_bytes = fig.to_image(format="png")
                st.session_state.report_html.append(f"<img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}'>")
            else:
                st.warning("No hay suficientes columnas numéricas para generar el heatmap")

        elif plot_type == "Pairplot":
            if cols:
                pair_df = df[cols]
                fig = sns.pairplot(pair_df)
                st.pyplot(fig)
                img = BytesIO()
                fig.savefig(img, format='png')
                img.seek(0)
                st.session_state.report_html.append(f"<img src='data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'>")

    # Botón de descarga
    if st.button("Descargar Resultados"):
        html_content = f"""
        <html>
            <head><title>Reporte</title></head>
            <body>
                <div style="width: 80%; margin: 0 auto;">
                    {''.join(st.session_state.report_html)}
                </div>
            </body>
        </html>
        """
        pdf = pdfkit.from_string(html_content, False)
        st.download_button(
            label="Descargar PDF",
            data=pdf,
            file_name="reporte.pdf",
            mime="application/pdf"
        )

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
