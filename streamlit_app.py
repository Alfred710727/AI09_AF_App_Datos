import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# Configuración inicial con tema oscuro
st.set_page_config(
    page_title="Sistema de Soporte a Decisiones",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Inyección de CSS mejorado
st.markdown(
    # """
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #4CAF50;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    </style>
    # """,
    unsafe_allow_html=True
)

# Carga de datos con mensaje de carga
@st.cache_data(show_spinner="Cargando datos... 🔄")
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    st.error("Formato no soportado ❌")
    return None

# Estado de la aplicación
if 'show_readme' not in st.session_state:
    st.session_state.show_readme = False
if 'mostrar_resumen' not in st.session_state:
    st.session_state.mostrar_resumen = False

# Sidebar mejorada
with st.sidebar:
    st.title("📊 Carga de Datos")
    uploaded_file = st.file_uploader(
        "Sube tu dataset",
        type=['csv', 'xls', 'xlsx', 'json'],
        help="Soporta CSV, Excel y JSON"
    )
    
    st.button(
        "README 📖",
        on_click=lambda: setattr(st.session_state, 'show_readme', True),
        help="Ver guía de uso"
    )

    if uploaded_file:
        with st.expander("Configuración Avanzada ⚙️"):
            st.checkbox("Modo Experto", help="Habilita opciones avanzadas")
            st.color_picker("Color Primario", "#4CAF50", help="Color principal de la app")

# Contenido principal
if st.session_state.show_readme:
    st.header("📚 Guía de Uso")
    st.markdown("""
    **Pasos para utilizar:**
    1. 📁 Carga tu dataset usando el uploader en la sidebar
    2. 📊 Activa el análisis exploratorio para ver estadísticas
    3. 📈 Selecciona tipo de gráfico en la sección de visualización
    4. 🎨 Personaliza los parámetros del gráfico
    5. 📉 Analiza los resultados generados
    
    **Tips:**
    - Usa el modo experto para opciones avanzadas
    - Los gráficos se pueden descargar con clic derecho
    """)
    st.button("Salir", on_click=lambda: setattr(st.session_state, 'show_readme', False))

elif uploaded_file:
    df = load_data(uploaded_file)
    
    # Barra de herramientas superior
    col1, col2, col3 = st.columns([1,4,1])
    with col1:
        st.button("🔄 Reset", help="Reiniciar aplicación")
    with col2:
        st.progress(100 if 'df' in locals() else 0)
    with col3:
        st.download_button(
            "💾 Descargar Dataset",
            data=df.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv"
        )

    # Sección de Análisis Exploratorio mejorada
    with st.expander("🔍 Análisis Exploratorio", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Datos", "Estadísticas", "Visualización"])
        
        with tab1:
            st.subheader("Datos Iniciales")
            st.dataframe(df.head(5).style.highlight_max(axis=0))
            
        with tab2:
            st.subheader("Resumen Estadístico")
            st.dataframe(df.describe(include='all').T.style.format("{:.2f}"))
            
        with tab3:
            st.subheader("Distribución de Variables")
            selected_col = st.selectbox("Selecciona variable", df.columns)
            if is_numeric_dtype(df[selected_col]):
                fig = px.histogram(df, x=selected_col, marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Seleccione una variable numérica")

    # Sección de Visualización Interactiva mejorada
    st.header("📈 Generador de Visualizaciones")
    with st.form("visual_config"):
        plot_type = st.selectbox(
            "Tipo de Gráfico",
            ["Histograma", "Scatter Plot", "Box Plot", "Bar Plot", "Pairplot"],
            help="Selecciona el tipo de visualización"
        )
        
        if plot_type in ["Histograma", "Box Plot"]:
            selected_col = st.selectbox(
                "Variable",
                df.select_dtypes(include=np.number).columns,
                help="Solo variables numéricas"
            )
            
        elif plot_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            x_col = col1.selectbox("Variable X", df.columns)
            y_col = col2.selectbox("Variable Y", df.columns)
            color_col = st.color_picker("Color de los puntos", "#2ecc71")
            size_col = st.slider("Tamaño de puntos", 5, 20, 10)
            
        elif plot_type == "Bar Plot":
            cat_cols = df.select_dtypes(include=['object']).columns
            num_cols = df.select_dtypes(include=np.number).columns
            cat_col = st.selectbox("Categórica", cat_cols)
            num_col = st.selectbox("Numérica", num_cols)
            
        elif plot_type == "Pairplot":
            cols = st.multiselect("Variables", df.columns)
            
        submit_button = st.form_submit_button("Generar Gráfico 📊")
        
    if submit_button:
        with st.spinner("Generando visualización..."):
            if plot_type == "Histograma":
                fig = px.histogram(df, x=selected_col, nbins=30)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "Scatter Plot":
                if is_numeric_dtype(df[x_col]) and is_numeric_dtype(df[y_col]):
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color_discrete_sequence=[color_col],
                        size=np.repeat(size_col, len(df))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Las variables X e Y deben ser numéricas")
                    
        # Agregar descarga de gráficos
        if 'fig' in locals():
            st.download_button(
                ".Download PNG",
                data=fig.to_image(format="png"),
                file_name=f"{plot_type.lower()}.png",
                mime="image/png"
            )

    # Resumen mejorado en sidebar
    with st.sidebar:
        st.subheader("📋 Resumen Dataset")
        st.metric("Filas", f"{len(df):,}")
        st.metric("Columnas", len(df.columns))
        st.metric("Memoria", f"{df.memory_usage().sum() / 1e6:.2f} MB")
        
        with st.expander("Variables"):
            for col in df.columns:
                st.text(f"• {col} ({df[col].dtype})")

else:
    st.info("📁 Por favor sube un dataset para comenzar")
    st.image("https://via.placeholder.com/600x400?text=Data+Analysis", use_column_width=True)
