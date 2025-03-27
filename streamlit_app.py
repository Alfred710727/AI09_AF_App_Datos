import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from scipy import stats
import io
import statsmodels.api as sm

# Configuración inicial
st.set_page_config(
    page_title="Sistema de Soporte a Decisiones",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
    st.session_state.filtered_df = None
    st.session_state.filters = {}

# Funciones auxiliares
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

def apply_filters(df):
    filtered_df = df.copy()
    
    # Filtrado categórico
    for col in filtered_df.select_dtypes(include=['object', 'category']).columns:
        if col in st.session_state.filters:
            if len(st.session_state.filters[col]) > 0:
                filtered_df = filtered_df[filtered_df[col].isin(st.session_state.filters[col])]
    
    # Filtrado numérico
    for col in filtered_df.select_dtypes(include=np.number).columns:
        if f"{col}_min" in st.session_state.filters and f"{col}_max" in st.session_state.filters:
            current_min = st.session_state.filters[f"{col}_min"]
            current_max = st.session_state.filters[f"{col}_max"]
            filtered_df = filtered_df[(filtered_df[col] >= current_min) & (filtered_df[col] <= current_max)]
    
    return filtered_df

# Carga de datos
st.sidebar.title("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu dataset", type=['csv', 'xls', 'xlsx', 'json'])

if uploaded_file is not None:
    # Cargar datos originales
    if st.session_state.original_df is None:
        st.session_state.original_df = load_data(uploaded_file)
        st.session_state.filtered_df = st.session_state.original_df.copy()
        
        # Inicializar filtros
        for col in st.session_state.original_df.select_dtypes(include=['object', 'category']).columns:
            st.session_state.filters[col] = st.session_state.original_df[col].unique().tolist()
        
        for col in st.session_state.original_df.select_dtypes(include=np.number).columns:
            st.session_state.filters[f"{col}_min"] = float(st.session_state.original_df[col].min())
            st.session_state.filters[f"{col}_max"] = float(st.session_state.original_df[col].max())

    # Mostrar estadísticas persistentes
    with st.sidebar.expander("Resumen del Dataset", expanded=True):
        st.write(f"Registros originales: {len(st.session_state.original_df)}")
        st.write(f"Registros filtrados: {len(st.session_state.filtered_df)}")
        st.write(f"Columnas: {len(st.session_state.original_df.columns)}")
        st.write("Tipos de datos:")
        st.write(st.session_state.original_df.dtypes.astype(str))

    # Sección de Filtrado
    with st.sidebar.expander("Filtrado de Datos", expanded=False):
        # Filtrado categórico
        cat_cols = st.session_state.original_df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            options = st.session_state.original_df[col].unique().tolist()
            selected = st.multiselect(
                f"Filtrar por {col}",
                options=options,
                default=st.session_state.filters.get(col, options),
                key=f"filter_{col}"
            )
            st.session_state.filters[col] = selected
        
        # Filtrado numérico
        num_cols = st.session_state.original_df.select_dtypes(include=np.number).columns
        for col in num_cols:
            min_val = float(st.session_state.original_df[col].min())
            max_val = float(st.session_state.original_df[col].max())
            
            # Slider con entrada manual
            col1, col2 = st.columns(2)
            current_min = col1.number_input(
                f"Min {col}",
                min_value=min_val,
                max_value=max_val,
                value=st.session_state.filters.get(f"{col}_min", min_val),
                step=0.1,
                key=f"num_min_{col}"
            )
            current_max = col2.number_input(
                f"Max {col}",
                min_value=min_val,
                max_value=max_val,
                value=st.session_state.filters.get(f"{col}_max", max_val),
                step=0.1,
                key=f"num_max_{col}"
            )
            st.session_state.filters[f"{col}_min"] = current_min
            st.session_state.filters[f"{col}_max"] = current_max

        # Botones de acción
        col1, col2 = st.columns(2)
        if col1.button("Aplicar Filtros"):
            st.session_state.filtered_df = apply_filters(st.session_state.original_df)
            st.success("Filtros aplicados con éxito!")
        
        if col2.button("Reiniciar Filtros"):
            st.session_state.filters = {}
            st.session_state.filtered_df = st.session_state.original_df.copy()
            st.experimental_rerun()

    # Resto del código (EDA, estadísticas avanzadas, visualización, etc.)
    # ... [MANTENER EL CÓDIGO EXISTENTE DE LAS OTRAS SECCIONES] ...

else:
    st.info("Por favor, sube un dataset para comenzar")
