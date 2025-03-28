import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from scipy import stats
import io
import statsmodels.api as sm  # Agregado para regresión lineal

# Configuración inicial
st.set_page_config(
    page_title="Sistema de Soporte a Decisiones",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.filtered_df = None

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
    
    # Filtrado por columnas categóricas
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col in st.session_state:
            if len(st.session_state[col]) > 0:
                filtered_df = filtered_df[filtered_df[col].isin(st.session_state[col])]
    
    # Filtrado por columnas numéricas
    for col in df.select_dtypes(include=np.number).columns:
        if f"{col}_min" in st.session_state and f"{col}_max" in st.session_state:
            current_min = st.session_state[f"{col}_min"]
            current_max = st.session_state[f"{col}_max"]
            filtered_df = filtered_df[(filtered_df[col] >= current_min) & (filtered_df[col] <= current_max)]
    
    return filtered_df

# Carga de datos
st.sidebar.title("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu dataset", type=['csv', 'xls', 'xlsx', 'json'])

if uploaded_file is not None:
    st.session_state.df = load_data(uploaded_file)
    st.session_state.filtered_df = st.session_state.df.copy()

    # Sección de Filtrado
    with st.sidebar.expander("Filtrado de Datos", expanded=False):
        if st.session_state.df is not None:
            # Filtrado categórico
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                options = st.session_state.df[col].unique().tolist()
                selected = st.multiselect(
                    f"Filtrar por {col}",
                    options=options,
                    default=options,
                    key=col
                )
            
            # Filtrado numérico con enteros y entrada manual
            num_cols = st.session_state.df.select_dtypes(include=np.number).columns
            for col in num_cols:
                min_val = int(st.session_state.df[col].min())
                max_val = int(st.session_state.df[col].max())
                
                # Slider con enteros
                current_range = st.slider(
                    f"Rango para {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key=f"{col}_slider"
                )
                
                # Entrada manual
                col1, col2 = st.columns(2)
                input_min = col1.number_input(f"Min {col}", value=current_range[0], step=1)
                input_max = col2.number_input(f"Max {col}", value=current_range[1], step=1)
                
                # Sincronización entre slider y entradas
                st.session_state[f"{col}_min"] = max(min(input_min, input_max), min_val)
                st.session_state[f"{col}_max"] = min(max(input_max, input_min), max_val)

    # Aplicar filtros
    if st.sidebar.button("Aplicar Filtros"):
        st.session_state.filtered_df = apply_filters(st.session_state.df)
        st.success("Filtros aplicados con éxito!")

    # Mostrar datos filtrados
    st.sidebar.write(f"Datos filtrados: {len(st.session_state.filtered_df)} registros")

    # Sección de Estadísticas Avanzadas
    st.sidebar.header("Estadísticas Avanzadas")
    stat_option = st.sidebar.selectbox(
        "Seleccione análisis",
        ["", "Prueba t", "ANOVA", "Regresión Lineal", "Chi-cuadrado"]
    )

    if stat_option:
        df = st.session_state.filtered_df
        st.header(f"Análisis: {stat_option}")
        
        if stat_option == "Prueba t":
            col1, col2 = st.columns(2)
            group_col = col1.selectbox("Variable de agrupación", df.select_dtypes(include=['object']).columns)
            num_col = col2.selectbox("Variable numérica", df.select_dtypes(include=np.number).columns)
            
            # Validación de grupos
            groups = df[group_col].dropna().unique()
            if len(groups) != 2:
                st.error("La prueba t requiere exactamente 2 grupos")
            else:
                group1 = df[df[group_col] == groups[0]][num_col].dropna()
                group2 = df[df[group_col] == groups[1]][num_col].dropna()
                
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    st.write(f"t-statistic: {t_stat:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                else:
                    st.error("Cada grupo debe tener al menos 2 observaciones")

        elif stat_option == "ANOVA":
            cat_col = st.selectbox("Variable categórica", df.select_dtypes(include=['object']).columns)
            num_col = st.selectbox("Variable numérica", df.select_dtypes(include=np.number).columns)
            
            # Validación de grupos
            groups = [df[df[cat_col] == val][num_col].dropna() for val in df[cat_col].dropna().unique()]
            if len(groups) < 2:
                st.error("ANOVA requiere al menos 2 grupos")
            else:
                valid_groups = [g for g in groups if len(g) > 1]
                if len(valid_groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*valid_groups)
                    st.write(f"F-statistic: {f_stat:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                else:
                    st.error("Cada grupo debe tener al menos 2 observaciones")

        elif stat_option == "Regresión Lineal":
            y_col = st.selectbox("Variable dependiente", df.select_dtypes(include=np.number).columns)
            X_cols = st.multiselect("Variables independientes", df.select_dtypes(include=np.number).columns)
            
            if X_cols:
                X = df[X_cols].dropna()
                y = df[y_col].dropna()
                
                if len(X) == len(y) and len(X) > 0:
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    st.write(model.summary())
                else:
                    st.error("Las variables deben tener observaciones coincidentes")

        elif stat_option == "Chi-cuadrado":
            cat_col1 = st.selectbox("Variable categórica 1", df.select_dtypes(include=['object']).columns)
            cat_col2 = st.selectbox("Variable categórica 2", df.select_dtypes(include=['object']).columns)
            
            # Crear tabla de contingencia
            valid_df = df[[cat_col1, cat_col2]].dropna()
            if len(valid_df) > 0:
                contingency_table = pd.crosstab(valid_df[cat_col1], valid_df[cat_col2])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f"Chi-cuadrado: {chi2:.4f}")
                st.write(f"p-value: {p:.4f}")
            else:
                st.error("No hay datos válidos para la prueba")

    # Sección de Visualización
    st.sidebar.header("Visualización Interactiva")
    plot_type = st.sidebar.selectbox("Tipo de Gráfico", 
                                    ["Histograma", "Scatter Plot", 
                                     "Box Plot", "Bar Plot", 
                                     "Heatmap", "Pairplot"])

    if plot_type:
        df = st.session_state.filtered_df
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_vars = st.multiselect("Selecciona variables", numeric_cols, default=numeric_cols)
        elif plot_type == "Pairplot":
            cols = st.multiselect("Selecciona variables", df.columns)
        
        # Generación de gráficos
        try:
            if plot_type == "Histograma":
                if is_numeric_dtype(df[selected_col]):
                    fig = px.histogram(df, x=selected_col, marginal="box", nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Exportar con validación
                    buf = io.BytesIO()
                    fig.write_image(buf, format="png", engine="kaleido")
                    st.download_button(
                        "Descargar gráfico",
                        data=buf.getvalue(),
                        file_name="histogram.png",
                        mime="image/png"
                    )

            # ... (otros casos similares con bloques try-except)

        except Exception as e:
            st.error(f"Error al generar gráfico: {str(e)}")

    # Exportar informe (código existente)

else:
    st.info("Por favor, sube un dataset para comenzar")
