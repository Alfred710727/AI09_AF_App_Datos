import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from scipy import stats
import io
from PIL import Image

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
            
            # Filtrado numérico
            num_cols = st.session_state.df.select_dtypes(include=np.number).columns
            for col in num_cols:
                min_val = float(st.session_state.df[col].min())
                max_val = float(st.session_state.df[col].max())
                current_range = st.slider(
                    f"Rango para {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"{col}_slider"
                )
                st.session_state[f"{col}_min"] = current_range[0]
                st.session_state[f"{col}_max"] = current_range[1]

    # Aplicar filtros
    if st.sidebar.button("Aplicar Filtros"):
        st.session_state.filtered_df = apply_filters(st.session_state.df)
        st.success("Filtros aplicados con éxito!")

    # Mostrar datos filtrados
    st.sidebar.write(f"Datos filtrados: {len(st.session_state.filtered_df)} registros")

    # Sección de Análisis Exploratorio
    st.sidebar.header("Análisis Exploratorio")
    show_eda = st.sidebar.checkbox("Mostrar Análisis Exploratorio")

    if show_eda:
        df = st.session_state.filtered_df
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
            st.download_button(
                "Descargar matriz de correlación",
                data=corr_matrix.to_csv().encode('utf-8'),
                file_name='correlation_matrix.csv'
            )

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
            
            groups = df[group_col].unique()
            if len(groups) == 2:
                group1 = df[df[group_col] == groups[0]][num_col]
                group2 = df[df[group_col] == groups[1]][num_col]
                t_stat, p_value = stats.ttest_ind(group1, group2)
                st.write(f"t-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")
            else:
                st.warning("La prueba t requiere exactamente 2 grupos")

        elif stat_option == "ANOVA":
            cat_col = st.selectbox("Variable categórica", df.select_dtypes(include=['object']).columns)
            num_col = st.selectbox("Variable numérica", df.select_dtypes(include=np.number).columns)
            
            groups = [df[df[cat_col] == val][num_col] for val in df[cat_col].unique()]
            f_stat, p_value = stats.f_oneway(*groups)
            st.write(f"F-statistic: {f_stat:.4f}")
            st.write(f"p-value: {p_value:.4f}")

        elif stat_option == "Regresión Lineal":
            y_col = st.selectbox("Variable dependiente", df.select_dtypes(include=np.number).columns)
            X_cols = st.multiselect("Variables independientes", df.select_dtypes(include=np.number).columns)
            
            if X_cols:
                X = df[X_cols]
                y = df[y_col]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                st.write(model.summary())

        elif stat_option == "Chi-cuadrado":
            cat_col1 = st.selectbox("Variable categórica 1", df.select_dtypes(include=['object']).columns)
            cat_col2 = st.selectbox("Variable categórica 2", df.select_dtypes(include=['object']).columns)
            
            contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-cuadrado: {chi2:.4f}")
            st.write(f"p-value: {p:.4f}")

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
        if plot_type == "Histograma":
            if is_numeric_dtype(df[selected_col]):
                fig = px.histogram(df, x=selected_col, marginal="box", nbins=30)
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportar visualización
                buf = io.BytesIO()
                fig.write_image(buf, format="png")
                st.download_button(
                    "Descargar gráfico",
                    data=buf.getvalue(),
                    file_name="histogram.png",
                    mime="image/png"
                )
            else:
                st.warning("La variable seleccionada no es numérica")

        elif plot_type == "Scatter Plot":
            if color_col == "Ninguna":
                fig = px.scatter(df, x=x_col, y=y_col)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            
            st.plotly_chart(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button(
                "Descargar gráfico",
                data=buf.getvalue(),
                file_name="scatter.png",
                mime="image/png"
            )

        elif plot_type == "Box Plot":
            fig = px.box(df, y=selected_col)
            st.plotly_chart(fig, use_container_width=True)
            
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button(
                "Descargar gráfico",
                data=buf.getvalue(),
                file_name="boxplot.png",
                mime="image/png"
            )

        elif plot_type == "Bar Plot":
            counts = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(counts, x=cat_col, y=num_col)
            st.plotly_chart(fig, use_container_width=True)
            
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button(
                "Descargar gráfico",
                data=buf.getvalue(),
                file_name="barplot.png",
                mime="image/png"
            )

        elif plot_type == "Heatmap":
            if len(selected_vars) > 1:
                corr_matrix = df[selected_vars].corr()
                fig = px.imshow(corr_matrix, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportar matriz
                st.download_button(
                    "Descargar matriz",
                    data=corr_matrix.to_csv().encode('utf-8'),
                    file_name='heatmap_corr.csv'
                )

        elif plot_type == "Pairplot":
            if len(cols) > 1:
                pair_df = df[cols]
                fig = sns.pairplot(pair_df)
                st.pyplot(fig)
                
                # Convertir a imagen
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                st.download_button(
                    "Descargar gráfico",
                    data=buf.getvalue(),
                    file_name="pairplot.png",
                    mime="image/png"
                )

    # Exportar informe
    st.sidebar.header("Exportar Resultados")
    if st.sidebar.button("Generar Reporte"):
        st.sidebar.markdown("### Resumen del Análisis")
        st.sidebar.write(f"Dataset: {uploaded_file.name}")
        st.sidebar.write(f"Registros originales: {len(st.session_state.df)}")
        st.sidebar.write(f"Registros filtrados: {len(st.session_state.filtered_df)}")
        
        # Crear reporte en buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            st.session_state.df.describe().to_excel(writer, sheet_name='Estadísticas Originales')
            st.session_state.filtered_df.describe().to_excel(writer, sheet_name='Estadísticas Filtradas')
            st.session_state.filtered_df.to_excel(writer, sheet_name='Datos Filtrados')
        
        st.sidebar.download_button(
            "Descargar Reporte Completo",
            data=buffer.getvalue(),
            file_name='reporte_analisis.xlsx'
        )

else:
    st.info("Por favor, sube un dataset para comenzar")
