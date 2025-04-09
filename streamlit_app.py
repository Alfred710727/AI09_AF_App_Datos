import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# Configuración inicial
st.set_page_config(
    page_title="Aplicación Dinámica de Soporte a Decisiones",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Inyección de CSS mejorado
st.markdown(
    """
    <style>
    .main {
        background-image: url('./gou_imagen.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), url('./gou_imagen.png');
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 8px;
        font-weight: bold;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 4px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 150%; /* Posición sobre el texto */
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.4em;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Carga de datos
@st.cache_data(show_spinner="Cargando datos... 🔄")
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Formato no soportado ❌")
        return None

# Estado para controlar la vista README
if 'show_readme' not in st.session_state:
    st.session_state.show_readme = False

# Función para mostrar el README
def show_readme():
    st.session_state.show_readme = True

# Función para ocultar el README
def hide_readme():
    st.session_state.show_readme = False

# Botón README siempre visible
with st.sidebar:
    #st.button("README", on_click=show_readme)
    st.button(
        "README 📖",
        on_click=lambda: setattr(st.session_state, 'show_readme', True),
        help="Ver guía de uso"
    )

# Barra lateral
st.sidebar.title("📊 Carga de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu dataset",
    type=['csv', 'xls', 'xlsx', 'json'],
    help="Soporta CSV, Excel y JSON"
    )

if not st.session_state.show_readme:
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Agrega la imagen en la parte superior
        col1, col2 = st.columns([3, 1])
        with col2:
            st.image("gou_imagen.png", width=200, caption="")

        # Sección de Análisis Exploratorio
        st.sidebar.header("🔍 Análisis Exploratorio")
        show_eda = st.sidebar.checkbox("Mostrar Análisis Exploratorio")

        if show_eda:

            # Sección de Análisis Exploratorio de Datos
            with st.expander("🔍 Análisis Exploratorio de Datos", expanded=True):
                tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Estadísticas", "Correlación","Valores Faltantes"])

                with tab1:
                    st.subheader("🗃️ Datos Iniciales")
                    st.dataframe(df.head(5).style.highlight_max(axis=0))

                with tab2:
                    st.subheader("🏦 Resumen Estadístico - Estadísticas Descriptivas")
                    # Filtrar columnas numéricas
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if not numeric_cols.empty:
                        # Calcular estadísticas solo para columnas numéricas
                        stats_df = df[numeric_cols].describe().T
                        # Aplicar formato solo a columnas numéricas
                        st.dataframe(stats_df.style.format("{:.2f}"))
                    else:
                        st.warning("⚠️ No se encontraron columnas numéricas en el dataset.")

                with tab3:
                    st.subheader("🧩 Correlación entre de Variables Numéricas")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        st.subheader("Matriz de Correlación")
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, text_auto=True)
                        st.plotly_chart(fig, use_container_width=True)

                with tab4:
                    st.subheader("Valores Faltantes")
                    missing_values = df.isnull().sum()
                    st.bar_chart(missing_values[missing_values > 0])


        # Sección de Visualización
        st.sidebar.header("📈 Visualización Interactiva")

        # Checkbox para habilitar gráficos avanzados
        enable_advanced = st.sidebar.checkbox("🔬 Habilitar Gráficos Avanzados")

        # Filtrar opciones de gráfico según estado de EDA
        traditional_plots = ["📊 Histograma", "🔗 Scatter Plot", "📦 Box Plot", "📊 Bar Plot", "📊 + 📊 Pairplot"]

        if not show_eda:
            traditional_plots.append("🌡️ Heatmap")

        advanced_plots = ["📈 KDE", "🎻 Violin Plot", "⬢ Hexbin", "3D 🚀 Scatter", "🔄 ParallelGroups", "🧬 ClusterMap"]

        if enable_advanced:
            advanced_plot = st.sidebar.selectbox(
                "🔬 Gráficos Avanzados",
                advanced_plots,
                key="advanced"
            )
            plot_type = advanced_plot  # Tomar el valor del selectbox activo
        else:
            traditional_plot = st.sidebar.selectbox(
                "📊 Gráficos Tradicionales",
                traditional_plots,
                key="traditional"
            )
            plot_type = traditional_plot  # Tomar el valor del selectbox activo

        # Diccionario con descripciones de gráficos
        PLOT_DESCRIPTIONS = {
            "📊 Histograma": "Muestra la distribución de una variable numérica mediante barras que representan frecuencias en intervalos. Ideal para identificar sesgos, curtosis y valores atípicos.",
            "🔗 Scatter Plot": "Visualiza la relación entre dos variables numéricas mediante puntos en un plano cartesiano. Útil para detectar correlaciones y patrones no lineales.",
            "📦 Box Plot": "Representa estadísticas descriptivas (mediana, cuartiles y outliers) de una variable. Esencial para comparar distribuciones entre grupos.",
            "📊 Bar Plot": "Compara magnitudes entre categorías mediante barras. Recomendado para variables categóricas y análisis de frecuencias relativas.",
            "📈 KDE": "Estima la densidad de probabilidad de una variable numérica mediante suavizado de kernel. Alternativa más precisa que el histograma para distribuciones continuas.",
            "📊 + 📊 Pairplot": "Muestra relaciones multivariadas mediante una matriz de scatter plots. Ideal para explorar correlaciones en datasets con múltiples variables numéricas.",
            "🌡️ Heatmap": "Visualiza matrices de datos (como correlaciones) mediante colores. Permite identificar patrones de asociación rápida y eficientemente.",
            "🎻 Violin Plot": "Combina un box plot con una estimación de densidad kernel. Muestra la distribución de datos en múltiples categorías con mayor detalle que el box plot tradicional.",
            "⬢ Hexbin": "Agrupa puntos densos en hexágonos para visualizar patrones en grandes datasets. Alternativa al scatter plot cuando hay sobreposición de datos.",
            "3D 🚀 Scatter": "Representa tres variables numéricas en un espacio tridimensional. Útil para explorar interacciones complejas entre múltiples dimensiones.",
            "🔄 ParallelGroups": "Visualiza datos multidimensionales mediante ejes paralelos. Permite identificar clusters y patrones en variables numéricas y categóricas simultáneamente.",
            "🧬 ClusterMap": "Aplica clustering jerárquico a filas y columnas de un dataset. Muestra grupos similares mediante un heatmap con dendrogramas, ideal para análisis de expresión génica o segmentación."
        }

        if plot_type:
            #st.header(f"{plot_type} Interactivo")
            # Título con tooltip
            st.markdown(f"""
                <h2 style='display: inline-block;'>
                    {plot_type} Interactivo
                    <span class='tooltip'>
                        ❓
                        <span class='tooltiptext'>
                            {PLOT_DESCRIPTIONS.get(plot_type, 'Sin descripción disponible')}
                        </span>
                    </span>
                </h2>
            """, unsafe_allow_html=True)

            # Selección de variables
            if plot_type in ["📊 Histograma", "📦 Box Plot"]:
                selected_col = st.selectbox("Selecciona una variable", df.columns)
            elif plot_type == "🔗 Scatter Plot":
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

            elif plot_type == "📊 Bar Plot":
                # Validar existencia de columnas categóricas y numéricas
                cat_cols_available = list(df.select_dtypes(include=['object']).columns)
                num_cols_available = list(df.select_dtypes(include=np.number).columns)

                if not cat_cols_available or not num_cols_available:
                    st.warning("El dataset no contiene columnas categóricas y/o numéricas necesarias")
                else:
                    cat_col = st.selectbox("Variable Categórica", cat_cols_available)
                    num_col = st.selectbox("Variable Numérica", num_cols_available)

            elif plot_type == "🌡️ Heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns

            elif plot_type == "📊 + 📊 Pairplot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cols = st.multiselect("Selecciona variables", numeric_cols, placeholder="Seleccionar variables")

            # Generación de gráficos
            if plot_type == "📊 Histograma":
                if is_numeric_dtype(df[selected_col]):
                    fig = px.histogram(df, x=selected_col, marginal="box", nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("La variable seleccionada no es numérica")

            elif plot_type == "🔗 Scatter Plot":
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

            elif plot_type == "📦 Box Plot":
                fig = px.box(df, y=selected_col)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "📊 Bar Plot":
                # Validación doble
                if 'cat_col' not in locals() or 'num_col' not in locals():
                    st.warning("Seleccione variables válidas")
                elif cat_col not in df.columns or num_col not in df.columns:
                    st.warning("Las variables seleccionadas no existen en el dataset")
                else:
                    counts = df.groupby(cat_col)[num_col].mean().reset_index()
                    fig = px.bar(counts, x=cat_col, y=num_col)
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "🌡️ Heatmap" and not show_eda:
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes columnas numéricas para generar el heatmap")

            elif plot_type == "📊 + 📊 Pairplot":
                if cols:
                    pair_df = df[cols]
                    fig = sns.pairplot(pair_df)
                    st.pyplot(fig)

            # --- Nuevos Gráficos ---
            if plot_type == "📈 KDE":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No hay columnas numéricas disponibles")
                else:
                    selected_col = st.selectbox("Variable Numérica", numeric_cols)
                    if selected_col:
                        fig, ax = plt.subplots()
                        sns.kdeplot(df[selected_col], fill=True, ax=ax)
                        st.pyplot(fig)
                        
            elif plot_type == "🎻 Violin Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                if not numeric_cols:
                    st.warning("No hay columnas numéricas")
                else:
                    num_var = st.selectbox("Variable Numérica", numeric_cols)
                    cat_var = st.selectbox("Variable Categórica (opcional)", ["Ninguna"] + cat_cols)
                    if cat_var != "Ninguna":
                        fig, ax = plt.subplots()
                        sns.violinplot(x=df[cat_var], y=df[num_var], ax=ax)
                        st.pyplot(fig)
                        
            elif plot_type == "⬢ Hexbin":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    st.warning("Se necesitan al menos 2 columnas numéricas")
                else:
                    x_col = st.selectbox("Eje X", numeric_cols)
                    y_col = st.selectbox("Eje Y", numeric_cols)
                    if x_col and y_col:
                        fig, ax = plt.subplots()
                        hb = ax.hexbin(df[x_col], df[y_col], gridsize=20, cmap='viridis')
                        fig.colorbar(hb, label='Densidad')
                        st.pyplot(fig)
                        
            elif plot_type == "3D 🚀 Scatter":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 3:
                    st.warning("Se necesitan al menos 3 columnas numéricas")
                else:
                    x_col = st.selectbox("Eje X", numeric_cols)
                    y_col = st.selectbox("Eje Y", numeric_cols)
                    z_col = st.selectbox("Eje Z", numeric_cols)
                    if x_col and y_col and z_col:
                        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col)
                        st.plotly_chart(fig)
                        
            elif plot_type == "🔄 ParallelGroups":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                if not numeric_cols or not cat_cols:
                    st.warning("Se necesita al menos 1 variable numérica y 1 categórica")
                else:
                    class_col = st.selectbox("Variable Clase", cat_cols)
                    features = st.multiselect("Variables Numéricas", numeric_cols, placeholder="Seleccionar variables")
                    if features and class_col:
                        from sklearn.preprocessing import StandardScaler, LabelEncoder
                        # Escalar variables numéricas
                        scaled_df = StandardScaler().fit_transform(df[features])
                        scaled_df = pd.DataFrame(scaled_df, columns=features)
                        
                        # Convertir variable categórica a numérica
                        le = LabelEncoder()
                        class_codes = le.fit_transform(df[class_col])
                        
                        # Agregar codificación al DataFrame escalado
                        scaled_df[class_col] = class_codes
                        
                        # Crear gráfico con codificación de color numérica
                        fig = px.parallel_coordinates(
                            scaled_df,
                            color=class_col,
                            labels={class_col: "Clase"},
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        st.plotly_chart(fig)
                        
            elif plot_type == "🧬 ClusterMap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    st.warning("Se necesitan al menos 2 columnas numéricas")
                else:
                    cluster_vars = st.multiselect("Variables para Clustering", numeric_cols, placeholder="Seleccionar al menos dos variables")
                    if cluster_vars:
                        try:
                            max_rows = 5000
                            if len(df) > max_rows:
                                st.warning(
                                    f"Dataset muy grande ({len(df):,} filas). "
                                    f"Usando muestra aleatoria de {max_rows} filas."
                                )
                                cluster_data = df[cluster_vars].sample(max_rows).astype(np.float32)
                            else:
                                cluster_data = df[cluster_vars].astype(np.float32)
                            
                            g = sns.clustermap(
                                cluster_data,
                                cmap="vlag",
                                figsize=(10, 8),
                                dendrogram_ratio=(0.1, 0.1),
                                cbar_pos=(0.02, 0.8, 0.05, 0.18)
                            )
                            st.pyplot(g.fig)

                        except ValueError as ve:
                            if "The number of observations cannot be determined on an empty distance matrix" in str(ve):
                                st.error("❌ Error en clustering: Selecciona por lo menos dos variables")
                            else:
                                raise ve
            
                        except Exception as e:
                            st.error(
                                f"❌ Error en clustering: {str(e)}\n"
                                "\n Intente reducir el número de variables o usar un subconjunto de datos."
                            )

        # Exportar informe
        if 'mostrar_resumen' not in st.session_state:
            st.session_state.mostrar_resumen = False

        # Resumen mejorado en sidebar
        with st.sidebar:
            st.subheader("📋 Dataset")
            with st.expander("🔍 Resumen"):
                st.metric("Filas", f"{len(df):,}")
                st.metric("Columnas", len(df.columns))
                st.metric("Memoria", f"{df.memory_usage().sum() / 1e6:.2f} MB")

            with st.expander("🔢 Variables"):
                for col in df.columns:
                    st.text(f"• {col} ({df[col].dtype})")

    else:
        st.info("📁 Por favor, sube un dataset para comenzar")
else:
    # Contenido del README
    st.header("📚 Guía de Uso de la Aplicación")
    st.markdown("""
    **Pasos para utilizar:**
    1. 📁 Carga tu dataset mediante el botón **Browse files** en la sección lateral (Parte Izquierda de la pantalla)
    2. 📊 Activa ✅ **Análisis Exploratorio** para ver estadísticas básicas y visualizaciones
    3. 📈 Selecciona el tipo de gráfico deseado en la sección de Visualización Interactiva
    4. 🔬 Activa ✅ **Habilitar Gráficos Avanzados** para análizar gráficos avanzados en la sección de Visualización Interactiva
    5. 🎨 Personaliza los parámetros del gráfico como colores, variables y dimensiones
    6. 📉 Explora, visualiza y analiza los resultados generados con gráficos interactivos y descárgalos en formato PNG/SVG

    **Tips:**
    - Genera análisis estadístico y matriz de correlación con un sólo clic
    - Los gráficos se pueden descargar con clic derecho
    - Elementos de ayuda (❓) para entender cada gráfico, con lo sólo pasar el mouse por encima.
    """)

    # Botón para salir del README
    st.button("⬅️ Regresar", on_click=lambda: setattr(st.session_state, 'show_readme', False),
              help="Inicia análisis exploratorio")
