import streamlit as st
from autogluon.tabular import TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Cargar el modelo
def cargar_modelo(ruta):
    modelo = TabularPredictor.load(ruta)
    return modelo


def mostrar_importancia_caracteristicas(modelo):
    try:
        # Intenta obtener la importancia de las características sin un conjunto de datos de prueba
        importancia_caracteristicas = modelo.feature_importance(
            feature_stage="transformed"
        )
    except AssertionError:
        # Si se produce un error, informa al usuario y no muestres el gráfico
        st.write(
            "No se pudo obtener la importancia de las características sin un conjunto de datos de prueba."
        )
        return

    # Si se obtuvo con éxito, procede a crear y mostrar el gráfico
    importancia_caracteristicas.sort_values(
        by="importance", ascending=True, inplace=True
    )

    # Convertir el índice y las importancias a formato de DataFrame para Plotly
    df_importancia = importancia_caracteristicas.reset_index()

    # Crear un gráfico de barras horizontal interactivo con Plotly
    fig = px.bar(
        df_importancia,
        x="importance",
        y="index",  # Asume que 'index' es el nombre de la columna después de reset_index
        orientation="h",
        title="Importancia de las Características en la Predicción del Alquiler",
        labels={
            "index": "Características",
            "importance": "Importancia",
        },  # Renombrar las etiquetas
        color="importance",  # Color de las barras basado en la importancia
        color_continuous_scale=px.colors.sequential.Viridis[
            ::-1
        ],  # Escala de colores, invertida
    )

    # Ajustes estéticos adicionales
    fig.update_layout(
        xaxis_title="Importancia",
        yaxis_title="Características",
        coloraxis_showscale=False,  # Oculta la barra de colores
    )

    # Mostrar el gráfico en la página
    st.plotly_chart(fig)


def cargar_datos(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    return datos


# Función para realizar una predicción
def predecir_renta(modelo, entrada):
    predicciones = modelo.predict(pd.DataFrame([entrada]))
    return predicciones[0]


coordenadas_ciudades = {
    "Belo Horizonte": [-19.9167, -43.9345],
    "Campinas": [-22.9099, -47.0626],
    "Porto Alegre": [-30.0346, -51.2177],
    "Rio de Janeiro": [-22.9068, -43.1729],
    "São Paulo": [-23.5505, -46.6333],
    # ... (añadir más ciudades y sus coordenadas si es necesario)
}

# Cargar el modelo entrenado
modelo = cargar_modelo("AutogluonModels/ag-20231017_225025")

# Crear la aplicación web con Streamlit
def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Predictor de Precios de Alquiler",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Predicción de Precios de Alquiler")

    # Aquí puedes agregar una descripción o imagen
    st.write(
        "Por favor ingrese los detalles de la propiedad para la cual le gustaría predecir el precio de alquiler."
    )

    # Diseño usando columnas
    col1, col2 = st.columns(2)

    with col1:
        ciudad = st.text_input("Ciudad")
        area = st.number_input("Área (en m2)", min_value=1)
        habitaciones = st.number_input("Número de habitaciones", min_value=1)
        baños = st.number_input("Número de baños", min_value=1)
        espacios_estacionamiento = st.number_input(
            "Número de espacios de estacionamiento", min_value=0
        )

    with col2:
        piso = st.number_input("Piso", min_value=1)
        mascotas = st.selectbox("Se permiten mascotas", ["Sí", "No"])
        muebles = st.selectbox("Amueblado", ["Amueblado", "No amueblado"])
        hoa = st.number_input(
            "Impuesto de la Asociación de Propietarios (en R$)", min_value=0
        )
        impuesto_propiedad = st.number_input(
            "Impuesto sobre la propiedad (en R$)", min_value=0
        )
        seguro_incendio = st.number_input(
            "Seguro contra incendios (en R$)", min_value=0
        )

    # Botón de predecir
    if st.button("Predecir"):
        entrada = {
            "city": ciudad,
            "area": area,
            "rooms": habitaciones,
            "bathroom": baños,
            "parking spaces": espacios_estacionamiento,
            "floor": piso,
            "animal": "yes" if mascotas == "Sí" else "no",
            "furniture": "furnished" if muebles == "Amueblado" else "not furnished",
            "hoa (R$)": hoa,
            "property tax (R$)": impuesto_propiedad,
            "fire insurance (R$)": seguro_incendio,
        }

        renta_predicha = predecir_renta(modelo, entrada)

        # Formatear el resultado para mostrar solo dos decimales
        renta_predicha = "{:.2f}".format(renta_predicha)

        st.success(
            f"La renta mensual estimada para esta propiedad es de: R$ {renta_predicha}"
        )

    st.title("Visualizaciones")

    # Asumiendo que 'modelo' está definido y es un modelo entrenado
    with st.expander("Importancia de las Características"):
        mostrar_importancia_caracteristicas(modelo)

    st.title("Análisis de Tendencias de Alquiler en Brasil")

    # Cargando los datos
    datos = cargar_datos("houses_to_rent_v2.csv")

    # Calculando el promedio de alquiler por ciudad
    promedio_ciudad = datos.groupby("city")["rent amount (R$)"].mean().reset_index()
    promedio_ciudad = promedio_ciudad.rename(
        columns={"rent amount (R$)": "promedio_alquiler"}
    )

    # Preparando los datos para el mapa
    for ciudad in promedio_ciudad["city"]:
        if ciudad in coordenadas_ciudades:
            promedio_ciudad.loc[
                promedio_ciudad["city"] == ciudad, "lat"
            ] = coordenadas_ciudades[ciudad][0]
            promedio_ciudad.loc[
                promedio_ciudad["city"] == ciudad, "lon"
            ] = coordenadas_ciudades[ciudad][1]

    # Sección de visualización de tendencias de alquiler
    with st.expander("Tendencias de Alquiler en Brasil"):
        fig = go.Figure()

        for i, row in promedio_ciudad.iterrows():
            if row["city"] in coordenadas_ciudades:
                fig.add_trace(
                    go.Scattergeo(
                        locationmode="country names",
                        lon=[row["lon"]],
                        lat=[row["lat"]],
                        text=f"{row['city']}: {row['promedio_alquiler']:.2f} R$",
                        marker=dict(
                            size=10,
                            color="red",
                            line=dict(width=3, color="rgba(68, 68, 68, 0)"),
                        ),
                        name=row["city"],
                    )
                )

        fig.update_layout(
            title="Promedio de Alquiler por Ciudad en Brasil",
            geo=dict(
                scope="south america",
                projection_type="mercator",
                showland=True,
                landcolor="rgb(250, 250, 250)",
                countrycolor="rgb(204, 204, 204)",
            ),
        )

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
