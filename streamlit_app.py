import streamlit as st
from autogluon.tabular import TabularPredictor
import pandas as pd

# Función para cargar el modelo
def cargar_modelo(ruta):
    modelo = TabularPredictor.load(ruta)
    return modelo


# Función para realizar una predicción
def predecir_renta(modelo, entrada):
    predicciones = modelo.predict(pd.DataFrame([entrada]))
    return predicciones[0]


# Cargar el modelo entrenado
modelo = cargar_modelo("modelo_autogluon/")

# Crear la aplicación web con Streamlit
def main():
    st.title("Predicción de precios de alquiler")

    # Entrada de datos por parte del usuario
    city = st.text_input("Ciudad")
    area = st.number_input("Área (en mts2)", min_value=1)
    rooms = st.number_input("Número de habitaciones", min_value=1)
    bathroom = st.number_input("Número de baños", min_value=1)
    parking_spaces = st.number_input(
        "Número de espacios de estacionamiento", min_value=0
    )
    floor = st.number_input("Piso", min_value=1)
    animal = st.selectbox("Se permiten animales", ["yes", "no"])
    furniture = st.selectbox("Amueblado", ["furnished", "not furnished"])
    hoa = st.number_input(
        "Impuesto de la Asociación de Residentes (en R$)", min_value=0
    )
    property_tax = st.number_input(
        "Impuesto sobre bienes inmuebles (en R$)", min_value=0
    )
    fire_insurance = st.number_input("Seguro contra incendios (en R$)", min_value=0)

    # Al presionar el botón "Predecir", se realiza la predicción y se muestra el resultado
    if st.button("Predecir"):
        entrada = {
            "city": city,
            "area": area,
            "rooms": rooms,
            "bathroom": bathroom,
            "parking spaces": parking_spaces,
            "floor": floor,
            "animal": animal,
            "furniture": furniture,
            "hoa (R$)": hoa,
            "property tax (R$)": property_tax,
            "fire insurance (R$)": fire_insurance,
        }

        renta_predicha = predecir_renta(modelo, entrada)

        st.write(
            f"La renta mensual estimada para esta propiedad es de: R$ {renta_predicha}"
        )


if __name__ == "__main__":
    main()
