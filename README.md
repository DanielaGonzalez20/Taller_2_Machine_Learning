# Proyecto ML: Premier League Analytics 

## 1. Integrantes
- Daniela Gonzalez Ortiz
- Belén Shalom Dueñas Pacheco

## 2. URL del Dashboard Desplegado 
 **[https://ml-premier-league-jaxujexnfjcpm6fp3vw3qv.streamlit.app/]**

## 3. Descripción Breve del *Approach* y *Features* Utilizadas 
Este proyecto implementa un pipeline analítico y predictivo enfocado en los remates y eventos de fútbol.

**Approach Metodológico:**
* Desarrollamos un extenso Análisis Exploratorio de Datos (EDA) espacial y temporal, aislando correlaciones directas entre el instante del partido (ej. Minutos 90+), el nivel de presión defensiva y la tasa de conversión a gol.
* Entrenamos modelos de *Machine Learning* Descriptivo y Predictivo abarcando tanto Algoritmos Supervisados (Regresión Logística, Random Forest, Gradient Boosting para cálculo de "Expected Goals") como No Supervisados (Clustering K-Means para agrupar comportamientos de remate).
* Para moderar la varianza generada por eventos fortuitos en el fútbol, aplicamos técnicas algorítmicas de regularización (Ridge - L2) en la inferencia de resultados globales del partido, demostrando mitigación del *overfitting*.

**Features Principales Utilizadas:**
* **Espaciales:** `Distance` (Distancia al arco), `Angle` (Apertura angular hacia la portería).
* **Contextuales:** `Minute` (Minuto del remate), `Pressure_Index` (Índice de aglomeración defensiva), `Is_Fast_Break` (Contraataque rápido).
* **Binarias/Categóricas:** `Is_Big_Chance` (Ocasión inmejorable), `Is_Header` (Remate de cabeza).

## 4. Instrucciones para Ejecutar el Notebook y Dashboard localmente 

**Paso 1: Preparación del Entorno**
Asegúrese de clonar el repositorio, abrir una terminal en la misma ruta y ejecutar los requerimientos pertinentes:
```bash
pip install -r requirements.txt
```

**Paso 2: Ejecución Completa de Análisis (Jupyter Notebook)**
Para acceder al análisis estático y metodológico a detalle:
```bash
jupyter notebook ML_taller.ipynb
```
*(Abra el archivo desde la interfaz en su navegador web)*.

**Paso 3: Lanzar el Dashboard Visual (Streamlit)**
Para experimentar con las interfaces, mapas interactivos 3D y simuladores tácticos:
```bash
streamlit run app.py
```
*(Esto ejecutará instantáneamente un servidor local direccionado a `http://localhost:8501`).*
