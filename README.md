# TFM Bayern Leverkusen temporada 2023/2024

# Análisis Táctico del Bayer Leverkusen - TFM

Este repositorio contiene el código fuente utilizado en el Trabajo Final de Máster para el análisis táctico del Bayer Leverkusen en la Bundesliga 2023/2024. Incluye herramientas para el procesamiento de datos de StatsBomb, visualizaciones con Matplotlib y una interfaz interactiva construida en Streamlit.

## Repositorio

Repositorio GitHub: https://github.com/blopezma/bayernleverksuen_tfm

## Estructura del Proyecto

```
.
├── data/                    # Datos de entrada (.csv)
├── figs/                    # Salidas visuales (figuras PNG)
├── streamlit/               # Aplicaciones interactivas
├── src/                     # Módulos de procesamiento
│   ├── red_pases.py         # Código para redes de pase
│   ├── freeze_frame.py      # Análisis freeze frame
│   ├── kmeans_formacion.py  # Formaciones automáticas
│   └── ...
├── .env                    # NO INCLUIDO por seguridad
├── requirements.txt        # Dependencias
└── README.md
```

## Requisitos

Se recomienda usar Python 3.10 o superior. Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

Dependencias principales:
- pandas, numpy
- matplotlib, mplsoccer
- seaborn, scikit-learn
- streamlit
- openai
- python-dotenv

## Seguridad: Clave API

El archivo `.env` no está incluido por motivos de seguridad, ya que contiene la clave de la API de OpenAI vinculada a una cuenta con método de pago. Para ejecutar los análisis que requieren interacción con la API, es necesario crear un archivo `.env` con la siguiente variable:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Visualizaciones disponibles

### Red de pases
- Script: `red_pases.py`
- Entrada: eventos de StatsBomb
- Salida: gráficos estilo StatsBomb y análisis táctico generado por IA

### Freeze Frame
- Script: `freeze_frame.py`
- Entrada: datos 360º
- Salida: visualización del posicionamiento de los jugadores y análisis asociado

### Formaciones tácticas
- Script: `kmeans_formacion.py`
- Análisis posicional y agrupación automática de formaciones por partido

### Aplicación Streamlit

Para lanzar la aplicación:

```bash
streamlit run app_freeze_streamlit.py
```

Esta aplicación permite seleccionar jugadas y visualizar todos los paneles sin necesidad de programación.

## Contacto

Proyecto desarrollado por Borja López como parte del Trabajo Final de Máster. Para dudas, sugerencias o aportes, puede abrir una issue en el repositorio.

© 2024 Borja López. Proyecto académico sin fines comerciales.
