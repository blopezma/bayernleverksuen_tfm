# Análisis Táctico del Bayer Leverkusen - TFM

Este repositorio contiene el código fuente utilizado en el Trabajo Final de Máster para el análisis táctico del Bayer Leverkusen en la Bundesliga 2023/2024. Incluye herramientas para el procesamiento de datos de StatsBomb, visualizaciones con Matplotlib y una interfaz interactiva construida en Streamlit.

## Repositorio

Repositorio GitHub: https://github.com/blopezma/bayernleverksuen_tfm

## Estructura del Proyecto

```
main
├── data/                    # Datos de entrada (.csv)
├── figs/                    # Salidas visuales (figuras PNG)
├── logos/                   # Logos utilizados                  
├── notebooks/               # Jupyter Notebooks explicativos
├── streamlit/               # Aplicaciones interactivas
├── env                      # NO INCLUIDO por seguridad
├── requirements.txt         # Dependencias
└── README.md
```

> Es importante mantener esta estructura de carpetas para asegurar la correcta ejecución de los notebooks y scripts, y garantizar la reproducibilidad completa del flujo de trabajo.

## Requisitos

> **Importante:** todos los comandos deben ejecutarse desde **Anaconda Prompt**, no desde PowerShell ni CMD, para asegurar que `python` y `pip` funcionen correctamente con el entorno configurado.

Se recomienda usar Python 3.10 o superior. Para instalar las dependencias:

1. Abre **Anaconda Prompt**.
2. Navega hasta el directorio raíz del proyecto (donde se encuentra el archivo `requirements.txt`):

```bash
cd ruta/a/la/carpeta/del/proyecto
pip install -r requirements.txt
```

Dependencias principales:
- pandas, numpy
- matplotlib, mplsoccer
- seaborn, scikit-learn
- statsbombpy
- streamlit
- openai
- python-dotenv
- requests, ipython, jupyterlab

## Seguridad: Clave API

El archivo .env no está incluido por motivos de seguridad, ya que contiene la clave de la API de OpenAI vinculada a una cuenta con método de pago. Para ejecutar los análisis que requieren interacción con la API, es necesario crear un archivo .env con la siguiente variable:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Notebooks incluidos

> El notebook `01_preprocesado_datos.ipynb` es el encargado de generar los archivos `.csv` intermedios en la carpeta `data/`, que son utilizados posteriormente por los notebooks `02`, `03`, `04` y `05`.

Los principales análisis están disponibles en formato Jupyter Notebook para facilitar la exploración, ejecución y replicación de resultados.

- `01_preprocesado_datos.ipynb`: Limpieza inicial y tratamiento de coordenadas.
- `02_red_pases.ipynb`: Generación de redes de pase y análisis automatizado con IA.
- `03_kmeans_formaciones.ipynb`: Detección de formaciones mediante clustering.
- `04_kmeans_pases.ipynb`: Detección de pases mediante clustering.
- `05_freeze_frame.ipynb`: Visualización y análisis táctico de datos 360°.

> Nota: el archivo .env con la clave de la API no se incluye por motivos de seguridad. Los archivos CSV no se incluyen en el repositorio debido a su tamaño. Se proporciona un notebook global para trabajar directamente con los datos ya extraídos desde la API de StatsBomb.

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
