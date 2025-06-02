import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from freeze.figura_freeze_campo import figura_freeze_campo
from freeze.obtener_texto_ia import obtener_texto_ia
from freeze.obtener_marcador import obtener_marcador
from freeze.obtener_tabla_datos import obtener_tabla_datos

st.set_page_config(layout="wide")
st.title("Análisis táctico: Freeze Frame")

@st.cache_data
def cargar_datos():
    return pd.read_csv("../data/database_v5.csv", low_memory=False)

df = cargar_datos()

@st.cache_data
def construir_partidos(df):
    df_partidos = df[['match_id', 'home_team', 'away_team']].drop_duplicates().sort_values("match_id").reset_index(drop=True)
    df_partidos['jornada'] = ["J" + str(i + 1) for i in range(len(df_partidos))]
    df_partidos['nombre_partido'] = df_partidos['jornada'] + " - " + df_partidos['home_team'] + " vs " + df_partidos['away_team']
    return df_partidos

df_partidos = construir_partidos(df)

st.sidebar.header("Filtros de jugada")
partido_seleccionado = st.sidebar.selectbox("Partido", df_partidos['nombre_partido'].tolist())
match_id = df_partidos[df_partidos['nombre_partido'] == partido_seleccionado]['match_id'].values[0]

equipo = "Bayer Leverkusen"
st.sidebar.selectbox("Equipo", [equipo], index=0, disabled=True)

tipo_jugada = st.sidebar.selectbox("Fase", ["Offensive", "Defensive", "Neutral"])
play_patterns = [None] + sorted(df["play_pattern"].dropna().unique().tolist())
tipo_fase_juego = st.sidebar.selectbox("Tipo de reanudación", play_patterns)

eventos_disponibles = ['Pass', 'Carry', 'Shot']
evento = st.sidebar.selectbox("Evento", eventos_disponibles)

exitoso = st.sidebar.selectbox("¿Exitoso?", [None, True, False], format_func=lambda x: "Todos" if x is None else str(x))

# Determinar el equipo que genera el evento
if tipo_jugada == "Defensive":
    info_partido = df[df["match_id"] == match_id].dropna(subset=["home_team", "away_team"]).iloc[0]
    equipo_evento = info_partido["away_team"] if info_partido["home_team"] == equipo else info_partido["home_team"]
else:
    equipo_evento = equipo

# Filtrado sin usar type_category
df_filtrado = df[
    (df['match_id'] == match_id) &
    (df['team'] == equipo_evento) &
    (df['event_type'] == evento)
]
if exitoso is not None:
    df_filtrado = df_filtrado[df_filtrado['exitoso'] == exitoso]
if tipo_fase_juego is not None:
    df_filtrado = df_filtrado[df_filtrado['play_pattern'] == tipo_fase_juego]
df_filtrado.reset_index(drop=True, inplace=True)

if not df_filtrado.empty:
    opciones = [f"Jugada {i+1} - minuto: {row['minute']}:{int(row['second']):02d}" for i, row in df_filtrado.iterrows()]
    seleccion = st.sidebar.selectbox("Selecciona jugada", options=list(enumerate(opciones)), format_func=lambda x: x[1])
    index = seleccion[0]
    generar = st.sidebar.button("Generar análisis")

    if generar:
        try:
            jugada_seleccionada = df_filtrado.iloc[[index]]
        except IndexError:
            st.error("Índice fuera de rango. Puede que los datos hayan cambiado.")
        else:
            marcador = obtener_marcador(df_filtrado, match_id, equipo_evento, index, tipo_jugada, evento, exitoso, tipo_fase_juego)

            if marcador["home_team"] == "Bayer Leverkusen":
                color_local = "#d7191c"
                color_visitante = "#2c7bb6"
            else:
                color_local = "#2c7bb6"
                color_visitante = "#d7191c"

            st.markdown(f'''
            <div style="text-align: center; background-color: #f0f0f0; padding: 10px 20px;">
              <div style="font-size: 18px; font-weight: bold;">Minuto {marcador['minuto']}:{marcador['segundo']:02d}</div>
              <div style="font-size: 24px; font-weight: bold; margin-top: 5px;">
                <span style="color: {color_local};">{marcador['home_team']} {marcador['goles_local']}</span>
                &nbsp;-&nbsp;
                <span style="color: {color_visitante};">{marcador['goles_visitante']} {marcador['away_team']}</span>
              </div>
            </div>
            ''', unsafe_allow_html=True)

            col1, col2 = st.columns([1.3, 1])

            with col1:
                fig = figura_freeze_campo(df, match_id, equipo_evento, index, tipo_jugada, evento, exitoso, tipo_fase_juego)
                if fig:
                    st.pyplot(fig)

            with col2:
                with st.spinner("Generando análisis táctico IA..."):
                    texto_ia = obtener_texto_ia(df, match_id, equipo_evento, index, tipo_jugada, evento, exitoso, tipo_fase_juego)

                    def convertir_negritas(texto):
                        texto = re.sub(r"\*\*(.+?)\*\*:", r"<strong>\1:</strong>", texto)
                        texto = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto)
                        texto = re.sub(r"\\*(\d+):", r"\1.", texto)
                        texto = re.sub(r"(?<=[a-záéíóú])\.(?= [A-ZÁÉÍÓÚ])", ".<br><br>", texto)
                        return texto

                    texto_ia_html = convertir_negritas(texto_ia)
                    st.markdown(f"<div style='padding: 10px; background-color: #f9f9f9; height: 100%;'>{texto_ia_html}</div>", unsafe_allow_html=True)

            st.markdown("### Información de la jugada")
            df_tabla = obtener_tabla_datos(df, match_id, equipo_evento, index, tipo_jugada, evento, exitoso, tipo_fase_juego)

            def render_tabla_etiqueta(df):
                html = "<table style='width:100%; border-collapse: collapse;'>"
                html += "<thead><tr style='background-color:#e6e6e6;'><th style='text-align:left;padding:6px;'>Etiqueta</th><th style='text-align:left;padding:6px;'>Valor</th></tr></thead><tbody>"
                for _, row in df.iterrows():
                    html += f"<tr><td style='padding:6px;'>{row['Etiqueta']}</td><td style='padding:6px;'>{row['Valor']}</td></tr>"
                html += "</tbody></table>"
                return html

            html_tabla = render_tabla_etiqueta(df_tabla)
            st.markdown(html_tabla, unsafe_allow_html=True)
else:
    st.warning("No hay jugadas disponibles con los filtros seleccionados.")