
from freeze.freeze import generar_prompt_jugada_tactica, generar_imagen_freeze_base64, analizar_jugada_con_imagen

def obtener_texto_ia(df, match_id, equipo, index=0,
                     tipo_jugada='Offensive',
                     evento='Pass',
                     exitoso=None,
                     tipo_fase_juego=None,
                     modelo="gpt-4o"):
    df_filtrado = df[
        (df['match_id'] == match_id) &
        (df['team'] == equipo) &
        (df['event_type'] == evento) &
        (df['type_category'] == tipo_jugada)
    ]

    if exitoso is not None:
        df_filtrado = df_filtrado[df_filtrado['exitoso'] == exitoso]
    if tipo_fase_juego is not None:
        df_filtrado = df_filtrado[df_filtrado['play_pattern'] == tipo_fase_juego]

    df_filtrado = df_filtrado.reset_index(drop=True)
    if df_filtrado.empty or index >= len(df_filtrado):
        return "[No se pudo generar análisis IA: jugada no encontrada]"

    jugada = df_filtrado.iloc[[index]]
    prompt = generar_prompt_jugada_tactica(jugada)
    base64_img = generar_imagen_freeze_base64(df, match_id, equipo, index, tipo_jugada, evento, exitoso, tipo_fase_juego, modelo)
    return analizar_jugada_con_imagen(base64_img, prompt, modelo)
