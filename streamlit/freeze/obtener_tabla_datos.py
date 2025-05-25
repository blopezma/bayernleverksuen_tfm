
import pandas as pd
import ast

def obtener_tabla_datos(df, match_id, equipo, index=0,
                        tipo_jugada='Offensive',
                        evento='Pass',
                        exitoso=None,
                        tipo_fase_juego=None):
    df_filtered = df[
        (df['match_id'] == match_id) &
        (df['team'] == equipo) &
        (df['event_type'] == evento) &
        (df['type_category'] == tipo_jugada)
    ]
    if exitoso is not None:
        df_filtered = df_filtered[df_filtered['exitoso'] == exitoso]
    if tipo_fase_juego is not None:
        df_filtered = df_filtered[df_filtered['play_pattern'] == tipo_fase_juego]

    df_filtered = df_filtered.reset_index(drop=True)
    if df_filtered.empty or index >= len(df_filtered):
        return pd.DataFrame(columns=["Etiqueta", "Valor"])

    jugada = df_filtered.iloc[[index]]

    raw_freeze = jugada['freeze_frame'].iloc[0]
    if isinstance(raw_freeze, str):
        freeze = ast.literal_eval(raw_freeze)
    else:
        freeze = raw_freeze

    actor = jugada['player'].values[0]
    receptor = jugada['pass_recipient'].values[0] if 'pass_recipient' in jugada.columns else None

    datos = {
        "Event Type": jugada['type'].values[0],
        "Play Type": jugada['play_pattern'].values[0],
        "Actor": actor,
        "Receptor": receptor if pd.notna(receptor) else None,
        "Outcome": "Success" if jugada['exitoso'].values[0] else "Failure",
        "Category": jugada['type_category'].values[0],
        "Compañeros visibles": sum(p.get("teammate", False) for p in freeze),
        "Rivales visibles": sum(not p.get("teammate", False) for p in freeze),
        "Duration": f"{jugada['duration'].values[0]:.2f}s" if 'duration' in jugada else "0.00s"
    }

    return pd.DataFrame(list(datos.items()), columns=["Etiqueta", "Valor"])
