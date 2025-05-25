
def obtener_marcador(df, match_id, equipo, index=0,
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
    jugada = df_filtered.iloc[[index]]

    minuto = int(jugada['minute'].values[0])
    segundo = int(jugada['second'].values[0])
    home_team = jugada['home_team'].values[0]
    away_team = jugada['away_team'].values[0]

    df_parcial = df[(df['match_id'] == match_id) &
                    ((df['minute'] < minuto) | ((df['minute'] == minuto) & (df['second'] <= segundo)))]

    goles_local = df_parcial[(df_parcial['type'] == 'Shot') &
                             (df_parcial['shot_outcome'] == 'Goal') &
                             (df_parcial['team'] == home_team)].shape[0]

    goles_visitante = df_parcial[(df_parcial['type'] == 'Shot') &
                                 (df_parcial['shot_outcome'] == 'Goal') &
                                 (df_parcial['team'] == away_team)].shape[0]

    return {
        "minuto": minuto,
        "segundo": segundo,
        "marcador": f"{home_team} {goles_local} - {goles_visitante} {away_team}",
        "home_team": home_team,
        "away_team": away_team,
        "goles_local": goles_local,
        "goles_visitante": goles_visitante
    }
