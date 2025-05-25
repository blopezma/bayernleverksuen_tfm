
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
from mplsoccer import Pitch
import pandas as pd

def figura_freeze_campo(df, match_id, equipo, index=0,
                        tipo_jugada='Offensive',
                        evento='Pass',
                        exitoso=None,
                        tipo_fase_juego=None,
                        figsize=(12, 8)):

    def convertir_a_lista(x):
        if isinstance(x, str):
            return eval(x)
        return x

    eventos_validos = ['Pass', 'Carry', 'Shot']
    if evento not in eventos_validos:
        print(f"[AVISO] El evento '{evento}' no está soportado para análisis. Solo se permiten: {eventos_validos}.")
        return None

    df_filtrado = df[
        (df['match_id'] == match_id) &
        (df['team'] == equipo) &
        (df['event_type'] == evento) &
        (df['type_category'] == tipo_jugada)
    ].copy()

    if exitoso is not None:
        df_filtrado = df_filtrado[df_filtrado['exitoso'] == exitoso]

    if tipo_fase_juego is not None:
        df_filtrado = df_filtrado[df_filtrado['play_pattern'] == tipo_fase_juego]

    df_filtrado.reset_index(drop=True, inplace=True)

    if df_filtrado.empty or index >= len(df_filtrado):
        print("No se encontraron eventos que coincidan con los filtros.")
        return None

    df_filtrado['location'] = df_filtrado['location'].apply(convertir_a_lista)
    df_filtrado['freeze_frame'] = df_filtrado['freeze_frame'].apply(convertir_a_lista)
    df_filtrado['visible_area'] = df_filtrado['visible_area'].apply(convertir_a_lista)
    df_filtrado[['x_start', 'y_start']] = pd.DataFrame(df_filtrado['location'].tolist(), index=df_filtrado.index)

    if evento == 'Pass':
        df_filtrado['end_location'] = df_filtrado['pass_end_location'].apply(convertir_a_lista)
    elif evento == 'Carry':
        df_filtrado['end_location'] = df_filtrado['carry_end_location'].apply(convertir_a_lista)
    elif evento == 'Shot':
        df_filtrado['end_location'] = df_filtrado['shot_end_location'].apply(convertir_a_lista)

    df_filtrado['end_location'] = df_filtrado['end_location'].apply(
        lambda x: x[:2] if isinstance(x, list) and len(x) >= 2 else None)
    df_filtrado = df_filtrado[df_filtrado['end_location'].notna()].copy()
    df_filtrado[['x_end', 'y_end']] = pd.DataFrame(df_filtrado['end_location'].tolist(), index=df_filtrado.index)

    jugada = df_filtrado.iloc[[index]]
    equipo_actor = jugada['team'].values[0]
    es_bayer = equipo_actor == "Bayer Leverkusen"


    x0, y0 = jugada['x_start'].iloc[0], jugada['y_start'].iloc[0]
    x1, y1 = jugada['x_end'].iloc[0], jugada['y_end'].iloc[0]
    freeze = jugada['freeze_frame'].iloc[0]
    vis_area = jugada['visible_area'].iloc[0]
    exitoso = bool(jugada['exitoso'].iloc[0])
    color_evento = 'green' if exitoso else 'red'

    # === FIGURA ===
    fig = plt.figure(figsize=figsize, facecolor='white')

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg_ax.axis("off")
    bg_ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=fig.transFigure,
                                   boxstyle="round,pad=0.01", facecolor='white',
                                   edgecolor="black", linewidth=1.5, zorder=-1))

    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    ax1.add_patch(Rectangle((0, 0), 120, 80, facecolor='#d9d9d9', zorder=0))

    # ZONA VISIBLE
    if isinstance(vis_area, list) and len(vis_area) >= 6:
        coords = [(vis_area[i], vis_area[i + 1]) for i in range(0, len(vis_area) - 1, 2)]
        ax1.add_patch(plt.Polygon(coords, color='white', zorder=1))

    # CAMPO
    pitch = Pitch(pitch_type='statsbomb', pitch_color=None, line_color='black', linewidth=1)
    pitch.draw(ax=ax1)
    for el in ax1.findobj():
        try:
            el.set_zorder(2.5)
        except:
            continue

    # EVENTO
    pitch.scatter(x0, y0, ax=ax1, c='black', s=100, edgecolors='black', zorder=3)

    if x1 is not None and y1 is not None:
        if evento == 'Shot':
            distancia = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
            altura_curva = min(15, distancia * 0.3)
            dy = -altura_curva if y1 < y0 else altura_curva
            verts = [(x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2 + dy), (x1, y1)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, color=color_evento, lw=2,
                                    arrowstyle='-|>', mutation_scale=12, zorder=3)
            ax1.add_patch(patch)
        else:
            pitch.lines(x0, y0, x1, y1, ax=ax1, comet=True, color=color_evento, lw=2, zorder=3)

    # FREEZE FRAME
    for idx, player in enumerate(freeze):
        loc = player['location']
        is_actor = player.get("actor", False)
        is_keeper = player.get("keeper", False)
        
        if player.get("teammate", False):
            color = 'red' if es_bayer else 'darkblue'
        else:
            color = 'darkblue' if es_bayer else 'red'

        marker = 'D' if is_keeper else 'o'
        size = 400 if is_actor else 300
        edge_width = 3 if is_actor else 1.5
        edge_color = 'gold' if is_actor else 'black'

        ax1.scatter(loc[0], loc[1], c=color, s=size, marker=marker,
                    edgecolors=edge_color, linewidths=edge_width, zorder=4)
        ax1.text(loc[0], loc[1], str(idx + 1), color='white', fontsize=9,
                 ha='center', va='center', zorder=5, weight='bold' if is_actor else 'normal')

    ax1.set_xlim(0, 120)
    ax1.set_ylim(80, 0)
    ax1.axis('off')
    fig.tight_layout()
    fig.suptitle("Visualización Freeze Frame", fontsize=14, weight='bold', y=1.02)

    return fig
