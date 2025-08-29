import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple, Set
from typing_extensions import Self
from .animation_classes import FrameObject, PlayFrame, Play, _AnimArtists
import numpy as np


'''Color Constants'''
'''https://teamcolorcodes.com/nfl-team-color-codes/'''
WHITE = '#FFFFFF'
BLACK = '#000000'
LIGHT_GREEN = '#BDD9BF'
BASIC_BLUE = '#00338D'
BASIC_RED = '#D92F38'
BASIC_GOLD = '#E89B00'
FOOTBALL_COLOR = '#FB14E8'
SILVER = '#C0C0C0'

BR_PURPLE = '#241773'
ORANGE = '#FB4F14'
GOLD_2 = '#FFB612'
BROWN = '#311D00'
AQUA = '#008E97'
NAUTICAL_BLUE = '#002244'
GOTHAM_GREEN = '#125740'
DEEP_STEEL_BLUE = '#03202F'
SPEED_BLUE = '#002C5F'
TEAL = '#006778'
TITANS_BLUE = '#4B92DB'
BRONCOS_ORANGE = '#FB4F14'
KC_RED = '#E31837'
POWDER_BLUE = '#0080C6'
DARK_NAVY = '#0B162A'
HONOLULU_BLUE = '#0076B6'
DARK_GREEN = '#203731'
MNV_PURPLE = '#4F2683'
ROYAL_BLUE = '#003594'
DARK_BLUE = '#0B2265'
MIDNIGHT_GREEN = '#004C54'
BURGUNDY = '#5A1414'
FALCONS_RED = '#A71930'
CAROLINA_BLUE = '#0085CA'
OLD_GOLD = '#D3BC8D'
TB_RED = '#D50A0A'
AZ_RED = '#97233F'
LAR_BLUE = '#003594'
SF_RED = '#AA0000'
COLLEGE_NAVY = '#002244'

team_colors = {'BUF': BASIC_BLUE,
               'LA': LAR_BLUE,
               'NO': OLD_GOLD,
               'ATL': FALCONS_RED,
               'CLE': BROWN,
               'CAR': CAROLINA_BLUE,
               'SF': SF_RED,
               'CHI': DARK_NAVY,
               'CIN': ORANGE,
               'PIT': GOLD_2,
               'PHI': MIDNIGHT_GREEN,
               'DET': HONOLULU_BLUE,
               'IND': SPEED_BLUE,
               'HOU': DEEP_STEEL_BLUE,
               'MIA': AQUA,
               'NE': NAUTICAL_BLUE,
               'NYJ': GOTHAM_GREEN,
               'BAL': BR_PURPLE,
               'TEN': TITANS_BLUE,
               'NYG': DARK_BLUE,
               'JAX': TEAL,
               'WAS': BURGUNDY,
               'KC': KC_RED,
               'ARI': AZ_RED,
               'LV': BLACK,
               'LAC': POWDER_BLUE,
               'MIN': MNV_PURPLE,
               'GB': DARK_GREEN,
               'TB': TB_RED,
               'DAL': ROYAL_BLUE,
               'DEN': BRONCOS_ORANGE,
               'SEA': COLLEGE_NAVY}

ARTISTS = _AnimArtists()

def init_animation(ax: matplotlib.axes._axes.Axes,
                   offensive_team_name: str,
                   defensive_team_name: str,
                   n_offense: int,
                   n_defense: int):
    """
    Create the three scatter artists once and mark them animated for blitting.
    Call this before FuncAnimation starts.
    """
    # Empty but valid offsets
    off_xy  = np.zeros((n_offense, 2), dtype=float)
    def_xy  = np.zeros((n_defense, 2), dtype=float)
    ball_xy = np.zeros((1, 2), dtype=float)

    ARTISTS.off_scatter = ax.scatter(off_xy[:,0], off_xy[:,1],
                                     s=100, marker='o',
                                     animated=True, zorder=2,
                                     label=offensive_team_name)
    ARTISTS.def_scatter = ax.scatter(def_xy[:,0], def_xy[:,1],
                                     s=100, marker='o',
                                     animated=True, zorder=2,
                                     label=defensive_team_name)
    ARTISTS.ball_scatter = ax.scatter(ball_xy[:,0], ball_xy[:,1],
                                      s=100, marker='H',
                                      animated=True, zorder=2,
                                      label="football")
    
    base_def = team_colors[defensive_team_name]
    base_rgba = to_rgba(base_def, 1.0)                        # (r,g,b,1)
    ARTISTS.def_face_rgba = np.tile(base_rgba, (n_defense, 1))  # shape (11, 4)
    ARTISTS.def_scatter.set_facecolors(ARTISTS.def_face_rgba)

    base_off = team_colors[offensive_team_name]
    ARTISTS.off_scatter.set_facecolors([to_rgba(base_off, 1.0)])
    ARTISTS.ball_scatter.set_facecolors(FOOTBALL_COLOR)

    # 1) Probability text centered on the dot
    ARTISTS.def_prob_texts = []
    for _ in range(n_defense):
        t = ax.text(0, 0, "", ha='center', va='center',
                    fontsize=8, weight='bold', zorder=3, c='w',
#                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
                    animated=True)
        ARTISTS.def_prob_texts.append(t)

    # 2) Name label slightly above the dot (using annotate for offset in points)
    ARTISTS.def_name_annos = []
    for _ in range(n_defense):
        t = ax.text(
            0, 0, "", ha='center', va='bottom',
            fontsize=5, color='black', zorder=3,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6),
            animated=True
        )
        ARTISTS.def_name_annos.append(t)

    # Optional small speedups
    ARTISTS.off_scatter.set_antialiased(False)
    ARTISTS.def_scatter.set_antialiased(False)
    ARTISTS.ball_scatter.set_antialiased(False)

    # Return the artists so init_func can use them if you want
    return [ARTISTS.off_scatter, ARTISTS.def_scatter,
            ARTISTS.ball_scatter, *ARTISTS.def_prob_texts,
            *ARTISTS.def_name_annos]


def create_field(ax: matplotlib.axes._axes.Axes,
                 offensive_team_name: str = None,
                 defensive_team_name: str = None,
                 ball_name: str = "football"):

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=2,
                            edgecolor= BLACK, facecolor= LIGHT_GREEN, zorder=0)
    ax.add_patch(rect)

    # plot line numbers
    for yard_line in range(10, 120, 10):
        ax.axvline(x=yard_line, color= WHITE, zorder=1)
    # added to set y-axis up for the numbers
    ax.axhline(y=0, color= WHITE, zorder=1)
    ax.axhline(y=53.3, color= WHITE, zorder=1)

    # plot numbers
    for x in range(20, 110, 10):
        yard_number = x
        if x > 50:
            yard_number = 120-x
        ax.text(x, 4, str(yard_number - 10), horizontalalignment='center',
                fontsize=15, color= WHITE, zorder=1)
        ax.text(x-0.97, 53.3-4, str(yard_number-10), horizontalalignment='center',
                fontsize=15, color= WHITE, rotation=180, zorder=1)

    # hash marks
    for x in range(11, 110):
        ax.plot([x, x], [0.4, 0.7], color= WHITE, zorder=1)
        ax.plot([x, x], [53.0, 52.5], color= WHITE, zorder=1)
        ax.plot([x, x], [23, 23.66], color= WHITE, zorder=1)
        ax.plot([x, x], [29.66, 30.33], color= WHITE, zorder=1)

    # hide axis
    ax.set_axis_off()

    off_color = team_colors[offensive_team_name]
    def_color = team_colors[defensive_team_name]

    # create base scatter plots for the players location, allows for legend creation
    ax.scatter([], [], c= off_color, label = offensive_team_name, zorder=2)
    ax.scatter([], [], c= def_color, label = defensive_team_name, zorder=2)
    ax.scatter([], [], c= FOOTBALL_COLOR , label = ball_name, zorder=2)
    ax.legend(loc='upper right')
    
    
def _alpha(prob: float,
           min_alpha: float = 0.25,   # visibility floor for defenders
           max_alpha: float = 1.0,) -> float:
        prob = max(0.0, min(1.0, float(prob)))
        return min_alpha + (max_alpha - min_alpha) * prob
    
    
def update(frame: int,
           play_frames: list[PlayFrame] = None,
           offensive_team_name: str = None,
           defensive_team_name: str = None,
           ax: matplotlib.axes._axes.Axes = None,
           *,
           show_labels: bool=True) -> List[matplotlib.collections.PathCollection]:
    # pass in the list of PlayFrame objects.
    # Each PlayFrame object has a list of FrameObject objects.
    # Each of those FrameObject objects is a player or ball with coords.
    print(f"Going through the update function at frame: {frame}")
    current_frame = play_frames[frame]
    
    offense = [p for p in current_frame.points if p.on_offense == 1]
    defense = [p for p in current_frame.points if p.on_offense == 0]
    ball    = [p for p in current_frame.points if p.on_offense == -1]

    offense_xs = [p.x for p in offense]
    offense_ys = [p.y for p in offense]

    defense_xs = [p.x for p in defense]
    defense_ys = [p.y for p in defense]

    ball_xs = [p.x for p in ball]
    ball_ys = [p.y for p in ball]

    off_color = team_colors[offensive_team_name]
    def_color = team_colors[defensive_team_name]

    # Build Nx2 arrays once per frame (no Python loop for set_* calls)
    off_xy = np.column_stack([offense_xs, offense_ys])  # shape (N_off, 2)
    def_xy = np.column_stack([defense_xs, defense_ys])  # shape (N_def, 2)
    ball_xy = np.column_stack([ball_xs,   ball_ys])     # shape (1, 2)

    # Update positions (fast path)
    ARTISTS.off_scatter.set_offsets(off_xy)
    ARTISTS.def_scatter.set_offsets(def_xy)
    ARTISTS.ball_scatter.set_offsets(ball_xy)

    probs  = np.fromiter((getattr(p, "blitz_probs", 0.0) for p in defense),
                     dtype=float, count=len(defense))
    alphas = 0.25 + 0.75 * np.clip(probs, 0.0, 1.0)
    ARTISTS.def_face_rgba[:, 3] = alphas
    ARTISTS.def_scatter.set_facecolors(ARTISTS.def_face_rgba)
        
    artists_to_redraw = [ARTISTS.off_scatter,
                         ARTISTS.def_scatter,
                         ARTISTS.ball_scatter]
    
    if show_labels:
        # Update label text/positions without creating new artists
        # Ensure 1-to-1 mapping: defender index i ↔ label index i
        for i, p in enumerate(defense):
            prob = getattr(p, "blitz_probs", 0.0)
            name = getattr(p, "name", "DEF")

            # Move + update probability text
            prob_text = ARTISTS.def_prob_texts[i]
            prob_text.set_position((p.x, p.y))
            prob_text.set_text(f"{prob:.2f}")

            # Move + update name annotation (xy is the anchor)
            name_anno = ARTISTS.def_name_annos[i]
            name_anno.set_position((p.x, p.y+1))   # for Annotation, call set_position for xy
#             name_anno.xy = (p.x, p.y)
            # Matplotlib’s Annotation wants .set_text(...) for content
            name_anno.set_text(name)

        # If you want to throttle label updates (optional):
        # if frame % 2 != 0:
        #     pass  # skip adding label artists to the redraw list this frame

        artists_to_redraw.extend(ARTISTS.def_prob_texts)
        artists_to_redraw.extend(ARTISTS.def_name_annos)

    return artists_to_redraw


''' Plot player animation on the football field '''
def plot_play(play: Play) -> matplotlib.animation.FuncAnimation:

    gpid = play.gpid
    id_parts = gpid.split("-")
    game_id = int(id_parts[0])
    play_id = int(id_parts[1])

    off_name = play.off_team
    def_name = play.def_team
    ball_name = "football" #play_df['club'].unique()[2]

    play_frames = play.frames # List of PlayFrame objects
    num_frames = len(play_frames)
    print(num_frames)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    create_field(ax,
                 offensive_team_name = off_name,
                 defensive_team_name = def_name,
                 ball_name = ball_name
                )
    # Fix axes once so autoscale doesn't run every frame
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal', adjustable='box')
    
    print("Initializing the animation")
    init_artists = init_animation(ax, off_name, def_name,
                                  n_offense=11, n_defense=11)

    def _init():
        # Return the artists so blitting knows what to draw initially
        return init_artists
    
    farg_tuple = (play_frames, off_name, def_name, ax)
    
    print("Creating animation")
    animation = FuncAnimation(fig, update,
                          frames=range(0, num_frames),
                          fargs=farg_tuple,
                          init_func=_init,
                          blit=True,
                          interval=125,
                          repeat=False,
                          cache_frame_data=False,
                          )

    plt.subplots_adjust(top=0.8)
    plt.subplots_adjust(right=0.7)
    return animation