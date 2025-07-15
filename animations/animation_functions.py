import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple, Set
from typing_extensions import Self
from .animation_classes import FrameObject, PlayFrame, Play


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
    plt.axis('off')

    off_color = team_colors[offensive_team_name]
    def_color = team_colors[defensive_team_name] + "80"

    # create base scatter plots for the players location, allows for legend creation
    ax.scatter([], [], c= off_color, label = offensive_team_name, zorder=2)
    ax.scatter([], [], c= def_color, label = defensive_team_name, zorder=2)
    ax.scatter([], [], c= FOOTBALL_COLOR , label = ball_name, zorder=2)
    ax.legend(loc='upper right')

    # statistics overview tables
    # plt.table(cellText=data,
    #                 colWidths=[0.1]*4,
    #                 colLabels=list(stat_overview.columns),
    #                 loc='right',
    #                 )
    # plt.table(cellText=data2,
    #                 colWidths=[0.1]*2,
    #                 colLabels=list(percentile_cal.columns),
    #                 loc='bottom')

    # initial plots for jersey numbers
    # for x in range(0, 14):
    #     d["label{0}".format(x)] = ax.text(0, 0, '', fontsize = 'small', fontweight = 700, zorder=4)

    # plot legend
    ax.legend(loc='upper right')
    
    
def update(frame: int,
           play_frames: list[PlayFrame] = None,
           offensive_team_name: str = None,
           defensive_team_name: str = None,
           ax: matplotlib.axes._axes.Axes = None) -> List[matplotlib.collections.PathCollection]:
    # pass in the list of PlayFrame objects.
    # Each PlayFrame object has a list of FrameObject objects.
    # Each of those FrameObject objects is a player or ball with coords.
#     print(f"Frame: {frame}")
    current_frame = play_frames[frame]
    
    offense_xs = [player.x for player in current_frame.points if player.on_offense == 1]
    offense_ys = [player.y for player in current_frame.points if player.on_offense == 1]
#     defense_xs = [player.x for player in current_frame.points if player.on_offense == 0]
#     defense_ys = [player.y for player in current_frame.points if player.on_offense == 0]
    no_blitz_xs = [player.x for player in current_frame.points if player.on_offense == 0 and player.is_blitzing == 0]
    no_blitz_ys = [player.y for player in current_frame.points if player.on_offense == 0 and player.is_blitzing == 0]
    blitz_xs = [player.x for player in current_frame.points if player.on_offense == 0 and player.is_blitzing == 1]
    blitz_ys = [player.y for player in current_frame.points if player.on_offense == 0 and player.is_blitzing == 1]
    ball_xs = [player.x for player in current_frame.points if player.on_offense == -1]
    ball_ys = [player.y for player in current_frame.points if player.on_offense == -1]

    off_color = team_colors[offensive_team_name]
    no_blitz_color = team_colors[defensive_team_name] + "80"
    blitz_color = team_colors[defensive_team_name]

    ax.clear()
    create_field(ax,
                 offensive_team_name = offensive_team_name,
                 defensive_team_name = defensive_team_name,
                 ball_name = "football")

    artists_to_redraw = []

    # visualize plots
    offense_plot = ax.scatter(offense_xs, offense_ys, s=100, linestyle='None', marker='o',
                          c= off_color, label=offensive_team_name, zorder=2)

#     defense_plot = ax.scatter(defense_xs, defense_ys, linestyle='None', marker='o',
#                           c= def_color, label=defensive_team_name, zorder=2)
    no_blitz_plot = ax.scatter(no_blitz_xs, no_blitz_ys, s=100, linestyle='None', marker='o',
                          c= no_blitz_color, label=defensive_team_name, zorder=2)
    
    blitz_plot = ax.scatter(blitz_xs, blitz_ys, s=100, linestyle='None', marker='X',
                          c= blitz_color, label="blitz", zorder=2)

    ball_plot = ax.scatter(ball_xs, ball_ys, s=100, linestyle='None', marker='H',
                         c= FOOTBALL_COLOR, label="football", zorder=2)

    artists_to_redraw.append(offense_plot)
#     artists_to_redraw.append(defense_plot)
    artists_to_redraw.append(no_blitz_plot)
    artists_to_redraw.append(blitz_plot)
    artists_to_redraw.append(ball_plot)

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

    fig, ax = plt.subplots(figsize=(25, 13))
    create_field(ax,
                 offensive_team_name = off_name,
                 defensive_team_name = def_name,
                 ball_name = ball_name
                )

    
    farg_tuple = (play_frames, off_name, def_name, ax)
    
    animation = FuncAnimation(fig, update, interval=125, repeat=True,
                            frames=range(0, num_frames), fargs=farg_tuple,
                            blit=True)

    plt.subplots_adjust(top=0.8)
    plt.subplots_adjust(right=0.7)
#     plt.show()
    return animation