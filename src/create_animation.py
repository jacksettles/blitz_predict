import sqlite3
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from typing_extensions import Self
import matplotlib.pyplot as plt
from animations.animation_classes import FrameObject, PlayFrame, Play
from animations.animation_functions import plot_play
from data_engineering.create_sequences import (
    remove_na_labels,
    impute_make_numeric,
    impute_make_numeric,
    sort_df,
    compute_football_dir_and_o_in_play,
    organize_play
)
import sys
import torch
from mamba_models import Mamba
from custom_models import BlitzLSTM

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="saved_models/sched_free_model3.pt", help="Path to trained model")
parser.add_argument("--week", type=int, default=1, help="Week number of tracking data to pull (1-9)")
parser.add_argument("--gpid", type=str, help="If looking to animate a specific play, pass in the gameId-playId (gpid). Set random_play to False (0).")
parser.add_argument("--random_play", type=int, default=1, help="Whether the play to animate is random or not. Defualt 1.")
parser.add_argument("--event", type=str, help="If random_play is False, can specify an event type, like 'qb_sack'.")
parser.add_argument("--save_as", type=str, default="gif", help="Save clip as gif or mp4")

conn = sqlite3.connect("./data/nfldata.db")

if torch.cuda.is_available():
    device = "cuda"
    print("Cuda device available, using GPU")
else:
    device = "cpu"
    print("Cuda not available, using CPU")


def load_model(model_path):
    print(f"Using model from: {model_path}!")
    model = torch.load(model_path, map_location=device, weights_only=False)['MODEL']
    return model

def read_table(week: int = 1):
    table = f"rush_labels_{week}"
    print(f"Reading in {table} from DB, could take about 5 minutes\n")
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    
    print("Removing plays where defenders have no label...")
    df = remove_na_labels(df)

    print("Converting data types...")
    df = impute_make_numeric(df)

    print("Sorting DF...")
    df = sort_df(df)
    return df


def get_gpid(df: pd.DataFrame, gpid: Optional[str]=None, event: Optional[str]=None,
             player: Optional[str]=None, team: Optional[str]=None) -> str:
    if gpid:
        filtered_df = df.loc[df["gpid"] == gpid]
        print(f"Testing gpid: {gpid}")
    else:
        if (event != None) & (player != None) & (team != None):
            filtered_df = df.loc[(df['event'] == event) & (df['club'] == team) & (df['displayName'] == player)]
        elif (event != None) & (team != None):
            filtered_df = df.loc[(df['event'] == event) & (df['club'] == team)]
        elif (event != None) & (player != None):
            filtered_df = df.loc[(df['event'] == event) & (df['displayName'] == player)]
        elif (player != None) & (team != None):
            filtered_df = df.loc[(df['displayName'] == player) & df['club'] == team]
        elif (event != None):
            filtered_df = df.loc[df['event'] == event]
        elif (team != None):
            filtered_df = df.loc[df['club'] == team]
        elif (player != None):
            filtered_df = df.loc[df['displayName'] == player]
        else:
            print("Random choice gpid")
            filtered_df = df
    return np.random.choice(filtered_df['gpid'].unique())


def construct_play(play_df: pd.DataFrame, gpid: str, predictions: torch.Tensor):
    play_length = play_df['frameId'].max()
    frames = []
    offensive_team = play_df.loc[play_df['on_offense'] == 1, 'club'].iloc[0]
    defensive_team = play_df.loc[play_df['on_offense'] == 0, 'club'].iloc[0]
    
    num_pred_frames = predictions.size(0)
    
    for i in range(play_length):
        if i < num_pred_frames:
            frame_predictions = predictions[i]
        points = []
        play_frame_df = play_df.query(f"frameId == {i+1}")
        idx = 0
        for j, row in play_frame_df.iterrows():
            name = row['displayName']
            x = row['x']
            y = row['y']
            on_offense = row['on_offense']
            if i < num_pred_frames:
                blitz = frame_predictions[idx].item()
            else:
                blitz = 0
            player = FrameObject(name, x, y, on_offense=on_offense, is_blitzing=blitz)
            points.append(player)
            idx += 1
        pf_name = f"{gpid}-{i+1}"
        frame = PlayFrame(pf_name, points)
        frames.append(frame)
    play = Play(gpid=gpid, frames=frames, off_team=offensive_team, def_team=defensive_team)
    return play
    

def main(args):
    model = load_model(args.model_path)
    df = read_table(args.week)
    
    if args.random_play == 1:
        gpid = get_gpid(df)
        print(f"Pulling data for play gpid: {gpid}")
    else:
        if args.gpid:
            gpid = get_gpid(df, gpid=args.gpid)
        elif args.event:
            gpid = get_gpid(df, event=args.event)
            print(f"Event type: {args.event}, pulling data for play gpid: {gpid}")
        
    play_df = df[df["gpid"] == gpid].copy()
    play_df = compute_football_dir_and_o_in_play(play_df)
    
    snap_frame = play_df.loc[play_df["frameType"] == "SNAP", "frameId"].unique()[0]
    play_df_for_features = play_df[play_df['frameId'] <= (snap_frame + 15)].copy()
    
    features, labels = organize_play(play_df_for_features)
    play_df.loc[play_df["displayName"] == "football", "on_offense"] = -1
    
    features = features.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(features) # shape [1, num_frames, 23, 3]
    predictions = torch.argmax(outputs, dim=-1) # shape [1, num_frames, 23]
    preds = predictions.squeeze(0) # shape [num_frames, 23]

    play = construct_play(play_df, gpid, preds)
    
    animation = plot_play(play)

    if args.save_as == "gif":
        animation.save(f"{gpid}.gif", writer='pillow', fps=15)
    elif args.save_as == "mp4":
        animation.save(f"{gpid}.mp4", writer='ffmpeg', fps=15)

    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)