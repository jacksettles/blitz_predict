import sqlite3
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import torch
import sys
import os
import random
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", required=True, type=str, choices=["mamba", "transformer"], help="What model you are processing this data for")

def remove_na_labels(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df['on_offense'] == 0) & (df['displayName'] != 'football') # Defenders only, no offense and no football
    def_only_df = df[mask]
    null_rush = def_only_df[def_only_df['is_rushing'].isna()]
    na_rush = def_only_df[def_only_df['is_rushing'] == 'NA']
    discard_plays = null_rush['gpid'].unique().tolist() + na_rush['gpid'].unique().tolist()
    print(f"Number of discard plays because defense does not have a label: {len(discard_plays)}")
    df = df[~df['gpid'].isin(discard_plays)]
    return df


def impute_make_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    int_cols = ['nflId', 'frameId', 'jerseyNumber']
    float_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir']
    label_col = ['is_rushing']
    numeric_cols = float_cols + int_cols + label_col

    df.replace(
        {
            'nflId': 'NA',
            'jerseyNumber': 'NA',
            'is_rushing': 'NA',
            'x': 'NA',
            'y': 'NA',
            's': 'NA',
            'a': 'NA',
            'dis': 'NA',
            'o': 'NA',
            'dir': 'NA'
        },
        np.nan, 
        inplace=True
        )

    df[numeric_cols] = df[numeric_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce')
    )
    df[float_cols] = df[float_cols].astype(float)
    df[int_cols] = df[int_cols].fillna(-1).astype(int)  # these '-1' values should be the football only
    df[label_col] = df[label_col].fillna(0).astype(int) # these '0' labels should be football and offensive players only
                                                        # they get masked out anyways during loss computation, so these 
                                                        # parameters just don't get updated
    print(f"Number of unique plays: {df['gpid'].nunique()}")
    return df


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
            ['gpid', 'frameId', 'on_offense', 'position'],
            ascending=[True, True, True, True]
        )


def compute_football_dir_and_o_in_play(play_df):
    """
    Computes and fills the 'dir' and 'o' columns for the football rows in a single play's DataFrame.

    Args:
        play_df (pd.DataFrame): DataFrame for one play, sorted by [frameId, on_offense, position]
                                and containing all agents including the football.

    Returns:
        pd.DataFrame: Same DataFrame with 'dir' and 'o' filled for the football rows.
    """
    football_mask = play_df['displayName'] == 'football'
    df_ball = play_df[football_mask]

    # Compute direction of motion using position deltas
    dx = df_ball['x'].diff()
    dy = df_ball['y'].diff()

    dir_rad = np.arctan2(dy, dx)
    dir_deg = np.rad2deg(dir_rad) % 360
    dir_deg = dir_deg.bfill()  # fill first frame

    # Assign to both 'dir' and 'o' for the football rows
    play_df.loc[football_mask, 'dir'] = dir_deg
    play_df.loc[football_mask, 'o'] = dir_deg

    return play_df


def standardize_left_to_right(df: pd.DataFrame):
    df = df.copy()
    mask = df["playDirection"] == 'left'  # only flip right-to-left plays
    
    # Flip coordinates
    df.loc[mask, "x"] = 120 - df.loc[mask, "x"]
    df.loc[mask, "y"] = 53.3 - df.loc[mask, "y"]
    
    # Flip orientation and direction
    for col in ["o", "dir"]:
        df.loc[mask, col] = (180 - df.loc[mask, col]) % 360
    return df


def compute_seconds_remaining_in_half(df):
    out = df.copy()

    # split into minutes and seconds
    mins_secs = out['gameClock'].str.split(':', expand=True)
    mins = mins_secs[0].astype(int)
    secs = mins_secs[1].astype(int)

    # total seconds left in quarter
    sec_left_qtr = mins * 60 + secs

    # add 15 minutes if it's 1st or 3rd quarter
    out['sec_left_half'] = sec_left_qtr + np.where(out['quarter'].isin([1, 3]), 15*60, 0)

    return out


def one_hot_encode_downs(df: pd.DataFrame):
    downs = pd.get_dummies(df['down'], prefix='down').astype(int)
    df = pd.concat([df.drop('down', axis=1), downs], axis=1)
    return df


def simple_scale(df: pd.DataFrame):
    """
    Scaling these columns is really simple.
    """
    df.copy()
    df['x'] = df['x'] / 120.0
    df['y'] = df['y'] / 53.3
    df['sec_left_half'] = df['sec_left_half'] / 1800.0 
    df['sin_o'] = np.sin(np.deg2rad(df['o']))      # orientation
    df['cos_o'] = np.cos(np.deg2rad(df['o']))
    df['sin_dir'] = np.sin(np.deg2rad(df['dir']))  # direction of motion
    df['cos_dir'] = np.cos(np.deg2rad(df['dir']))
    return df


def organize_play(df: pd.DataFrame, model_type="mamba") -> pd.DataFrame:
    """
    df here is an individual play's df, after grouping by 'gpid'.
    """
    tracking_feat_cols = ['x','y','s','a','dis', 'sin_o', 'cos_o', 'sin_dir', 'cos_dir']
    play_feat_cols = ['down_1', 'down_2', 'down_3', 'down_4', 'sec_left_half', 'yardsToGo' ,'score_diff']
    label_col = 'is_rushing'
    df = df.copy()
    play_feats = df[play_feat_cols].drop_duplicates().values # array
    
    tracking_feats = torch.tensor(df[tracking_feat_cols].values, dtype=torch.float32)
    play_feats = torch.tensor(play_feats, dtype=torch.float32)   # tensor
    labels = torch.tensor(df[label_col].values, dtype=torch.float32)

    if model_type == "mamba":
        seq_len = df['frameId'].nunique()
        tracking_feats = tracking_feats.reshape(seq_len, 22 * len(tracking_feat_cols))
        labels = labels.reshape(seq_len, -1)
    elif model_type == "transformer":
        seq_len = df['frameId'].nunique() * 22
        tracking_feats = tracking_feats.reshape(seq_len, len(tracking_feat_cols))
        labels = labels.reshape(seq_len)

    return tracking_feats, play_feats, labels


def make_datasets(df: pd.DataFrame, model_type="mamba"):
    tuple_list = []
    for gpid, play_df in df.groupby('gpid'):
        tensor_tuple = organize_play(play_df, model_type=model_type)
        for tens in tensor_tuple:
            if torch.isnan(tens).any():
                print(f"Skipping play {gpid}, found nans in tensors")
                continue
        tuple_list.append(tensor_tuple)
    return tuple_list
    

def main(args):
    conn = sqlite3.connect("/scratch/jts75596/fb/data/nfldata.db")
    
    pbp_cols = "gameId, playId, quarter, down, yardsToGo, yardlineNumber, gameClock, preSnapHomeScore, preSnapVisitorScore"
    pbp = pd.read_sql_query(f"SELECT {pbp_cols} FROM plays;", conn)
    pbp['gpid'] = pbp['gameId'].astype(str) + "-" + pbp['playId'].astype(str)
    pbp = pbp.drop(columns=['gameId', 'playId'])
    pbp['score_diff'] = pbp['preSnapHomeScore'].astype(int) - pbp['preSnapVisitorScore'].astype(int)
    pbp['down'] = pbp['down'].astype(int)
    pbp['yardsToGo'] = pbp['yardsToGo'].astype(int)
    pbp['quarter'] = pbp['quarter'].astype(int)
    pbp = compute_seconds_remaining_in_half(pbp) #sec_left_half
    pbp = pbp.set_index("gpid")

    play_df_list = []
    for i in tqdm(range(1, 10), total=9, desc="Week iteration"):
        table = f"rush_labels_{i}"
        print(f"Reading in {table} from DB, could take about 5 minutes\n")
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)

        print("Removing plays where defenders have no label...")
        df = remove_na_labels(df)

        print("Converting data types...")
        df = impute_make_numeric(df)

        print("Sorting DF...")
        df = sort_df(df)
        df = standardize_left_to_right(df)
        df = df.set_index("gpid").join(pbp).reset_index()
        
        # Binary variable for identifying the ball easily
        df['Is_ball'] = (df['displayName'] == 'football').astype(int)

        grouped_plays = df.groupby('gpid', sort=False)

        print("Iterating over plays individually to organize them...")

        for gpid, play_df in tqdm(grouped_plays, total=len(grouped_plays)):
            try:
                play_df = play_df[play_df['Is_ball'] == 0]
                start_frame = play_df.loc[play_df['event'] == 'line_set', 'frameId'].unique()[0]
                start_frame = max(0, start_frame - 15)
                snap_frame = play_df.loc[play_df['frameType'] == 'SNAP', 'frameId'].unique()[0]
                play_df = play_df[(play_df['frameId'] >= start_frame) & (play_df['frameId'] <= snap_frame)]
                play_df_list.append(play_df)
            except Exception as e:
                print(f"GPID: {gpid} - {e}")
                continue
    
    # Do train-val-test split
    random.seed(42)
    split_idx = int(len(play_df_list) * 0.8)   # splitting 80% off for train
    random.shuffle(play_df_list)
    train_list = play_df_list[:split_idx]
    temp = play_df_list[split_idx:]
    split_idx = int(len(temp) * 0.5)           # splitting 50% of the remaining 20% for val and test
    val_list = temp[:split_idx]
    test_list = temp[split_idx:]
    
    train_df = pd.concat(train_list, axis=0)
    val_df = pd.concat(val_list, axis=0)
    test_df = pd.concat(test_list, axis=0)
    
    # Standard scale cols
    scaler = StandardScaler()
    scale_cols = ['s', 'a', 'dis', 'yardsToGo', 'score_diff']
    scaler.fit(train_df[scale_cols])
    joblib.dump(scaler, "scaler.pkl")
    
    train_df[scale_cols] = scaler.transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])
    
    # One-hot encode downs
    train_df = one_hot_encode_downs(train_df)
    val_df = one_hot_encode_downs(val_df)
    test_df = one_hot_encode_downs(test_df)
    
    # Scale out of the max value
    train_df = simple_scale(train_df)
    val_df = simple_scale(val_df)
    test_df = simple_scale(test_df)
    
    # Returns a list of tuples of tensors (tracking_features, play_features, labels)
    train = make_datasets(train_df, model_type=args.model_type)
    val = make_datasets(val_df, model_type=args.model_type)
    test = make_datasets(test_df, model_type=args.model_type)
    
    print(f"# train plays: {len(train)}")
    print(f"# val plays: {len(val)}")
    print(f"# test plays: {len(test)}")
    
    save_dir = f"../data/processed_data/{args.model_type}"
    if not os.path.exists(save_dir):
        print(f"Making save directory {save_dir}!")
        os.makedirs(save_dir)
        
    print(f"Saving to directory {save_dir}!")
    
    torch.save(train, f"{save_dir}/train.pt")
    torch.save(val, f"{save_dir}/val.pt")
    torch.save(test, f"{save_dir}/test.pt")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)