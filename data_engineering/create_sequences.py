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

def remove_na_labels(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df['on_offense'] == 0) & (df['displayName'] != 'football') # Defenders only, no offense and no football
    def_only_df = df[mask]
    null_rush = def_only_df[def_only_df['is_rushing'].isna()]
    na_rush = def_only_df.query("is_rushing == 'NA'")
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
    df[int_cols] = df[int_cols].fillna(-1).astype(int) # these '-1' values should be the football only
    df[label_col] = df[label_col].fillna(2).astype(int) # these '2' labels should be football and offensive players only
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


def organize_play(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = ['x','y','s','a','dis', 'sin_o', 'cos_o', 'sin_dir', 'cos_dir']
    label_col = 'is_rushing'
    df = df.copy()
    
    df['x'] = df['x'] / 120.0
    df['y'] = df['y'] / 53.3
    df['s'] = df['s'] / 9.99 # observed max speed in total dataset, players only
    df['a'] = df['a'] / 9.99 # observed max acceleration in total dataset, players only
    df['dis'] = df['dis'] / 6.29 # observed max displacement in total dataset, players only
    
    df['sin_o'] = np.sin(np.deg2rad(df['o']))      # orientation
    df['cos_o'] = np.cos(np.deg2rad(df['o']))

    df['sin_dir'] = np.sin(np.deg2rad(df['dir']))  # direction of motion
    df['cos_dir'] = np.cos(np.deg2rad(df['dir']))
    
    feats = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    labels = torch.tensor(df[label_col].values, dtype=torch.float32)

    seq_len = df['frameId'].nunique()

    feats = feats.reshape(seq_len, 23 * len(feat_cols))
    labels = labels.reshape(seq_len, -1)
    return feats, labels


def split_dataset(data_dict, train_ratio=0.9, seed=42):
    """
    This function splits the dictionary of data up into train, val, and test sets.
    
    Args:
        data_dict (Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]):
            This input dictionary should contain 9 entries, 1 for each of the 9 weeks from
            the NGS tracking data. Each of the values associated with these 9 keys should be
            a dictionary itself. The keys in that dictionary are "gpid" strings: 'gameId-playId'.
            The values for those entries should be tuples of torch.Tensor objects. The first tensor
            is a feature tensor in the shape of [num_frames, feats_per_player*23]. The second tensor
            in that tuple is a label tensor in the shape of [num_frames, 23].
        
    Returns:
        Tuple containing:
            - train_set (Dict[str, Dict[str, Tuple[Tensor, Tensor]]])
            - val_set (Dict[str, Dict[str, Tuple[Tensor, Tensor]]])
            - test_set (Dict[str, Dict[str, Tuple[Tensor, Tensor]]])
    """
    random.seed(seed)
    
    keys = sorted(data_dict.keys())  # consistent ordering
    assert len(keys) >= 9, "Expected at least 9 groups"

    train_set = {}
    val_set = {}

    # Use only the first 8 groups for train/val
    for group_key in keys[:8]:
        group_data = list(data_dict[group_key].items())
        random.shuffle(group_data)

        split_idx = int(len(group_data) * train_ratio)
        train_set[group_key] = dict(group_data[:split_idx])
        val_set[group_key] = dict(group_data[split_idx:])

    # Group 9 is the test set
    test_key = keys[8]
    test_set = {test_key: data_dict[test_key]}

    return train_set, val_set, test_set
    

def main():
    conn = sqlite3.connect("/scratch/jts75596/fb/data/nfldata.db")

    week_dict = {}
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

        grouped_plays = df.groupby('gpid', sort=False)

        play_dict = {}
        print("Iterating over plays individually to organize them...")

        for gpid, play_df in tqdm(grouped_plays, total=len(grouped_plays)):
            try:
                play_df = compute_football_dir_and_o_in_play(play_df)

                snap_frame = play_df.loc[play_df["frameType"] == "SNAP", "frameId"].unique()[0]
                play_df = play_df[play_df['frameId'] <= (snap_frame + 15)]

                features, labels = organize_play(play_df)
            except Exception as e:
                print(f"GPID: {gpid} - {e}")
                continue
            if torch.isnan(features).any() or torch.isnan(labels).any():
                print(f"Found missing values in features or labels, not adding to dataset, check play {gpid}")
                continue
            play_dict[gpid] = (features, labels)
        week_dict[f"week_{i}"] = play_dict
    
    # Do train-val-test split
    train, val, test = split_dataset(week_dict)
    
    save_dir = "../data/processed_data"
    if not os.path.exists(save_dir):
        print(f"Making save directory {save_dir}!")
        os.makedirs(save_dir)
        
    print(f"Saving to directory {save_dir}!")
    
    torch.save(train, f"{save_dir}/train_2.pt")
    torch.save(val, f"{save_dir}/val_2.pt")
    torch.save(test, f"{save_dir}/test_2.pt")

if __name__ == "__main__":
    main()