import sqlite3
import pandas as pd
from tqdm import tqdm


def main():
    conn = sqlite3.connect("../data/nfldata.db")
    
    for i in tqdm(range(1, 10), total=9):
        table = f"tr_week{i}"
        print(f"Working on table: {table}")
        print()
        
        conn.execute(f"DROP TABLE IF EXISTS rush_labels_{i};")

        query = f"""
        CREATE TABLE rush_labels_{i} AS
        WITH 
            -- A. Label offense vs. defense and create gpid
            off_def AS (
                SELECT
                    t.*,
                    p.playDescription,
                    CASE
                        WHEN t.club = p.possessionTeam THEN 1 ELSE 0
                    END AS on_offense,
                    CAST(t.gameId AS TEXT) || "-" || CAST(t.playId AS TEXT) AS gpid
                FROM {table}       AS t
                JOIN plays          AS p
                    ON t.gameId = p.gameId
                    AND t.playId = p.playId
            )
            
        SELECT
            off_def.*,
            pp.wasInitialPassRusher AS is_rushing,
            pl.position
        FROM off_def
        LEFT JOIN player_play AS pp
            ON off_def.gameId = pp.gameId
            AND off_def.playId = pp.playId
            AND off_def.nflId = pp.nflId
        LEFT JOIN players AS pl
            ON off_def.nflId = pl.nflId;
        """
        conn.execute(query)
        conn.commit()
    conn.close()
    print("\n\nDONE!")


if __name__ == "__main__":
    main()