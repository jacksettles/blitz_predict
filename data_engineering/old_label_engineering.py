import sqlite3
import pandas as pd
from tqdm import tqdm


def main():
    conn = sqlite3.connect("../data/nfldata.db")

    K = 15
    accel_threshold = 1.0
    

    for i in tqdm(range(1, 10), total=9):
        table = f"tr_week{i}"
        print(f"Working on table: {table}")
        print()

        conn.execute(f"DROP TABLE IF EXISTS rush_labels_{i};")
        query = f"""
        CREATE TABLE rush_labels_{i} AS
        WITH 
            -- 0. Forgot to label offense vs. defense frames
            off_def AS (
                SELECT
                    t.*,
                    CASE
                        WHEN t.club = p.possessionTeam
                        THEN 1
                        ELSE 0
                    END AS on_offense,
                    CAST(t.gameId AS TEXT)
                        || "-"
                        || CAST(t.playId AS TEXT)
                    AS gpid
                FROM {table}       AS t
                JOIN plays          AS p
                    ON t.gameId = p.gameId
                    AND t.playId = p.playId
            ),

            -- A. Find the frame the ball is snapped on for every play
            snap AS (
                SELECT DISTINCT 
                    gameId, 
                    playId, 
                    frameId AS snap_frame
                FROM off_def
                WHERE frameType = 'SNAP'
            ),

            -- B. Find football's starting X coordinate
            ball_start AS (
                SELECT
                    t.gameId,
                    t.playId,
                    t.x AS ball_x
                FROM off_def    AS t
                JOIN snap       AS s
                    ON t.gameId = s.gameId
                    AND t.playId = s.playId
                    WHERE t.frameId = s.snap_frame
                    AND t.displayName = 'football'
            ),

            -- C. For each defender at snap: their start_x and pursuit_dir
            start_pos AS (
                SELECT
                    t.gameId,
                    t.playId,
                    t.nflId,
                    t.x         AS start_x,
                    CASE WHEN t.x < b.ball_x THEN 1 ELSE -1 END AS pursuit_dir,
                    s.snap_frame
                FROM off_def        AS t
                JOIN snap           AS s
                ON t.gameId = s.gameId
                AND t.playId = s.playId
                JOIN ball_start     AS b
                ON t.gameId = b.gameId
                AND t.playId = b.playId
                WHERE t.frameId = s.snap_frame
                AND t.displayName != 'football'
                AND t.on_offense = 0
            ),

            -- D. Compute player's displacement over the first {K} frames
            defender_features AS (
                SELECT
                    sp.gameId,
                    sp.playId,
                    sp.nflId,
                    (fk.x - sp.start_x) * sp.pursuit_dir    AS delta_x,
                    AVG(t.a)                                AS avg_accel
                FROM start_pos      AS sp

                JOIN off_def        AS fk
                ON fk.gameId = sp.gameId
                AND fk.playId = sp.playId
                AND fk.nflId = sp.nflId
                AND fk.frameId = sp.snap_frame + {K}

                JOIN off_def        AS t
                ON t.gameId = sp.gameId
                AND t.playId = sp.playId
                AND t.nflId = sp.nflId
                AND t.frameId > sp.snap_frame
                AND t.frameId <= sp.snap_frame + {K}

                GROUP BY sp.gameId, sp.playId, sp.nflId
            ),

            -- E. Final classification: if they moved toward the line AND above accel threshold
            rush_labels AS (
                SELECT
                    gameId,
                    playId,
                    nflId,
                    CASE
                        WHEN delta_x > 0
                        AND avg_accel > {accel_threshold}
                        THEN 1
                        ELSE 0
                    END AS is_rushing
                FROM defender_features
            )

        -- F. Join the labels back to every row of off_def
        SELECT
            od.*,
            rl.is_rushing,
            pl.position
        FROM off_def AS od
        LEFT JOIN rush_labels AS rl
            USING (gameId, playId, nflId)
        LEFT JOIN players AS pl
            ON od.nflId = pl.nflId;
        """
        conn.execute(query)
        conn.commit()
    conn.close()
    print("\n\nDONE!")


if __name__ == "__main__":
    main()