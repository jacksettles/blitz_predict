import sqlite3
import pandas as pd
from pathlib import Path

def build_db(csv_dir, db_path):
    csv_dir = Path(csv_dir)
    db_path = Path(db_path)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    print(f"Creating database at {db_path}...")
    conn = sqlite3.connect(db_path)

    for csv_file in csv_files:
        table_name = csv_file.stem
        print(f"Inserting {csv_file.name} as table '{table_name}'...")
        df = pd.read_csv(csv_file)
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.close()
    print("✅ Done: All CSVs inserted into database.")

if __name__ == "__main__":
    build_db(csv_dir="../data/raw", db_path="../data/nfldata.db")