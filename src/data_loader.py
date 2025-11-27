import sqlite3
import pandas as pd
from .config import DB_PATH

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM bank_marketing_clean", conn)
        conn.close()
        print(f"📌 Loaded {len(df)} rows from {DB_PATH}")
        return df
    except Exception as e:
        print(f"Error loading database: {e}")
        raise
