import os
import sqlite3
import pandas as pd

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "bmarket.db")

def load_data(db_path: str = None) -> pd.DataFrame:
    """
    Load the bank marketing dataset from SQLite database.
    """
    path_to_use = db_path if db_path is not None else DEFAULT_DB_PATH
    try:
        conn = sqlite3.connect(path_to_use)
        df = pd.read_sql_query("SELECT * FROM bank_marketing", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading database from {path_to_use}: {e}")
        raise
