# src/data_loader.py
import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "bmarket.db")

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM bank_marketing", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading database: {e}")
        raise
