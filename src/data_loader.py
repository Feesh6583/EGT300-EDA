import sqlite3
import pandas as pd
from .config import DB_PATH

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            raise ValueError("❌ No tables found inside the SQLite database.")

        table_name = tables[0][0]
        print(f"📌 Loading table: {table_name}")

        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df

    except Exception as e:
        print("❌ Error loading database:", e)
        raise
