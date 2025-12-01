# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "..", "data", "bmarket_clean_final.db")

TARGET_COLUMN = "SubBin"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# output dirs (created automatically)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
REPORT_DIR = os.path.join(BASE_DIR, "..", "reports")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)