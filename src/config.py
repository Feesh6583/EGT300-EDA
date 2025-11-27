import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "..", "data", "bmarket_clean_final.db")

TARGET_COLUMN = "SubBin"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paths for saving models
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
