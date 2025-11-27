import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    # Replace None with "unknown"
    df = df.fillna("unknown")

    # Columns that must NOT be encoded
    do_not_encode = ["Client ID", "SubBin"]

    # Label encode all object columns except Client ID
    for col in df.select_dtypes(include="object"):
        if col not in do_not_encode:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df
