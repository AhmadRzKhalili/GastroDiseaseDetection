from pathlib import Path
from pandas import read_csv, DataFrame


from ..constants import PACKAGE_ROOT

def analyse_table_features(csv_file_path):

    df = load_csv_to_df(csv_file_path)
    
    return df.columns.to_series().groupby(df.dtypes).groups

def analyse_numeric_features_range(csv_file_path):

    df = load_csv_to_df(csv_file_path)
    
    numeric_cols = df.select_dtypes(include='number').columns
    ranges = {col: (float(df[col].min()), float(df[col].max())) for col in numeric_cols}

    return ranges

def load_csv_to_df(csv_file_path):

    path = Path(csv_file_path)
        
    if not path.is_file():
        # Fallback to resolving relative to the repository/package root
        candidate = PACKAGE_ROOT / "datasets" / path.name
        if candidate.is_file():
            path = candidate
        elif (PACKAGE_ROOT / path).is_file():
            path = PACKAGE_ROOT / path

    return read_csv(path)