from pathlib import Path
from pandas import read_csv, DataFrame

from ..constants import PACKAGE_ROOT

def get_rows_with_missing_data(csv_file_path):
    if isinstance(csv_file_path, DataFrame):
        df = csv_file_path
    else:
        df = load_csv_to_df(csv_file_path)

    return df[df.isna().any(axis=1)]

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