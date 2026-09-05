from pathlib import Path
from pandas import read_csv, DataFrame

from ..constants import PACKAGE_ROOT


def load_csv_to_df(csv_file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    If the provided path is a DataFrame, returns it directly.
    If the path is not found, falls back to resolving relative to PACKAGE_ROOT/datasets or PACKAGE_ROOT.
    """
    if isinstance(csv_file_path, DataFrame):
        return csv_file_path

    path = Path(csv_file_path)

    if not path.is_file():
        # Fallback to resolving relative to the repository/package root
        candidate = PACKAGE_ROOT / "datasets" / path.name
        if candidate.is_file():
            path = candidate
        elif (PACKAGE_ROOT / path).is_file():
            path = PACKAGE_ROOT / path

    return read_csv(path)
