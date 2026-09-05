from pathlib import Path
from pandas import read_csv

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

def get_table_features(csv_file_path):
    
    path = Path(csv_file_path)
    if not path.is_file():
        # Fallback to resolving relative to the repository/package root
        candidate = PACKAGE_ROOT / "datasets" / path.name
        if candidate.is_file():
            path = candidate
        elif (PACKAGE_ROOT / path).is_file():
            path = PACKAGE_ROOT / path

    return read_csv(path).head


if __name__ == "__main__":
    print(get_table_features("../../datasets/gastrointestinal_disease_dataset.csv"))