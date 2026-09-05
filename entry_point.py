import pprint

from src import get_rows_with_missing_data

if __name__ == "__main__":
    pprint.pp(get_rows_with_missing_data("../../datasets/gastrointestinal_disease_dataset.csv"))