import pprint

from src import analyse_numeric_features_range

if __name__ == "__main__":
    pprint.pp(analyse_numeric_features_range("../../datasets/gastrointestinal_disease_dataset.csv"))