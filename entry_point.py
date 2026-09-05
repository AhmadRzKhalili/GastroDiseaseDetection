import pprint

from src import analyse_table_features

if __name__ == "__main__":
    pprint.pp(analyse_table_features("../../datasets/gastrointestinal_disease_dataset.csv"))