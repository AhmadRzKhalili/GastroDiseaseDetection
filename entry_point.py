import pprint

from src import explore_correlation

if __name__ == "__main__":
    corr_matrix = explore_correlation("../../datasets/gastrointestinal_disease_dataset.csv")
    pprint.pp(corr_matrix)