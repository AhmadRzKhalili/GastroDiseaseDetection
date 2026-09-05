import pprint

# pyrefly: ignore [missing-import]
from src import explore_correlation, analyse_table_features

if __name__ == "__main__":
    # pprint.pp(analyse_table_features("../../datasets/gastrointestinal_disease_dataset.csv"))
    corr_matrix = explore_correlation(
        "../../datasets/gastrointestinal_disease_dataset.csv",
        method="pearson",
        encode_non_numeric=True
    )