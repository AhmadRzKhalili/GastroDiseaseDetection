import pprint

# pyrefly: ignore [missing-import]
from src import (
    analyse_table_features,
    analyse_numeric_features_range,
    calculate_feature_variance,
    explore_correlation,
    analyse_feature_dependency,
    calculate_vif,
    calculate_cramers_v,
    calculate_mutual_information,
)

if __name__ == "__main__":
    dataset_path = "../../datasets/gastrointestinal_disease_dataset.csv"

    print("=== 1. Feature Data Types ===")
    pprint.pp(analyse_table_features(dataset_path))

    print("\n=== 2. Feature Variance (Normalized & Encoded) ===")
    pprint.pp(
        calculate_feature_variance(
            dataset_path,
            encode_non_numeric=True,
            normalize=True,
        )
    )

    print("\n=== 3. Multicollinearity Analysis (VIF) ===")
    vif_scores = calculate_vif(dataset_path, encode_non_numeric=True, plot_matrix=True)
    pprint.pp(vif_scores.sort_values(ascending=False).head(10))

    print("\n=== 4. Categorical Feature Association (Cramér's V) ===")
    cramers_matrix = calculate_cramers_v(dataset_path, encode_non_numeric=False, plot_matrix=True)
    pprint.pp(cramers_matrix)

    print("\n=== 5. Pairwise Mutual Information (NMI) ===")
    nmi_matrix = calculate_mutual_information(dataset_path, encode_non_numeric=True, plot_matrix=True)
    pprint.pp(nmi_matrix.iloc[:5, :5])

    print("\n=== 6. Comprehensive Feature Dependency Suite ===")
    all_dependencies = analyse_feature_dependency(
        dataset_path,
        method="all",
        encode_non_numeric=True,
        plot_matrix=True
    )
    print("Dependency Analysis Completed. Analysis components:", list(all_dependencies.keys()))