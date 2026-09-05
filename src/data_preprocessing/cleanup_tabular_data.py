from ..utils import load_csv_to_df


def get_rows_with_missing_data(csv_file_path):
    df = load_csv_to_df(csv_file_path)
    return df[df.isna().any(axis=1)]


def encode_non_numeric_features(csv_file_path):
    """
    Encodes non-numeric (categorical/object/string) columns into numeric values using category codes.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.

    Returns:
    --------
    DataFrame
        DataFrame with non-numeric features converted to numeric values.
    """
    df = load_csv_to_df(csv_file_path).copy()
    non_numeric_cols = df.select_dtypes(exclude="number").columns

    for col in non_numeric_cols:
        cat = df[col].astype("category")
        codes = cat.cat.codes
        if df[col].isna().any():
            df[col] = codes.where(codes != -1, float("nan"))
        else:
            df[col] = codes
    print(df)
    return df


encode_categorical_features = encode_non_numeric_features
