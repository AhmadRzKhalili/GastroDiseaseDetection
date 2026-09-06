from ..utils import load_csv_to_df


def analyse_table_features(csv_file_path):

    df = load_csv_to_df(csv_file_path)

    return df.columns.to_series().groupby(df.dtypes).groups


def analyse_numeric_features_range(csv_file_path):

    df = load_csv_to_df(csv_file_path)

    numeric_cols = df.select_dtypes(include='number').columns
    ranges = {col: (float(df[col].min()), float(df[col].max())) for col in numeric_cols}

    return ranges


def analyse_feature_variance(
    csv_file_path,
    encode_non_numeric: bool = False,
    normalize: bool = False,
    **kwargs,
):
    """
    Calculates the variance for features in a dataset.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.
    encode_non_numeric : bool, default False
        Whether to encode non-numeric features into numeric values before calculating variance.
    normalize : bool, default False
        Whether to normalize (min-max scale between 0 and 1) numeric features before calculating variance.

    Returns:
    --------
    dict
        Dictionary mapping feature names to their calculated variance.
    """
    if "normalise" in kwargs:
        normalize = kwargs.pop("normalise")

    df = load_csv_to_df(csv_file_path)

    if encode_non_numeric:
        from .cleanup_tabular_data import encode_non_numeric_features
        df = encode_non_numeric_features(df)

    numeric_cols = df.select_dtypes(include='number').columns

    if normalize:
        df = df.copy()
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0.0

    variances = {col: float(df[col].var()) for col in numeric_cols}

    return variances


calculate_feature_variance = analyse_feature_variance