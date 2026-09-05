from ..utils import load_csv_to_df


def analyse_table_features(csv_file_path):

    df = load_csv_to_df(csv_file_path)

    return df.columns.to_series().groupby(df.dtypes).groups


def analyse_numeric_features_range(csv_file_path):

    df = load_csv_to_df(csv_file_path)

    numeric_cols = df.select_dtypes(include='number').columns
    ranges = {col: (float(df[col].min()), float(df[col].max())) for col in numeric_cols}

    return ranges