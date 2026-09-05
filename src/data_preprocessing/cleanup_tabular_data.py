from ..utils import load_csv_to_df


def get_rows_with_missing_data(csv_file_path):
    df = load_csv_to_df(csv_file_path)
    return df[df.isna().any(axis=1)]