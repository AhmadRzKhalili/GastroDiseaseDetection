import matplotlib.pyplot as plt
from pandas import DataFrame

from ..utils import load_csv_to_df


def explore_correlation(
    csv_file_path,
    method: str = "pearson",
    annot: bool = False,
    cmap: str = "coolwarm",
    figsize: tuple = (12, 10),
    title: str = "Feature Correlation Matrix",
    show_plot: bool = True,
    save_path=None,
):
    """
    Computes and plots the correlation matrix for numerical features in a dataset.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.
    method : {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Method of correlation computation.
    annot : bool, default False
        Whether to annotate correlation coefficients inside each heatmap cell.
    cmap : str, default 'coolwarm'
        Colormap used for the heatmap.
    figsize : tuple, default (12, 10)
        Dimensions of the matplotlib figure.
    title : str, default 'Feature Correlation Matrix'
        Title of the correlation plot.
    show_plot : bool, default True
        Whether to display the plot interactively using plt.show().
    save_path : str or Path, optional
        Path where the generated figure should be saved.

    Returns:
    --------
    DataFrame
        Correlation matrix of the numeric features.
    """
    df = load_csv_to_df(csv_file_path)

    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns
        sns.heatmap(
            corr_matrix,
            annot=annot,
            cmap=cmap,
            fmt=".2f",
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
            cbar_kws={"label": "Correlation Coefficient"},
        )
    except Exception:
        cax = ax.imshow(corr_matrix, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        fig.colorbar(cax, ax=ax, label="Correlation Coefficient")
        cols = corr_matrix.columns
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticklabels(cols)

        if annot:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = corr_matrix.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="black" if -0.5 < val < 0.5 else "white",
                        fontsize=7,
                    )

    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()

    return corr_matrix


# Convenient aliases
analyse_feature_correlation = explore_correlation
explore_feature_correlation = explore_correlation
