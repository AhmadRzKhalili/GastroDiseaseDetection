import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..utils import load_csv_to_df
from .cleanup_tabular_data import encode_non_numeric_features


def plot_dependency_matrix(
    matrix,
    title: str = "Feature Dependency Matrix",
    cmap: str = "viridis",
    annot: bool = False,
    figsize: tuple = (12, 10),
    show_plot: bool = True,
    save_path=None,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """Plots a heatmap for a dependency matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        import seaborn as sns

        sns.heatmap(
            matrix,
            annot=annot,
            cmap=cmap,
            fmt=".2f",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Dependency Score"},
        )
    except Exception:
        cax = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(cax, ax=ax, label="Dependency Score")
        cols = matrix.columns
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticklabels(cols)

        if annot:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = matrix.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white" if val > (vmin + vmax) / 2 else "black",
                        fontsize=7,
                    )

    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig, ax


def plot_vif_scores(
    vif_series,
    title: str = "Variance Inflation Factor (VIF)",
    figsize: tuple = (10, 8),
    show_plot: bool = True,
    save_path=None,
):
    """Plots a horizontal bar chart of VIF scores."""
    fig, ax = plt.subplots(figsize=figsize)
    sorted_vif = vif_series.sort_values(ascending=True)
    ax.barh(sorted_vif.index, sorted_vif.values, color="skyblue")
    ax.set_xlabel("VIF Score")
    ax.set_title(title, fontsize=14, pad=12)
    ax.axvline(x=5, color="orange", linestyle="--", label="VIF = 5 Threshold")
    ax.axvline(x=10, color="red", linestyle="--", label="VIF = 10 Threshold")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig, ax


def calculate_cramers_v(
    csv_file_path,
    encode_non_numeric: bool = False,
    show_plot: bool = False,
    plot_matrix: bool = False,
    annot: bool = False,
    cmap: str = "viridis",
    figsize: tuple = (12, 10),
    save_path=None,
):
    """
    Calculates Cramér's V matrix to measure pairwise association between categorical features.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.
    encode_non_numeric : bool, default False
        Whether to encode non-numeric columns into category codes before calculation.
    show_plot : bool, default False
        Whether to display the heatmap plot.
    plot_matrix : bool, default False
        Alias for show_plot.
    annot : bool, default False
        Whether to annotate matrix cell values.
    cmap : str, default 'viridis'
        Colormap for heatmap.
    figsize : tuple, default (12, 10)
        Figure size.
    save_path : str or Path, optional
        File path to save the generated figure.

    Returns:
    --------
    DataFrame
        Cramér's V matrix with values between 0 (independent) and 1 (perfectly associated).
    """
    df = load_csv_to_df(csv_file_path)

    if encode_non_numeric:
        df = encode_non_numeric_features(df)
        cols = df.columns
    else:
        cols = df.select_dtypes(exclude="number").columns
        if len(cols) == 0:
            cols = df.columns

    n_cols = len(cols)
    matrix = pd.DataFrame(np.ones((n_cols, n_cols)), index=cols, columns=cols)

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1, col2 = cols[i], cols[j]
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.values.sum()
            if n <= 1:
                val = 0.0
            else:
                phi2 = chi2 / n
                r, k = confusion_matrix.shape
                phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                rcorr = r - ((r - 1) ** 2) / (n - 1)
                kcorr = k - ((k - 1) ** 2) / (n - 1)
                min_dim = min((kcorr - 1), (kcorr - 1))
                if min_dim <= 0:
                    val = 0.0
                else:
                    val = float(np.sqrt(phi2corr / min_dim))
            matrix.loc[col1, col2] = val
            matrix.loc[col2, col1] = val

    if show_plot or plot_matrix or save_path:
        plot_dependency_matrix(
            matrix,
            title="Cramér's V Categorical Feature Association",
            cmap=cmap,
            annot=annot,
            figsize=figsize,
            show_plot=show_plot or plot_matrix,
            save_path=save_path,
        )

    return matrix


def calculate_vif(
    csv_file_path,
    encode_non_numeric: bool = True,
    show_plot: bool = False,
    plot_matrix: bool = False,
    figsize: tuple = (10, 8),
    save_path=None,
):
    """
    Calculates Variance Inflation Factor (VIF) for features to detect multicollinearity.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.
    encode_non_numeric : bool, default True
        Whether to encode non-numeric features into numeric codes before calculating VIF.
    show_plot : bool, default False
        Whether to display the bar chart plot of VIF scores.
    plot_matrix : bool, default False
        Alias for show_plot.
    figsize : tuple, default (10, 8)
        Figure size.
    save_path : str or Path, optional
        File path to save figure.

    Returns:
    --------
    Series
        Series mapping each feature to its VIF score.
    """
    df = load_csv_to_df(csv_file_path)

    if encode_non_numeric:
        df = encode_non_numeric_features(df)

    numeric_df = df.select_dtypes(include="number").dropna()
    corr_matrix = numeric_df.corr().values

    inv_corr = np.linalg.pinv(corr_matrix)
    vif_values = np.diag(inv_corr)

    vif_series = pd.Series(vif_values, index=numeric_df.columns, name="VIF")

    if show_plot or plot_matrix or save_path:
        plot_vif_scores(
            vif_series,
            title="Variance Inflation Factor (VIF) Multicollinearity",
            figsize=figsize,
            show_plot=show_plot or plot_matrix,
            save_path=save_path,
        )

    return vif_series


def calculate_mutual_information(
    csv_file_path,
    encode_non_numeric: bool = True,
    bins: int = 10,
    show_plot: bool = False,
    plot_matrix: bool = False,
    annot: bool = False,
    cmap: str = "viridis",
    figsize: tuple = (12, 10),
    save_path=None,
):
    """
    Calculates Pairwise Normalized Mutual Information (NMI) matrix across features.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.
    encode_non_numeric : bool, default True
        Whether to encode non-numeric features into numeric codes.
    bins : int, default 10
        Number of bins for continuous numerical features.
    show_plot : bool, default False
        Whether to display the NMI matrix plot.
    plot_matrix : bool, default False
        Alias for show_plot.
    annot : bool, default False
        Whether to annotate cell values.
    cmap : str, default 'viridis'
        Colormap for heatmap.
    figsize : tuple, default (12, 10)
        Figure size.
    save_path : str or Path, optional
        Path to save figure.

    Returns:
    --------
    DataFrame
        NMI matrix with values between 0 and 1.
    """
    df = load_csv_to_df(csv_file_path)

    if encode_non_numeric:
        df = encode_non_numeric_features(df)

    cols = df.columns
    n_cols = len(cols)
    matrix = pd.DataFrame(np.ones((n_cols, n_cols)), index=cols, columns=cols)

    binned_df = df.copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(binned_df[col]) and binned_df[col].nunique() > bins:
            binned_df[col] = pd.qcut(binned_df[col], q=bins, duplicates="drop")

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1, col2 = cols[i], cols[j]
            contingency = pd.crosstab(binned_df[col1], binned_df[col2])
            total = contingency.values.sum()
            if total == 0:
                nmi_val = 0.0
            else:
                px = contingency.values.sum(axis=1) / total
                py = contingency.values.sum(axis=0) / total
                pxy = contingency.values / total

                hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
                hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

                nonzero = pxy > 0
                hxy = -np.sum(pxy[nonzero] * np.log(pxy[nonzero]))

                mi = max(0.0, hx + hy - hxy)
                denom = (hx + hy) / 2.0
                nmi_val = float(mi / denom) if denom > 0 else 0.0

            matrix.loc[col1, col2] = nmi_val
            matrix.loc[col2, col1] = nmi_val

    if show_plot or plot_matrix or save_path:
        plot_dependency_matrix(
            matrix,
            title="Normalized Mutual Information (NMI) Matrix",
            cmap=cmap,
            annot=annot,
            figsize=figsize,
            show_plot=show_plot or plot_matrix,
            save_path=save_path,
        )

    return matrix


def analyse_feature_dependency(
    csv_file_path,
    method: str = "vif",
    encode_non_numeric: bool = True,
    bins: int = 10,
    show_plot: bool = False,
    plot_matrix: bool = False,
    annot: bool = False,
    cmap: str = "viridis",
    figsize: tuple = (12, 10),
    save_path=None,
):
    """
    Analyzes feature dependency across a dataset with optional plotting.

    Parameters:
    -----------
    csv_file_path : str, Path, or DataFrame
        Path to the CSV dataset or an existing pandas DataFrame.
    method : {'vif', 'cramers_v', 'mutual_info', 'all'}, default 'vif'
        Method of dependency analysis.
    encode_non_numeric : bool, default True
        Whether to encode non-numeric features into numeric values.
    bins : int, default 10
        Number of bins for continuous variables when computing mutual info.
    show_plot : bool, default False
        Whether to display plots for the analysis.
    plot_matrix : bool, default False
        Alias for show_plot.
    annot : bool, default False
        Whether to annotate heatmaps.
    cmap : str, default 'viridis'
        Heatmap colormap.
    figsize : tuple, default (12, 10)
        Figure dimensions.
    save_path : str or Path, optional
        Path to save figure.

    Returns:
    --------
    DataFrame, Series, or dict
        Result of requested feature dependency analysis.
    """
    method = method.lower()
    plot_flag = show_plot or plot_matrix

    if method == "vif":
        return calculate_vif(
            csv_file_path,
            encode_non_numeric=encode_non_numeric,
            show_plot=plot_flag,
            figsize=figsize,
            save_path=save_path,
        )
    elif method in ("cramers_v", "cramer_v", "cramers"):
        return calculate_cramers_v(
            csv_file_path,
            encode_non_numeric=encode_non_numeric,
            show_plot=plot_flag,
            annot=annot,
            cmap=cmap,
            figsize=figsize,
            save_path=save_path,
        )
    elif method in ("mutual_info", "mi", "nmi"):
        return calculate_mutual_information(
            csv_file_path,
            encode_non_numeric=encode_non_numeric,
            bins=bins,
            show_plot=plot_flag,
            annot=annot,
            cmap=cmap,
            figsize=figsize,
            save_path=save_path,
        )
    elif method == "all":
        vif_res = calculate_vif(
            csv_file_path,
            encode_non_numeric=encode_non_numeric,
            show_plot=plot_flag,
            save_path=save_path,
        )
        cramers_res = calculate_cramers_v(
            csv_file_path,
            encode_non_numeric=encode_non_numeric,
            show_plot=plot_flag,
            annot=annot,
            cmap=cmap,
            save_path=save_path,
        )
        nmi_res = calculate_mutual_information(
            csv_file_path,
            encode_non_numeric=encode_non_numeric,
            bins=bins,
            show_plot=plot_flag,
            annot=annot,
            cmap=cmap,
            save_path=save_path,
        )
        return {
            "vif": vif_res,
            "cramers_v": cramers_res,
            "mutual_info": nmi_res,
        }
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'vif', 'cramers_v', 'mutual_info', or 'all'."
        )


explore_feature_dependency = analyse_feature_dependency
