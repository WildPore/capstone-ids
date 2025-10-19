import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, Union, List


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Union[str, List[str]],
    label_col: str = "Label",
    figsize: Optional[tuple] = None,
    save_path: str = "feature_violin_plots.png",
    max_classes: Optional[int] = None,
) -> dict:
    """
    Create violin plots comparing feature distributions across traffic classes.

    Args:
        df: DataFrame containing the features and labels
        features: Single feature name or list of feature names to plot
        label_col: Name of the column containing class labels
        figsize: Figure size (width, height). Auto-calculated if None
        save_path: Path to save the output plot
        max_classes: Maximum number of classes to include (uses top N by count)

    Returns:
        Dictionary containing statistics about the distributions
    """
    if isinstance(features, str):
        features = [features]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    df_plot = df.copy()
    if max_classes is not None:
        top_classes = df[label_col].value_counts().head(max_classes).index
        df_plot = df_plot[df_plot[label_col].isin(top_classes)]

    if figsize is None:
        n_features = len(features)
        figsize = (14, max(6, 5 * n_features))

    n_plots = len(features)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    stats = {}

    for idx, feature in enumerate(features):
        ax = axes[idx]

        plot_df = df_plot[[feature, label_col]].dropna()

        sns.violinplot(
            data=plot_df,
            x=label_col,
            y=feature,
            hue=label_col,
            ax=ax,
            palette="Set2",
            inner="quartile",
            legend=False,
        )

        ax.set_title(
            f"Distribution of {feature} Across Traffic Classes",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Traffic Class", fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        ax.tick_params(axis="x", rotation=45)

        feature_stats = {}
        for class_label in plot_df[label_col].unique():
            class_data = plot_df[plot_df[label_col] == class_label][feature]
            feature_stats[str(class_label)] = {
                "mean": float(class_data.mean()),
                "median": float(class_data.median()),
                "std": float(class_data.std()),
                "min": float(class_data.min()),
                "max": float(class_data.max()),
                "q25": float(class_data.quantile(0.25)),
                "q75": float(class_data.quantile(0.75)),
            }

        stats[feature] = feature_stats

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return stats


def plot_multi_feature_comparison(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "Label",
    figsize: tuple = (16, 10),
    save_path: str = "multi_feature_violin_grid.png",
) -> None:
    """
    Create a grid of violin plots for multiple features side-by-side.

    Args:
        df: DataFrame containing the features and labels
        features: List of feature names to plot
        label_col: Name of the column containing class labels
        figsize: Figure size (width, height)
        save_path: Path to save the output plot
    """
    df_melted = df[features + [label_col]].melt(
        id_vars=[label_col], value_vars=features, var_name="Feature", value_name="Value"
    )

    g = sns.FacetGrid(
        df_melted,
        col="Feature",
        col_wrap=min(3, len(features)),
        height=4,
        aspect=1.2,
        sharey=False,
    )

    g.map_dataframe(
        sns.violinplot,
        x=label_col,
        y="Value",
        hue=label_col,
        palette="Set2",
        inner="quartile",
        legend=False,
    )

    g.set_titles("{col_name}", fontsize=12, fontweight="bold")
    g.set_axis_labels("Traffic Class", "Value", fontsize=10)

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
