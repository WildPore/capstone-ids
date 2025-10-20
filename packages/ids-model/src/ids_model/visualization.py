import pandas as pd

from ids_model.barplot import plot_class_distribution
from ids_model.heatmap import create_correlation_heatmap, get_top_correlated_features
from ids_model.violin_plot import plot_feature_distributions


def generate_all_visualizations(
    df: pd.DataFrame,
    y: pd.Series,
    label_col: str = "Label",
    n_top_features: int = 10,
    n_violin_features: int = 5,
    max_classes: int = 10,
) -> dict:
    """
    Generate all visualizations for the dataset.

    Args:
        df: Full DataFrame including features and labels
        y: Labels Series
        label_col: Name of the label column
        n_top_features: Number of top correlated features to identify
        n_violin_features: Number of features to plot in violin plots
        max_classes: Maximum number of classes to display in violin plots

    Returns:
        Dictionary containing feature statistics from violin plots
    """
    create_correlation_heatmap(df)
    plot_class_distribution(y)

    top_features = get_top_correlated_features(df, label_col, n=n_top_features)
    print("\nGenerating violin plots for top features...")
    feature_stats = plot_feature_distributions(
        df,
        features=top_features[:n_violin_features],
        label_col=label_col,
        save_path="feature_distributions.png",
        max_classes=max_classes,
    )

    print("\nFeature distribution statistics:")
    for feature, stats in feature_stats.items():
        print(f"\n{feature}:")
        for class_label, class_stats in stats.items():
            print(
                f"  {class_label}: mean={class_stats['mean']:.4f}, median={class_stats['median']:.4f}"
            )

    return feature_stats
