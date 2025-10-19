import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def get_top_correlated_features(df: pd.DataFrame, label_col: str, n: int = 20):
    numeric_df = df.select_dtypes(include=["number"])

    # Remove constant features (zero variance)
    non_constant_features = numeric_df.columns[numeric_df.std() != 0]
    numeric_df = numeric_df[non_constant_features]

    le = LabelEncoder()
    encoded_labels = le.fit_transform(df[label_col])

    correlations_with_target = numeric_df.corrwith(pd.Series(encoded_labels)).abs()

    # Drop NaN values that might result from correlations
    correlations_with_target = correlations_with_target.dropna()

    top_features = (
        correlations_with_target.sort_values(ascending=False).head(n).index.tolist()
    )

    return top_features


def create_correlation_heatmap(
    df: pd.DataFrame,
    label_col: str = "Label",
    top_n: int = 20,
    output_path: str = "correlation_heatmap.png",
):
    top_features = get_top_correlated_features(df, label_col, top_n)
    numeric_df = df[top_features]

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
    )

    plt.title(f"Feature Correlation Heatmap (Top {top_n} Features)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
