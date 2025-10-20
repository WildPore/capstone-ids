import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def extract_features_labels(
    df: pd.DataFrame, label_col: str = "Label"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and labels from the DataFrame.

    Args:
        df: Input DataFrame
        label_col: Name of the label column

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    X = df.drop([label_col], axis=1)
    y = df[label_col]
    return X, y


def select_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Select only numeric features from the feature DataFrame.

    Args:
        X: Feature DataFrame

    Returns:
        DataFrame with only numeric columns
    """
    return X.select_dtypes(include=[np.number])


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode categorical labels to numeric values.

    Args:
        y: Series of categorical labels

    Returns:
        Tuple of (encoded labels array, fitted LabelEncoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")

    return y_encoded, le  # type: ignore
