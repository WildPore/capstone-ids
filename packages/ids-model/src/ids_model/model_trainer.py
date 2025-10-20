from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Feature DataFrame
        y: Encoded labels array
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_model(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    device: str = "cuda"
) -> XGBClassifier:
    """
    Create and configure an XGBoost classifier.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate for boosting
        random_state: Random seed for reproducibility
        device: Device to use for training ('cuda', 'cpu', etc.)

    Returns:
        Configured XGBClassifier instance
    """
    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",
        device=device,
    )
    return model


def train_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    verbose: bool = True
) -> XGBClassifier:
    """
    Train the XGBoost model.

    Args:
        model: XGBClassifier instance to train
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        verbose: Whether to print training progress

    Returns:
        Trained model
    """
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder
) -> np.ndarray:
    """
    Evaluate the trained model and print classification report.

    Args:
        model: Trained XGBClassifier
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted LabelEncoder for class names

    Returns:
        Predicted labels array
    """
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    return y_pred


def save_model(model: XGBClassifier, save_path: Path) -> None:
    """
    Save the trained model to a file in Universal Binary JSON format.

    Args:
        model: Trained XGBClassifier to save
        save_path: Path where the model should be saved (including filename)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_path))
    print(f"\nModel saved to: {save_path}")
