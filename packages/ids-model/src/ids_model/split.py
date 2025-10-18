from sklearn.model_selection import train_test_split


def split(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
