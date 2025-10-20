from capstone_ids.utils import get_project_root
from ids_model.data_loader import load_data, preprocess_data
from ids_model.feature_engineering import (
    encode_labels,
    extract_features_labels,
    select_numeric_features,
)
from ids_model.model_trainer import (
    create_model,
    evaluate_model,
    save_model,
    split_data,
    train_model,
)
from ids_model.visualization import generate_all_visualizations


def main():
    """
    Main workflow for IDS model training and evaluation.
    """
    data_path = get_project_root() / "data/MachineLearningCVE"
    df = load_data(data_path)
    df = preprocess_data(df)

    X, y = extract_features_labels(df, label_col="Label")
    generate_all_visualizations(
        df, y, label_col="Label", n_top_features=10, n_violin_features=5, max_classes=10
    )

    X = select_numeric_features(X)
    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = split_data(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = create_model(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        device="cuda",
    )
    model = train_model(model, X_train, y_train, X_test, y_test, verbose=True)

    evaluate_model(model, X_test, y_test, label_encoder)

    # Save the trained model
    model_path = get_project_root() / "models" / "ids_model.ubj"
    save_model(model, model_path)


if __name__ == "__main__":
    main()
