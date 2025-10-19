import glob

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from capstone_ids.utils import get_project_root
from ids_model.barplot import plot_class_distribution
from ids_model.heatmap import create_correlation_heatmap, get_top_correlated_features
from ids_model.violin_plot import plot_feature_distributions

data_path = get_project_root() / "data"
file_pattern = str(data_path) + "/*.csv"

csv_files = glob.glob(file_pattern)

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

create_correlation_heatmap(df)

X = df.drop(["Label"], axis=1)
y = df["Label"]

plot_class_distribution(y)


top_features = get_top_correlated_features(df, "Label", n=10)
print("\nGenerating violin plots for top features...")
feature_stats = plot_feature_distributions(
    df,
    features=top_features[:5],
    label_col="Label",
    save_path="feature_distributions.png",
    max_classes=10,
)

print("\nFeature distribution statistics:")
for feature, stats in feature_stats.items():
    print(f"\n{feature}:")
    for class_label, class_stats in stats.items():
        print(
            f" {class_label}: mean={class_stats['mean']:.4f}, median={class_stats['median']:.4f}"
        )


X = X.select_dtypes(include=[np.number])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


model = XGBClassifier(
    objective="multi:softprob",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
