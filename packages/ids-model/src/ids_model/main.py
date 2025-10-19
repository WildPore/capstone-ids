import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from capstone_ids.utils import get_project_root
from ids_model.heatmap import create_correlation_heatmap

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

X = X.select_dtypes(include=[np.number])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# model = XGBClassifier(
#     objective="multi:softprob",
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     eval_metric="mlogloss",
#     random_state=42,
#     n_jobs=-1,
# )

# model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# y_pred = model.predict(X_test)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=le.classes_))
