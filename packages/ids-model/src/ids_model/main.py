import glob

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


from capstone_ids.utils import get_project_root


data_path = get_project_root() / "data"
file_pattern = str(data_path) + "/*.csv"

csv_files = glob.glob(file_pattern)

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"Loaded {file}: {df.shape}")

df_combined = pd.concat(dfs, ignore_index=True)
print(f"Combinaed shape: {df_combined.shape}")

print(df_combined["Label"].value_counts())
