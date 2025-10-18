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

data_path = get_project_root() / "data"
file_pattern = str(data_path) + "/*.csv"

csv_files = glob.glob(file_pattern)

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("Infinite values per column:")
print(df.isin([np.inf, -np.inf]).sum())
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# df.info()
# df.describe()
print("\nMissing values:")
print(df.isnull().sum())

print("\n")
print(df.describe())

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(figsize=(20, 15), bins=50)
plt.tight_layout()
plt.savefig("distributions.png")
plt.close()
