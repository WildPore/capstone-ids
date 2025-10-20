import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load all CSV files from the specified directory and concatenate them.

    Args:
        data_path: Path to the directory containing CSV files

    Returns:
        Concatenated DataFrame from all CSV files
    """
    file_pattern = str(data_path) + "/*.csv"
    csv_files = glob.glob(file_pattern)

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data.

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
