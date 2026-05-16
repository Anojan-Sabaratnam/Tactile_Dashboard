"""Load and merge the BioTac SP Grasp Stability CSV files.

The BioTac SP dataset (Garcia-Garcia et al., 2019) contains tactile
readings from 3 BioTac SP sensors mounted on a Shadow Dexterous Hand.
Each sample has 72 features (3 fingers x 24 taxels) plus an object
label and a binary stability outcome.

Dataset source: https://github.com/3dperceptionlab/biotacsp-stability-set-v2
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _REPO_ROOT / "data_biotac"

TRAIN_FILES = [
    "bts3v2_palm_down.csv",
    "bts3v2_palm_side.csv",
    "bts3v2_palm_45.csv",
]
TEST_FILES = [
    "bts3v2_palm_down_test.csv",
    "bts3v2_palm_side_test.csv",
    "bts3v2_palm_45_test.csv",
]

# Column groups
FINGER_PREFIXES = ["ff", "mf", "th"]        # index, middle, thumb
TAXELS_PER_FINGER = 24
SENSOR_COLS = [
    f"{fp}_biotac_{i}"
    for fp in FINGER_PREFIXES
    for i in range(1, TAXELS_PER_FINGER + 1)
]
N_FEATURES = len(SENSOR_COLS)                # 72


def _add_orientation(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Tag each row with the hand orientation inferred from the filename."""
    if "palm_down" in filename:
        df["orientation"] = "palm_down"
    elif "palm_side" in filename:
        df["orientation"] = "palm_side"
    elif "palm_45" in filename:
        df["orientation"] = "palm_45"
    return df


def load_raw(data_dir: Path | str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test DataFrames from the CSV files.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Each has columns: object, slipped, orientation, ff_biotac_1 … th_biotac_24
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    train_dfs = []
    for fn in TRAIN_FILES:
        df = pd.read_csv(data_dir / fn)
        _add_orientation(df, fn)
        train_dfs.append(df)

    test_dfs = []
    for fn in TEST_FILES:
        df = pd.read_csv(data_dir / fn)
        _add_orientation(df, fn)
        test_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    return train_df, test_df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the 72-feature tactile matrix from a DataFrame."""
    return df[SENSOR_COLS].values.astype(np.float32)


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract binary stability labels (0=stable, 1=slip)."""
    return df["slipped"].values.astype(np.int64)


if __name__ == "__main__":
    train_df, test_df = load_raw()
    print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
    print(f"Features: {N_FEATURES}")
    print(f"Train objects: {train_df['object'].nunique()}")
    print(f"Test objects:  {test_df['object'].nunique()}")
    print(f"Train stability: {train_df['slipped'].value_counts().to_dict()}")
    print(f"Test stability:  {test_df['slipped'].value_counts().to_dict()}")
