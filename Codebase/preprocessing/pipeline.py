"""Feature engineering and data preparation pipeline.

Transforms the raw 72-taxel BioTac readings into three different
representations, one per model architecture:

1. **Tabular features** (Random Forest) — 72 raw + 30 handcrafted = 102
2. **Tactile images** (2D CNN) — reshape 24 taxels/finger into a 4×6 grid,
   stack 3 fingers as 3 channels → (3, 4, 6)
3. **Finger sequences** (BiLSTM) — treat 3 fingers as a sequence of length 3,
   each with 24 features → (3, 24)

All outputs are normalised (z-score on training statistics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .loader import FINGER_PREFIXES, TAXELS_PER_FINGER, N_FEATURES


# ── BioTac SP taxel layout ────────────────────────────────────────────
# The 24 electrodes on each BioTac SP are arranged roughly as a 4×6 grid
# on the fingertip surface. The mapping below places taxel indices (0-23)
# into a 4-row × 6-column spatial grid for the CNN.
TAXEL_GRID_ROWS = 4
TAXEL_GRID_COLS = 6
TAXEL_TO_GRID = np.arange(24).reshape(TAXEL_GRID_ROWS, TAXEL_GRID_COLS)


@dataclass
class PreparedData:
    """Container returned by ``prepare_all``."""

    # Tabular (Random Forest)
    X_train_tab: np.ndarray    # (N_train, 102)
    X_test_tab: np.ndarray     # (N_test,  102)

    # 2D tactile images (CNN)
    X_train_img: np.ndarray    # (N_train, 3, 4, 6)  channels-first
    X_test_img: np.ndarray     # (N_test,  3, 4, 6)

    # Finger sequences (LSTM)
    X_train_seq: np.ndarray    # (N_train, 3, 24)
    X_test_seq: np.ndarray     # (N_test,  3, 24)

    # Labels
    y_train: np.ndarray        # (N_train,)
    y_test: np.ndarray         # (N_test,)

    # Metadata
    feature_names_tab: list    # 102 column names
    scaler: StandardScaler     # fitted on training data


# ── Handcrafted features for the Random Forest ────────────────────────

def _engineer_features(X_raw: np.ndarray) -> Tuple[np.ndarray, list]:
    """Add 30 handcrafted features to the 72 raw ones.

    For each of the 3 fingers (24 taxels each):
        - mean, std, max, min, range           (5)
        - contact area (# taxels > threshold)  (1)
        - centre of pressure x, y              (2)
        - total force (sum)                     (1)
        - skewness                              (1)
    = 10 features × 3 fingers = 30 handcrafted

    Returns (N, 102) array and list of 102 feature names.
    """
    n = X_raw.shape[0]
    extra = []
    names = []

    for fidx, fp in enumerate(FINGER_PREFIXES):
        finger = X_raw[:, fidx * 24 : (fidx + 1) * 24]

        mean_v = finger.mean(axis=1)
        std_v = finger.std(axis=1)
        max_v = finger.max(axis=1)
        min_v = finger.min(axis=1)
        range_v = max_v - min_v
        contact = (finger > np.median(finger)).sum(axis=1).astype(np.float32)
        total = finger.sum(axis=1)

        # Centre of pressure on the 4×6 grid
        grid_x = np.tile(np.arange(TAXEL_GRID_COLS), TAXEL_GRID_ROWS)
        grid_y = np.repeat(np.arange(TAXEL_GRID_ROWS), TAXEL_GRID_COLS)
        cop_denom = total + 1e-9
        cop_x = (finger * grid_x).sum(axis=1) / cop_denom
        cop_y = (finger * grid_y).sum(axis=1) / cop_denom

        # Skewness
        centered = finger - mean_v[:, None]
        skew = (centered ** 3).mean(axis=1) / (std_v ** 3 + 1e-9)

        block = np.column_stack([
            mean_v, std_v, max_v, min_v, range_v,
            contact, cop_x, cop_y, total, skew,
        ])
        extra.append(block)
        for feat in ["mean", "std", "max", "min", "range",
                      "contact_area", "cop_x", "cop_y", "total_force", "skewness"]:
            names.append(f"{fp}_{feat}")

    raw_names = [f"{fp}_biotac_{i}" for fp in FINGER_PREFIXES
                 for i in range(1, 25)]
    X_tab = np.hstack([X_raw, np.hstack(extra)])
    return X_tab.astype(np.float32), raw_names + names


# ── Reshape for CNN ───────────────────────────────────────────────────

def _to_tactile_images(X_raw: np.ndarray) -> np.ndarray:
    """Reshape (N, 72) → (N, 3, 4, 6) tactile images.

    Each finger's 24 taxels are laid out on a 4×6 spatial grid.
    The 3 fingers become 3 channels (like RGB).
    """
    n = X_raw.shape[0]
    imgs = np.zeros((n, 3, TAXEL_GRID_ROWS, TAXEL_GRID_COLS), dtype=np.float32)
    for fidx in range(3):
        finger = X_raw[:, fidx * 24 : (fidx + 1) * 24]
        imgs[:, fidx] = finger.reshape(n, TAXEL_GRID_ROWS, TAXEL_GRID_COLS)
    return imgs


# ── Reshape for LSTM ──────────────────────────────────────────────────

def _to_finger_sequences(X_raw: np.ndarray) -> np.ndarray:
    """Reshape (N, 72) → (N, 3, 24) finger sequences.

    Treats the 3 fingers as a sequence: index → middle → thumb.
    Each timestep has 24 taxel features.
    """
    n = X_raw.shape[0]
    return X_raw.reshape(n, 3, 24).astype(np.float32)


# ── Main pipeline ─────────────────────────────────────────────────────

def prepare_all(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> PreparedData:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    X_train_raw, X_test_raw : (N, 72) float arrays of raw taxel readings
    y_train, y_test : (N,) int arrays of binary labels (0/1)

    Returns
    -------
    PreparedData with all three representations normalised.
    """
    # 1) Tabular
    X_tab_train, feat_names = _engineer_features(X_train_raw)
    X_tab_test, _ = _engineer_features(X_test_raw)

    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab_train)
    X_tab_test = scaler.transform(X_tab_test)

    # 2) CNN images — normalise on flattened raw, then reshape
    raw_scaler = StandardScaler()
    X_raw_train_n = raw_scaler.fit_transform(X_train_raw)
    X_raw_test_n = raw_scaler.transform(X_test_raw)

    X_img_train = _to_tactile_images(X_raw_train_n)
    X_img_test = _to_tactile_images(X_raw_test_n)

    # 3) LSTM sequences
    X_seq_train = _to_finger_sequences(X_raw_train_n)
    X_seq_test = _to_finger_sequences(X_raw_test_n)

    return PreparedData(
        X_train_tab=X_tab_train.astype(np.float32),
        X_test_tab=X_tab_test.astype(np.float32),
        X_train_img=X_img_train,
        X_test_img=X_img_test,
        X_train_seq=X_seq_train,
        X_test_seq=X_seq_test,
        y_train=y_train,
        y_test=y_test,
        feature_names_tab=feat_names,
        scaler=scaler,
    )
