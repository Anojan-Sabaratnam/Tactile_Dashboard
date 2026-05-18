"""Exploratory Data Analysis for the BioTac SP Grasp Stability dataset.

Run from anywhere:
    python Codebase/preprocessing/eda.py
    python -m preprocessing.eda          (from Codebase/)

Displays four interactive plot windows:
    1. Class Balance (Stable vs Slip)
    2. Spatial Taxel Heatmaps (4×6 grid per finger)
    3. Per-Orientation Slip Rates
    4. Feature Distributions (violin plots)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .loader import FINGER_PREFIXES, get_feature_matrix, get_labels, load_raw
    from .pipeline import _engineer_features
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocessing.loader import FINGER_PREFIXES, get_feature_matrix, get_labels, load_raw
    from preprocessing.pipeline import _engineer_features

plt.style.use("dark_background")

_BG       = "#0f172a"
_PANEL    = "#1e293b"
_BORDER   = "#334155"
_TEXT     = "white"
_MUTED    = "#94a3b8"
_CYAN     = "#00e5ff"
_GREEN    = "#00e676"
_RED      = "#ff1744"
_PURPLE   = "#b000ff"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.yaxis.label.set_color(_MUTED)
    ax.xaxis.label.set_color(_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)


def _show(fig: plt.Figure, title: str) -> None:
    print(f"  Showing: {title}")
    plt.figure(fig.number)
    plt.show()


# ── Plot 1: Class Balance ─────────────────────────────────────────────

def plot_class_balance(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=_BG)

    for ax, df, title in zip(axes, [train_df, test_df], ["Training Set", "Test Set"]):
        counts = df["slipped"].value_counts().sort_index()
        bars = ax.bar(["Stable (0)", "Slip (1)"], counts.values,
                      color=[_GREEN, _RED], width=0.5)
        ax.set_title(title, color=_TEXT, fontsize=13, pad=10)
        ax.set_ylabel("Sample Count", color=_MUTED)
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(val), ha="center", color=_TEXT, fontsize=11, fontweight="bold",
            )
        ax.set_ylim(0, max(counts) * 1.15)
        _style_ax(ax)

    fig.suptitle("Class Balance: Stable vs Slip", color=_TEXT, fontsize=15, fontweight="bold")
    plt.tight_layout()
    _show(fig, "Class Balance")


# ── Plot 2: Spatial Heatmaps ──────────────────────────────────────────

def plot_spatial_heatmaps(train_df: pd.DataFrame) -> None:
    finger_labels = {"ff": "Index Finger", "mf": "Middle Finger", "th": "Thumb"}
    stable = train_df[train_df["slipped"] == 0]
    slip   = train_df[train_df["slipped"] == 1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=_BG)

    for col, fp in enumerate(FINGER_PREFIXES):
        taxel_cols   = [f"{fp}_biotac_{i}" for i in range(1, 25)]
        stable_grid  = stable[taxel_cols].mean().values.reshape(4, 6)
        slip_grid    = slip[taxel_cols].mean().values.reshape(4, 6)

        for row, (grid, cmap, label) in enumerate([
            (stable_grid, "YlGn",   "Stable"),
            (slip_grid,   "YlOrRd", "Slip"),
        ]):
            ax = axes[row][col]
            sns.heatmap(
                grid, cmap=cmap, ax=ax, annot=True, fmt=".0f",
                linewidths=0.5, linecolor=_BG, cbar=False,
                annot_kws={"size": 8, "color": "black"},
            )
            ax.set_title(f"{label} — {finger_labels[fp]}", color=_TEXT, fontsize=10)
            ax.set_xlabel("Taxel Column", color=_MUTED, fontsize=8)
            ax.set_ylabel("Taxel Row",    color=_MUTED, fontsize=8)
            ax.tick_params(colors=_TEXT, labelsize=7)

    fig.suptitle(
        "Mean Taxel Pressure Distribution: Stable vs Slip (4×6 Grid per Finger)",
        color=_TEXT, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _show(fig, "Spatial Heatmaps")


# ── Plot 3: Per-Orientation Slip Rate ─────────────────────────────────

def plot_orientation_slip_rates(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    colors = [_CYAN, _PURPLE, _GREEN]

    for ax, df, title in zip(axes, [train_df, test_df], ["Training Set", "Test Set"]):
        rates = df.groupby("orientation")["slipped"].mean() * 100
        bars  = ax.bar(rates.index, rates.values, color=colors, width=0.5)
        ax.set_title(f"Slip Rate by Orientation — {title}", color=_TEXT, fontsize=12, pad=10)
        ax.set_ylabel("Slip Rate (%)", color=_MUTED)
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.1f}%", ha="center", color=_TEXT, fontsize=10, fontweight="bold",
            )
        _style_ax(ax)

    fig.suptitle("Per-Orientation Slip Rate Analysis", color=_TEXT, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _show(fig, "Orientation Slip Rates")


# ── Plot 4: Feature Distributions (violin) ───────────────────────────

def plot_feature_distributions(train_df: pd.DataFrame) -> None:
    X_raw      = get_feature_matrix(train_df)
    y          = get_labels(train_df)
    X_tab, feat_names = _engineer_features(X_raw)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}

    features_to_plot = [
        "ff_cop_x", "ff_cop_y", "ff_range", "ff_skewness",
        "mf_cop_x", "mf_cop_y", "th_range", "th_skewness",
    ]
    features_to_plot = [f for f in features_to_plot if f in feat_idx]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=_BG)
    axes = axes.flatten()

    for ax, feat in zip(axes, features_to_plot):
        idx         = feat_idx[feat]
        vals_stable = X_tab[y == 0, idx]
        vals_slip   = X_tab[y == 1, idx]

        parts = ax.violinplot(
            [vals_stable, vals_slip], positions=[0, 1],
            showmedians=True, showextrema=False,
        )
        parts["cmedians"].set_color(_TEXT)
        parts["cmedians"].set_linewidth(2)
        for pc, color in zip(parts["bodies"], [_GREEN, _RED]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stable", "Slip"], color=_TEXT)
        ax.set_title(feat, color=_CYAN, fontsize=10, pad=6)
        _style_ax(ax)

    fig.suptitle(
        "Feature Distributions: Stable vs Slip (Key Engineered Features)",
        color=_TEXT, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _show(fig, "Feature Distributions")


# ── Entry point ───────────────────────────────────────────────────────

def run_eda() -> None:
    print("Loading data...")
    train_df, test_df = load_raw()
    print(f"  Train : {train_df.shape}  |  Test : {test_df.shape}")
    print(f"  Train class balance : {train_df['slipped'].value_counts().to_dict()}")
    print(f"  Test  class balance : {test_df['slipped'].value_counts().to_dict()}")

    print("\nGenerating EDA plots...")
    plot_class_balance(train_df, test_df)
    plot_spatial_heatmaps(train_df)
    plot_orientation_slip_rates(train_df, test_df)
    plot_feature_distributions(train_df)
    print("\nAll plots displayed.")


if __name__ == "__main__":
    run_eda()
