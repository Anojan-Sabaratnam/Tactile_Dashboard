"""Training orchestrator for all three tactile models.

Loads data, runs the preprocessing pipeline, trains the Random Forest,
CNN, and BiLSTM, and saves the weights and metrics to disk for the dashboard.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.loader import get_feature_matrix, get_labels, load_raw
from preprocessing.pipeline import prepare_all
from models.random_forest import train_random_forest
from models.cnn_model import TactileCNN2D
from models.lstm_model import TactileBiLSTM


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAVED_DIR = _REPO_ROOT / "Codebase" / "models" / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    """Generic training loop for PyTorch binary classifiers."""
    print(f"\n--- Training {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Data loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "model_name": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "train_time_sec": train_time,
        "n_params": model.get_num_params(),
        "y_pred": preds.tolist(),
        "y_prob": probs.tolist(),
    }

    print(f"{name} Test Accuracy: {metrics['accuracy']:.4f}")

    # Save weights
    torch.save(model.state_dict(), SAVED_DIR / f"{name.lower().replace(' ', '_')}.pth")
    return metrics


def main():
    print("Loading raw data...")
    train_df, test_df = load_raw()

    X_train_raw = get_feature_matrix(train_df)
    y_train = get_labels(train_df)
    X_test_raw = get_feature_matrix(test_df)
    y_test = get_labels(test_df)

    print("Running preprocessing pipeline...")
    data = prepare_all(X_train_raw, X_test_raw, y_train, y_test)

    # Save scaler for inference
    joblib.dump(data.scaler, SAVED_DIR / "scaler.pkl")

    all_metrics = {}

    # 1) Random Forest
    start = time.time()
    rf_model, rf_metrics = train_random_forest(
        data.X_train_tab, data.y_train,
        data.X_test_tab, data.y_test,
        feature_names=data.feature_names_tab,
        tune=True,
    )
    rf_metrics["train_time_sec"] = time.time() - start
    print(f"\nRandom Forest Test Accuracy: {rf_metrics['accuracy']:.4f}")
    joblib.dump(rf_model, SAVED_DIR / "random_forest.pkl")
    all_metrics["Random Forest"] = rf_metrics

    # 2) 2D CNN
    cnn = TactileCNN2D()
    cnn_metrics = train_pytorch_model(
        cnn,
        data.X_train_img, data.y_train,
        data.X_test_img, data.y_test,
        name="2D CNN",
        epochs=40,
        batch_size=64,
        lr=0.001,
    )
    all_metrics["2D CNN"] = cnn_metrics

    # 3) BiLSTM
    lstm = TactileBiLSTM()
    lstm_metrics = train_pytorch_model(
        lstm,
        data.X_train_seq, data.y_train,
        data.X_test_seq, data.y_test,
        name="BiLSTM",
        epochs=40,
        batch_size=64,
        lr=0.001,
    )
    all_metrics["BiLSTM"] = lstm_metrics

    # Save truth labels for the dashboard to compare against
    truth_data = {
        "y_test_true": data.y_test.tolist(),
        "test_objects": test_df["object"].tolist(),
        "test_orientations": test_df["orientation"].tolist(),
    }
    with open(SAVED_DIR / "test_truth.json", "w") as f:
        json.dump(truth_data, f)

    # Save metrics (excluding large prediction arrays to save space in the main metrics file)
    summary_metrics = {}
    predictions = {}
    for k, v in all_metrics.items():
        summary = v.copy()
        predictions[k] = {
            "y_pred": summary.pop("y_pred"),
            "y_prob": summary.pop("y_prob"),
        }
        summary_metrics[k] = summary

    with open(SAVED_DIR / "metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)
        
    with open(SAVED_DIR / "predictions.json", "w") as f:
        json.dump(predictions, f)

    print("\nTraining complete! Artifacts saved to models/saved/")
    for model_name, m in summary_metrics.items():
        print(f"{model_name:>15} -> Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f}")


if __name__ == "__main__":
    main()
