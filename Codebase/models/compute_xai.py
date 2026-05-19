"""XAI computation script — TreeSHAP (RF) + Integrated Gradients (CNN, BiLSTM, Ensemble).

Run once after training to generate explainability artefacts used by the dashboard:

    cd Tactile_Dashboard
    python Codebase/models/compute_xai.py

Outputs written to Codebase/models/saved/:
    shap_rf_values.npy           (n_samples, 102)     SHAP value per feature per sample
    shap_rf_base.npy             scalar                model expected value (base rate)
    shap_rf_feature_names.json                         102 feature name strings
    ig_cnn_values.npy            (n_samples, 3, 4, 6) IG attributions per taxel (CNN)
    ig_lstm_values.npy           (n_samples, 3, 24)   IG attributions per taxel (BiLSTM)
    ig_ensemble_values.npy       (n_samples, 3, 24)   IG attributions per finger (Ensemble)
    xai_metadata.json            sample-level info (index, true label, predicted probs, object, orientation)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap
from preprocessing.loader import get_feature_matrix, get_labels, load_raw
from preprocessing.pipeline import prepare_all
from models.cnn_model import TactileCNN2D
from models.lstm_model import TactileBiLSTM
from models.specialist_ensemble import TactileSpecialistEnsemble


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAVED_DIR = _REPO_ROOT / "Codebase" / "models" / "saved"

N_EXPLAIN = 10   # number of high-risk samples to explain
IG_STEPS   = 50  # integration steps for Integrated Gradients


# ── Integrated Gradients (pure PyTorch, no captum) ────────────────────────────

def _integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int = IG_STEPS,
) -> np.ndarray:
    """Compute IG attributions for a single sample.

    Args:
        model:        Eval-mode PyTorch model returning scalar logit.
        input_tensor: (1, *shape) input sample.
        baseline:     (1, *shape) reference state (zeros = no-contact).
        n_steps:      Number of Riemann integration steps.

    Returns:
        ndarray of same shape as input_tensor[0], values = IG attributions.
    """
    model.eval()
    alphas = torch.linspace(0.0, 1.0, n_steps, device=input_tensor.device)
    grad_sum = torch.zeros_like(input_tensor[0], dtype=torch.float32)

    for alpha in alphas:
        interp = (baseline + alpha * (input_tensor - baseline)).detach().requires_grad_(True)
        logit = model(interp)
        logit.backward()
        grad_sum += interp.grad[0].detach()

    ig = (input_tensor[0] - baseline[0]) * (grad_sum / n_steps)
    return ig.cpu().numpy()


def _compute_ig_batch(
    model: torch.nn.Module,
    X_batch: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run IG over a batch of samples. Returns array of same shape as X_batch."""
    baseline = torch.zeros(1, *X_batch.shape[1:], device=device)
    out = []
    for i, x in enumerate(X_batch):
        inp = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        attr = _integrated_gradients(model, inp, baseline)
        out.append(attr)
        if (i + 1) % 5 == 0:
            print(f"  IG: {i+1}/{len(X_batch)} samples done")
    return np.stack(out, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading raw data...")
    train_df, test_df = load_raw()

    X_train_raw = get_feature_matrix(train_df)
    y_train     = get_labels(train_df)
    X_test_raw  = get_feature_matrix(test_df)
    y_test      = get_labels(test_df)

    print("Running preprocessing pipeline...")
    data = prepare_all(X_train_raw, X_test_raw, y_train, y_test)

    # ── Load saved RF predictions to find highest-risk samples ───────────────
    with open(SAVED_DIR / "predictions.json") as f:
        preds = json.load(f)

    rf_probs = np.array(preds["Random Forest"]["y_prob"])
    # Top-N highest predicted slip probability from test set
    top_idx = np.argsort(rf_probs)[::-1][:N_EXPLAIN]
    print(f"Explaining top-{N_EXPLAIN} highest-risk samples: indices {top_idx.tolist()}")

    # ── 1. TreeSHAP for Random Forest ────────────────────────────────────────
    print("\nComputing TreeSHAP (Random Forest)...")
    rf_model = joblib.load(SAVED_DIR / "random_forest.pkl")
    explainer = shap.TreeExplainer(rf_model)

    X_explain_tab = data.X_test_tab[top_idx]
    explanation   = explainer(X_explain_tab)

    shap_values = explanation.values           # (N_EXPLAIN, 102, n_classes) for RF classifier
    base_value  = float(explainer.expected_value[1])  # base rate for class 1 (slip)

    # For binary RF, shap_values shape can be (N, 102, 2) — take class-1 slice
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    np.save(SAVED_DIR / "shap_rf_values.npy", shap_values.astype(np.float32))
    np.save(SAVED_DIR / "shap_rf_base.npy",   np.array([base_value], dtype=np.float32))

    with open(SAVED_DIR / "shap_rf_feature_names.json", "w") as f:
        json.dump(data.feature_names_tab, f)

    print(f"  SHAP base value (expected slip rate): {base_value:.4f}")
    print(f"  SHAP values shape: {shap_values.shape}")

    # ── 2. Integrated Gradients — 2D CNN ─────────────────────────────────────
    print("\nComputing Integrated Gradients (2D CNN)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = TactileCNN2D().to(device)
    cnn.load_state_dict(torch.load(SAVED_DIR / "2d_cnn.pth", map_location=device))
    cnn.eval()

    X_explain_img = data.X_test_img[top_idx]   # (N_EXPLAIN, 3, 4, 6)
    ig_cnn = _compute_ig_batch(cnn, X_explain_img, device)
    np.save(SAVED_DIR / "ig_cnn_values.npy", ig_cnn.astype(np.float32))
    print(f"  CNN IG shape: {ig_cnn.shape}")

    # ── 3. Integrated Gradients — BiLSTM ─────────────────────────────────────
    print("\nComputing Integrated Gradients (BiLSTM)...")
    lstm = TactileBiLSTM().to(device)
    lstm.load_state_dict(torch.load(SAVED_DIR / "bilstm.pth", map_location=device))
    lstm.eval()

    X_explain_seq = data.X_test_seq[top_idx]   # (N_EXPLAIN, 3, 24)
    ig_lstm = _compute_ig_batch(lstm, X_explain_seq, device)
    np.save(SAVED_DIR / "ig_lstm_values.npy", ig_lstm.astype(np.float32))
    print(f"  BiLSTM IG shape: {ig_lstm.shape}")

    # ── 4. Integrated Gradients — Specialist Ensemble ────────────────────────
    print("\nComputing Integrated Gradients (Specialist Ensemble)...")
    ensemble = TactileSpecialistEnsemble().to(device)
    ensemble.load_state_dict(
        torch.load(SAVED_DIR / "specialist_ensemble.pth", map_location=device)
    )
    ensemble.eval()

    # Use per-finger normalised sequences — same preprocessing the ensemble trained on
    X_explain_seq_pf = data.X_test_seq_pf[top_idx]   # (N_EXPLAIN, 3, 24)
    ig_ensemble = _compute_ig_batch(ensemble, X_explain_seq_pf, device)
    np.save(SAVED_DIR / "ig_ensemble_values.npy", ig_ensemble.astype(np.float32))
    print(f"  Ensemble IG shape: {ig_ensemble.shape}")

    # ── 5. Sample metadata ────────────────────────────────────────────────────
    with open(SAVED_DIR / "test_truth.json") as f:
        truth = json.load(f)

    cnn_probs      = np.array(preds["2D CNN"]["y_prob"])
    lstm_probs     = np.array(preds["BiLSTM"]["y_prob"])
    ensemble_probs = np.array(preds.get("Specialist Ensemble", {}).get("y_prob", [0.0] * len(rf_probs)))

    metadata = {
        "n_samples": N_EXPLAIN,
        "sample_indices": top_idx.tolist(),
        "shap_rf_base_value": base_value,
        "samples": [
            {
                "rank":           i + 1,
                "test_index":     int(top_idx[i]),
                "true_label":     int(truth["y_test_true"][top_idx[i]]),
                "object":         truth["test_objects"][top_idx[i]],
                "orientation":    truth["test_orientations"][top_idx[i]],
                "rf_prob":        float(rf_probs[top_idx[i]]),
                "cnn_prob":       float(cnn_probs[top_idx[i]]),
                "lstm_prob":      float(lstm_probs[top_idx[i]]),
                "ensemble_prob":  float(ensemble_probs[top_idx[i]]),
            }
            for i in range(N_EXPLAIN)
        ],
    }
    with open(SAVED_DIR / "xai_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nXAI artefacts saved to models/saved/")
    print(f"  shap_rf_values.npy       {shap_values.shape}")
    print(f"  ig_cnn_values.npy        {ig_cnn.shape}")
    print(f"  ig_lstm_values.npy       {ig_lstm.shape}")
    print(f"  ig_ensemble_values.npy   {ig_ensemble.shape}")
    print(f"  xai_metadata.json        {N_EXPLAIN} samples")


if __name__ == "__main__":
    main()
