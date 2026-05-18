# AIFI Tactile Dashboard

A production-ready Industry 5.0 web dashboard for monitoring AI-enhanced BioTac SP tactile sensors on humanoid robotic grippers. Combines three machine learning models with a full Explainable AI (XAI) pipeline so both engineers and shift supervisors can understand every slip prediction the system makes.

---

## What it does

The dashboard monitors grasp stability in real time using a three-finger BioTac SP sensor array (72 taxels total). Three models are trained and compared:

| Model | Input | Explainability |
|---|---|---|
| Random Forest | 102 tabular features (72 raw + 30 engineered) | Full — SHAP (TreeSHAP) |
| 2D CNN | (3, 4, 6) tactile images | Partial — Integrated Gradients taxel maps |
| BiLSTM | (3, 24) finger sequences | Partial — Integrated Gradients per-finger |

The **AI Diagnostics** page exposes two XAI pillars:
- **SHAP waterfall** — shows exactly which sensor features drove the highest-risk prediction and by how much
- **Integrated Gradients heatmaps** — traces the CNN's decision back to individual taxel locations on each finger

---

## Prerequisites

- Python 3.10+
- macOS or Linux (for `start.sh`); Windows users see the manual steps below

---

## Quick start (macOS / Linux)

```bash
cd Tactile_Dashboard
chmod +x start.sh
./start.sh
```

`start.sh` will:
1. Create a `.venv` virtual environment if one doesn't exist
2. Install all dependencies from `Codebase/requirements.txt`
3. Run `compute_xai.py` on first launch to generate SHAP and Integrated Gradients artefacts
4. Start the server and open `http://127.0.0.1:8051` once it is ready

---

## Manual setup (all platforms)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r Codebase/requirements.txt

# 3. Train the models (required once — generates metrics.json, .pth, .pkl files)
python Codebase/models/train_all.py

# 4. Generate XAI artefacts (required once — generates SHAP + IG .npy files)
python Codebase/models/compute_xai.py

# 5. Start the dashboard
python run.py
```

Open `http://127.0.0.1:8051` in your browser.

---

## Repository layout

```
Tactile_Dashboard/
├── run.py                              Entry point — starts the Dash server on port 8051
├── start.sh                            One-command launcher (macOS / Linux)
└── Codebase/
    ├── requirements.txt
    ├── data_biotac/                    Raw BioTac SP dataset (train / test CSV)
    ├── preprocessing/
    │   ├── loader.py                   Data loading and label extraction
    │   └── pipeline.py                 Feature engineering + normalisation pipeline
    ├── models/
    │   ├── random_forest.py            Random Forest with GridSearchCV tuning
    │   ├── cnn_model.py                TactileCNN2D — (3, 4, 6) tactile image input
    │   ├── lstm_model.py               TactileBiLSTM — (3, 24) finger sequence input
    │   ├── train_all.py                Training orchestrator — runs all three models
    │   ├── compute_xai.py              XAI pipeline — TreeSHAP + Integrated Gradients
    │   └── saved/                      Generated artefacts (models, metrics, XAI .npy files)
    └── dashboard/
        ├── app.py                      Dash app initialisation
        └── pages/
            ├── overview.py             Fleet overview and live telemetry simulation
            ├── model_comparison.py     AI Diagnostics — XAI, confusion matrices, drift monitor
            └── ...
```

---

## Dashboard pages

| Page | What it shows |
|---|---|
| Fleet Overview | Real-time simulation of 74-channel tactile data and slip predictions |
| Robot Fleet Status | Live health metrics per robotic arm |
| AI Diagnostics | SHAP waterfall, Integrated Gradients taxel maps, model comparison, deployment recommendation, 30-day concept drift monitor |
| ROI & System Config | Interactive savings calculator and risk mitigation matrices |

---

## XAI pipeline details

`compute_xai.py` selects the 10 highest-risk predictions from the test set and computes:

- **TreeSHAP** (Random Forest) — exact Shapley values for all 102 features; no approximation
- **Integrated Gradients** (CNN + BiLSTM) — 50-step Riemann integration from a zero-contact baseline to the active sensor state; implemented in pure PyTorch without external dependencies

Outputs saved to `Codebase/models/saved/`:

| File | Contents |
|---|---|
| `shap_rf_values.npy` | (10, 102) SHAP values |
| `shap_rf_base.npy` | Model base rate (expected slip probability) |
| `ig_cnn_values.npy` | (10, 3, 4, 6) CNN taxel attributions |
| `ig_lstm_values.npy` | (10, 3, 24) BiLSTM finger attributions |
| `xai_metadata.json` | Per-sample object, orientation, and predicted probabilities |

---

## Retraining

If you update the dataset or want fresh model weights:

```bash
python Codebase/models/train_all.py   # retrain all three models
python Codebase/models/compute_xai.py # recompute XAI artefacts
```

Then restart the dashboard. No other steps are needed.
