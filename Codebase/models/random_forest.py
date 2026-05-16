"""Random Forest classifier for tactile grasp stability prediction.

This is the ML baseline model. It operates on handcrafted tabular
features (72 raw taxel values + 30 engineered features = 102 total).

Design rationale:
- Interpretable: feature importance rankings directly show which
  sensor readings and derived features matter most for stability.
- Fast inference: suitable for edge deployment on sensor MCUs.
- Robust baseline: no hyperparameter sensitivity to learning rates.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list | None = None,
    tune: bool = True,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train a Random Forest and return the model + evaluation metrics.

    Parameters
    ----------
    X_train, X_test : (N, 102) tabular feature arrays
    y_train, y_test : (N,) binary labels
    feature_names   : optional list of 102 feature names
    tune            : whether to run GridSearchCV (slower but better)

    Returns
    -------
    model : fitted RandomForestClassifier
    metrics : dict with accuracy, f1, precision, recall, auc, confusion_matrix,
              classification_report, feature_importances, best_params
    """
    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(
            base, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Feature importances
    importances = model.feature_importances_
    if feature_names is not None:
        importance_ranking = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        importance_ranking = sorted(
            enumerate(importances),
            key=lambda x: x[1],
            reverse=True,
        )

    metrics = {
        "model_name": "Random Forest",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importances": importances.tolist(),
        "feature_importance_ranking": importance_ranking[:20],
        "best_params": best_params,
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "n_params": sum(tree.tree_.node_count for tree in model.estimators_),
    }

    return model, metrics
