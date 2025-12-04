#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


TRAIN_PATH = "combined_training.csv"
DRUGBANK_RDKIT_PATH = "DrugBank_approved_rdkit.csv"
OUT_PATH = "DrugBank_approved_predictions_all_models.csv"


# =========================
# 1. Load training data
# =========================

def load_training_data(path: str):
    print(f"\nLoading training data from: {path}")
    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Training file must contain a 'Class' column (0/1 labels).")

    y = df["Class"].values
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()

    # Force numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    # Replace inf with NaN so imputer can handle them
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"  Training samples: {X.shape[0]}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Feature columns example: {feature_cols[:5]} ...")

    return X.values, y, feature_cols


# =========================
# 2. Define & train models
# =========================

def get_model_dict():
    """
    Return dict of model_name -> sklearn estimator (unfitted).
    SVCs have probability=True so we can get predict_proba.
    """
    models = {
        "SVC_linear": SVC(kernel="linear", probability=True, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "SVC_rbf": SVC(kernel="rbf", probability=True, random_state=0),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            random_state=0,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            criterion="entropy",
            random_state=0
        ),
    }
    return models


def train_models(X: np.ndarray, y: np.ndarray):
    """
    Fit imputer + scaler, then train all models.
    Returns: imputer, scaler, dict(model_name -> trained_model)
    """
    # Imputer + scaler
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    scaler = StandardScaler()

    # Fit on training data
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    models = get_model_dict()
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    print("\nTraining models with 5-fold ROC AUC cross-validation...")
    for name, clf in models.items():
        print(f"\nModel: {name}")
        # Cross-validation on scaled data
        auc_scores = cross_val_score(
            clf,
            X_scaled,
            y,
            cv=cv,
            scoring="roc_auc"
        )
        print(f"  Mean ROC AUC (5-fold CV): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

        # Fit on full training set
        clf.fit(X_scaled, y)
        trained_models[name] = clf
        print("  Fitted on full training set.")

    return imputer, scaler, trained_models


# =========================
# 3. Load & clean DrugBank RDKit descriptors
# =========================

def load_and_clean_drugbank(path: str, feature_cols_training):
    """
    Load DrugBank RDKit descriptors file.

    Assumes first column is DrugBank_ID (but will also search common names
    just in case). Aligns features to training feature set; missing features
    are created as NaN. Returns (drugbank_ids, X_new_values).
    """
    print(f"\nLoading DrugBank data from: {path}")
    df = pd.read_csv(path)

    # Detect ID column; prefer explicit 'DrugBank_ID'
    id_col = None
    if "DrugBank ID" in df.columns:
        id_col = "DrugBank ID"
    else:
        id_candidates = ["DrugBankID", "DrugBank Id", "ID"]
        for c in id_candidates:
            if c in df.columns:
                id_col = c
                break

    if id_col is None:
        # As a fallback, assume first column is the ID
        id_col = df.columns[0]
        print(f"  WARNING: Could not find standard DrugBank ID column, using first column: {id_col}")

    drugbank_ids = df[id_col].astype(str).tolist()
    print(f"  DrugBank compounds: {len(drugbank_ids)}")

    # Create missing feature columns as NaN to match training features
    missing_cols = [c for c in feature_cols_training if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns missing in DrugBank file; filling with NaN.")
        for c in missing_cols:
            df[c] = np.nan

    # Make sure the order of columns matches the training feature order
    X_new = df[feature_cols_training].copy()

    # Enforce numeric and replace inf with NaN
    X_new = X_new.apply(pd.to_numeric, errors="coerce")
    X_new.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_nan = X_new.isna().sum().sum()
    print(f"  Number of NaN values after cleaning: {n_nan}")
    print(f"  Number of features (aligned with training): {X_new.shape[1]}")

    return drugbank_ids, X_new.values


# =========================
# 4. Predict with all models
# =========================

def clean_nonfinite_after_scaling(X_scaled: np.ndarray):
    """
    Replace any remaining non-finite values with 0 and clip extremes
    to stay within a safe float32 range.
    """
    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    bad_mask = ~np.isfinite(X_scaled)
    if bad_mask.any():
        n_bad = bad_mask.sum()
        print(f"  WARNING: Found {n_bad} non-finite values AFTER scaling. Setting them to 0.")
        X_scaled[bad_mask] = 0.0

    max_safe = np.finfo(np.float32).max / 10.0
    X_scaled = np.clip(X_scaled, -max_safe, max_safe)

    return X_scaled


def predict_with_models(imputer, scaler, models_dict, X_new_raw, drugbank_ids, out_path: str):
    """
    Apply imputer + scaler + each trained model to the DrugBank descriptors
    and write a CSV with one column per model's class prediction and probability.
    """
    print("\nPreprocessing DrugBank descriptors and predicting activity...")

    # Impute and scale
    X_new_imputed = imputer.transform(X_new_raw)
    X_new_scaled = scaler.transform(X_new_imputed)
    X_new_scaled = clean_nonfinite_after_scaling(X_new_scaled)

    # Collect predictions
    pred_classes = {}
    pred_probas = {}

    for name, clf in models_dict.items():
        print(f"  Predicting with {name}...")
        y_pred = clf.predict(X_new_scaled)

        # Probabilities (probability that Class=1, i.e., active)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_new_scaled)[:, 1]
        elif hasattr(clf, "decision_function"):
            # Scale decision_function to [0,1] as a pseudo-probability
            scores = clf.decision_function(X_new_scaled)
            s_min = scores.min()
            s_max = scores.max()
            if s_max > s_min:
                proba = (scores - s_min) / (s_max - s_min)
            else:
                proba = np.zeros_like(scores, dtype=float)
        else:
            proba = np.full_like(y_pred, np.nan, dtype=float)

        pred_classes[name] = y_pred
        pred_probas[name] = proba

    # Build output table
    data = {"DrugBank ID": drugbank_ids}

    for name in models_dict.keys():
        data[f"{name}_Predicted_Class"] = pred_classes[name]
        data[f"{name}_Prob_Active"] = pred_probas[name]

    # Optional: majority vote across models
    preds_matrix = np.column_stack([pred_classes[name] for name in models_dict.keys()])
    majority_vote = (preds_matrix.sum(axis=1) >= (preds_matrix.shape[1] / 2)).astype(int)
    data["MajorityVote_Class"] = majority_vote

    out_df = pd.DataFrame(data)

    print(f"\nWriting predictions to: {out_path}")
    out_df.to_csv(out_path, index=False)
    print("Done.")


# =========================
# 5. Main
# =========================

def main():
    # 1. Load training data
    X, y, feature_cols = load_training_data(TRAIN_PATH)

    # 2. Train all models (with shared imputer + scaler)
    imputer, scaler, trained_models = train_models(X, y)

    # 3. Load & clean DrugBank RDKit descriptors
    drugbank_ids, X_new_raw = load_and_clean_drugbank(
        DRUGBANK_RDKIT_PATH,
        feature_cols_training=feature_cols
    )

    # 4. Predict with all models and write CSV
    predict_with_models(
        imputer=imputer,
        scaler=scaler,
        models_dict=trained_models,
        X_new_raw=X_new_raw,
        drugbank_ids=drugbank_ids,
        out_path=OUT_PATH,
    )


if __name__ == "__main__":
    main()
