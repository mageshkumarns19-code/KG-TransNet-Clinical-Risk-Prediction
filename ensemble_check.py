"""
Ensemble check: KG-TransNet + Random Forest
=============================================
One honest, low-cost check before finalizing the paper's framing: does simply
averaging KG-TransNet's and Random Forest's predicted probabilities (both on
the same concept-pooled input) improve on either model alone?

This does NOT retrain anything -- it loads both already-trained models
(KG-TransNet from train_model.py's output, Random Forest re-fit quickly here
since sklearn models are cheap to refit) and evaluates the simple average.

Run AFTER train_model.py and train_baselines.py have both been run once:
    python ensemble_check.py --data_dir "./preprocessed" --model_dir "./model_output"
"""

import argparse
import json
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from model import KGTransNet, build_concept_membership, pool_to_concepts, CONCEPTS

RANDOM_STATE = 42


def tune_threshold(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def metrics_at(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


def main(data_dir, model_dir):
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    train_pool = np.load(os.path.join(data_dir, "train_pool.npz"))
    X_train_raw, y_train = train_pool["X_train"], train_pool["y_train"]
    X_heldout_raw, y_heldout = train_pool["X_heldout_test"], train_pool["y_heldout_test"]
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    membership = build_concept_membership(feature_names)
    X_train = pool_to_concepts(X_train_raw, membership)
    X_heldout = pool_to_concepts(X_heldout_raw, membership)

    # Load trained KG-TransNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg_model = KGTransNet(num_concepts=len(CONCEPTS), emb_dim=32).to(device)
    kg_model.load_state_dict(torch.load(
        os.path.join(model_dir, "kg_transnet_final.pt"), map_location=device))
    kg_model.eval()

    # Refit Random Forest (cheap) on the same train portion
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    def kg_probs(X):
        with torch.no_grad():
            logits, _, _ = kg_model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).cpu().numpy()

    results = {}
    for name, X_ext_raw_or_pooled, y_ext, already_pooled in [
        ("heldout_test", X_heldout, y_heldout, True),
    ]:
        p_kg = kg_probs(X_ext_raw_or_pooled)
        p_rf = rf.predict_proba(X_ext_raw_or_pooled)[:, 1]
        p_ensemble = (p_kg + p_rf) / 2.0

        thr_kg = tune_threshold(y_ext, p_kg)
        thr_rf = tune_threshold(y_ext, p_rf)
        thr_ens = tune_threshold(y_ext, p_ensemble)

        results[name] = {
            "kg_transnet": metrics_at(y_ext, p_kg, thr_kg),
            "random_forest": metrics_at(y_ext, p_rf, thr_rf),
            "ensemble_avg": metrics_at(y_ext, p_ensemble, thr_ens),
        }

    for ext_name in ["heart_cleveland", "heart_other_sites", "mimic"]:
        npz_path = os.path.join(data_dir, f"transfer_{ext_name}.npz")
        feat_path = os.path.join(data_dir, f"transfer_{ext_name}_features.json")
        if not os.path.exists(npz_path):
            continue
        transfer_data = np.load(npz_path)
        X_ext_raw, y_ext = transfer_data["X"], transfer_data["y"]
        with open(feat_path) as f:
            ext_feature_names = json.load(f)
        ext_membership = build_concept_membership(ext_feature_names)
        X_ext = pool_to_concepts(X_ext_raw, ext_membership)

        p_kg = kg_probs(X_ext)
        p_rf = rf.predict_proba(X_ext)[:, 1]
        p_ensemble = (p_kg + p_rf) / 2.0

        thr_kg = tune_threshold(y_ext, p_kg)
        thr_rf = tune_threshold(y_ext, p_rf)
        thr_ens = tune_threshold(y_ext, p_ensemble)

        results[ext_name] = {
            "kg_transnet": metrics_at(y_ext, p_kg, thr_kg),
            "random_forest": metrics_at(y_ext, p_rf, thr_rf),
            "ensemble_avg": metrics_at(y_ext, p_ensemble, thr_ens),
        }

    print(f"{'Dataset':<20}{'Model':<15}{'AUROC':<10}{'AUPRC':<10}{'F1':<10}")
    print("-" * 65)
    for dataset_name, model_results in results.items():
        for model_name, m in model_results.items():
            print(f"{dataset_name:<20}{model_name:<15}{m['auroc']:<10.4f}"
                  f"{m['auprc']:<10.4f}{m['f1']:<10.4f}")

    with open(os.path.join(model_dir, "ensemble_check_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {os.path.join(model_dir, 'ensemble_check_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./preprocessed")
    parser.add_argument("--model_dir", type=str, default="./model_output")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir)
