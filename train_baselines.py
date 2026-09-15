"""
Baseline comparisons for KG-TransNet
======================================
Two families of baselines, for two different questions:

(A) CONCEPT-POOLED baselines (Logistic Regression, Random Forest, Plain MLP)
    -- these use the EXACT SAME 18-dim concept-pooled input as KG-TransNet.
    Purpose: isolate whether the KG graph-attention layer itself adds value,
    holding the input representation constant. This is the key ablation for
    the paper's core claim. These baselines CAN be evaluated on the transfer
    sets too, exactly like KG-TransNet.

(B) RAW-FEATURE MLP baseline -- uses the full un-pooled raw feature vector
    (44 dims for the train pool) instead of concept-pooled input.
    Purpose: an honest "ceiling" reference showing what's lost by compressing
    to concepts. IMPORTANT: this baseline is architecturally NOT transferable
    to the other datasets (they have different raw feature counts: 13, 7, 6),
    so it is only evaluated on the train-domain held-out test. That inability
    to transfer is itself part of the paper's argument for why concept-level
    KG-conditioning is necessary, not just one modeling choice among others.

Run:
    python train_baselines.py --data_dir "./preprocessed" --out_dir "./model_output"
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, average_precision_score)

from model import build_concept_membership, pool_to_concepts, CONCEPTS

RANDOM_STATE = 42
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 128
HIDDEN_DIM = 32


# ----------------------------------------------------------------------
# Plain MLP (no KG graph attention, no concept identity embeddings) --
# same input representation as KG-TransNet, roughly comparable capacity.
# ----------------------------------------------------------------------
class PlainMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def compute_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "threshold_used": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def train_mlp(X_train, y_train, X_val, y_val, epochs=EPOCHS, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlainMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    pos_rate = max(y_train.mean(), 1e-6)
    pos_weight = torch.tensor([min((1 - pos_rate) / pos_rate, 3.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    n = X_train_t.shape[0]
    best_val_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
        scheduler.step(val_auc)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model


def evaluate_mlp(model, X, y, threshold):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()
    return compute_metrics(y, probs, threshold), probs


def run_cv_sklearn(model_fn, X, y, cv_folds, name):
    print(f"\n--- {name}: 5-fold CV ---")
    fold_metrics = []
    for fold in cv_folds:
        tr_idx, val_idx = np.array(fold["train_idx"]), np.array(fold["val_idx"])
        clf = model_fn()
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        thr = tune_threshold(y[val_idx], probs)
        metrics = compute_metrics(y[val_idx], probs, thr)
        fold_metrics.append(metrics)
        print(f"  Fold {fold['fold']}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
              f"F1={metrics['f1']:.4f}")
    return fold_metrics


def run_cv_mlp(X, y, cv_folds, name):
    print(f"\n--- {name}: 5-fold CV ---")
    fold_metrics = []
    for fold in cv_folds:
        tr_idx, val_idx = np.array(fold["train_idx"]), np.array(fold["val_idx"])
        model = train_mlp(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
        with torch.no_grad():
            device = next(model.parameters()).device
            probs = torch.sigmoid(model(torch.tensor(X[val_idx], dtype=torch.float32,
                                                        device=device))).cpu().numpy()
        thr = tune_threshold(y[val_idx], probs)
        metrics = compute_metrics(y[val_idx], probs, thr)
        fold_metrics.append(metrics)
        print(f"  Fold {fold['fold']}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
              f"F1={metrics['f1']:.4f}")
    return fold_metrics


def summarize(fold_metrics):
    summary = {}
    for key in ["auroc", "auprc", "f1", "precision", "recall"]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    train_pool = np.load(os.path.join(data_dir, "train_pool.npz"))
    X_train_raw, y_train = train_pool["X_train"], train_pool["y_train"]
    X_heldout_raw, y_heldout = train_pool["X_heldout_test"], train_pool["y_heldout_test"]

    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(data_dir, "cv_folds.json")) as f:
        cv_folds = json.load(f)

    membership = build_concept_membership(feature_names)
    X_train_concept = pool_to_concepts(X_train_raw, membership)
    X_heldout_concept = pool_to_concepts(X_heldout_raw, membership)

    all_results = {}

    # ================================================================
    # (A) CONCEPT-POOLED baselines -- transfer-compatible, fair ablation
    # ================================================================
    print("=" * 60)
    print("(A) CONCEPT-POOLED BASELINES (same input as KG-TransNet)")
    print("=" * 60)

    baseline_defs = {
        "logistic_regression": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

    for name, model_fn in baseline_defs.items():
        fold_metrics = run_cv_sklearn(model_fn, X_train_concept, y_train, cv_folds, name)
        summary = summarize(fold_metrics)
        print(f"  {name} CV summary: AUROC={summary['auroc']['mean']:.4f}+/-{summary['auroc']['std']:.4f} "
              f"AUPRC={summary['auprc']['mean']:.4f}")

        # Final fit on full train portion, evaluate held-out + transfer
        clf = model_fn()
        clf.fit(X_train_concept, y_train)
        val_probs = clf.predict_proba(X_train_concept)[:, 1]
        thr = tune_threshold(y_train, val_probs)

        heldout_probs = clf.predict_proba(X_heldout_concept)[:, 1]
        heldout_metrics = compute_metrics(y_heldout, heldout_probs, thr)
        print(f"  {name} held-out test: AUROC={heldout_metrics['auroc']:.4f} "
              f"F1={heldout_metrics['f1']:.4f}")

        transfer_metrics = {}
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
            ext_probs = clf.predict_proba(X_ext)[:, 1]
            m = compute_metrics(y_ext, ext_probs, thr)
            transfer_metrics[ext_name] = m
            print(f"    transfer -> {ext_name}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

        all_results[name] = {
            "cv_fold_metrics": fold_metrics, "cv_summary": summary,
            "heldout_test_metrics": heldout_metrics, "transfer_results": transfer_metrics,
        }

    # Plain MLP (concept-pooled input, no graph attention)
    fold_metrics = run_cv_mlp(X_train_concept, y_train, cv_folds, "plain_mlp_concept")
    summary = summarize(fold_metrics)
    print(f"  plain_mlp_concept CV summary: AUROC={summary['auroc']['mean']:.4f}+/-"
          f"{summary['auroc']['std']:.4f} AUPRC={summary['auprc']['mean']:.4f}")

    n = len(X_train_concept)
    perm = np.random.RandomState(RANDOM_STATE).permutation(n)
    split = int(0.9 * n)
    mlp_model = train_mlp(X_train_concept[perm[:split]], y_train[perm[:split]],
                            X_train_concept[perm[split:]], y_train[perm[split:]])
    with torch.no_grad():
        device = next(mlp_model.parameters()).device
        val_probs = torch.sigmoid(mlp_model(torch.tensor(
            X_train_concept[perm[split:]], dtype=torch.float32, device=device))).cpu().numpy()
    thr = tune_threshold(y_train[perm[split:]], val_probs)

    heldout_metrics, _ = evaluate_mlp(mlp_model, X_heldout_concept, y_heldout, thr)
    print(f"  plain_mlp_concept held-out test: AUROC={heldout_metrics['auroc']:.4f} "
          f"F1={heldout_metrics['f1']:.4f}")

    transfer_metrics = {}
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
        m, _ = evaluate_mlp(mlp_model, X_ext, y_ext, thr)
        transfer_metrics[ext_name] = m
        print(f"    transfer -> {ext_name}: AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    all_results["plain_mlp_concept"] = {
        "cv_fold_metrics": fold_metrics, "cv_summary": summary,
        "heldout_test_metrics": heldout_metrics, "transfer_results": transfer_metrics,
    }

    # ================================================================
    # (B) RAW-FEATURE MLP -- ceiling reference, NOT transferable
    # ================================================================
    print("\n" + "=" * 60)
    print("(B) RAW-FEATURE MLP (ceiling reference -- NOT transferable)")
    print("=" * 60)
    print("  NOTE: this baseline uses the full un-pooled feature vector and "
          "CANNOT be evaluated on the transfer sets (different raw dimensionality). "
          "Reported only on the train-domain held-out test as a reference point.")

    fold_metrics_raw = run_cv_mlp(X_train_raw, y_train, cv_folds, "raw_mlp (no concept pooling)")
    summary_raw = summarize(fold_metrics_raw)
    print(f"  raw_mlp CV summary: AUROC={summary_raw['auroc']['mean']:.4f}+/-"
          f"{summary_raw['auroc']['std']:.4f} AUPRC={summary_raw['auprc']['mean']:.4f}")

    raw_mlp_model = train_mlp(X_train_raw[perm[:split]], y_train[perm[:split]],
                                X_train_raw[perm[split:]], y_train[perm[split:]])
    with torch.no_grad():
        device = next(raw_mlp_model.parameters()).device
        val_probs_raw = torch.sigmoid(raw_mlp_model(torch.tensor(
            X_train_raw[perm[split:]], dtype=torch.float32, device=device))).cpu().numpy()
    thr_raw = tune_threshold(y_train[perm[split:]], val_probs_raw)
    heldout_metrics_raw, _ = evaluate_mlp(raw_mlp_model, X_heldout_raw, y_heldout, thr_raw)
    print(f"  raw_mlp held-out test: AUROC={heldout_metrics_raw['auroc']:.4f} "
          f"F1={heldout_metrics_raw['f1']:.4f}  (not transfer-tested -- see note above)")

    all_results["raw_mlp_not_transferable"] = {
        "cv_fold_metrics": fold_metrics_raw, "cv_summary": summary_raw,
        "heldout_test_metrics": heldout_metrics_raw,
        "transfer_results": "N/A - architecturally incompatible with other datasets' "
                             "raw feature dimensionality",
    }

    with open(os.path.join(out_dir, "baseline_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll baseline results saved to: {os.path.join(out_dir, 'baseline_results.json')}")
    print("\nCompare these against your KG-TransNet results.json to see whether the "
          "graph-attention layer adds value over plain classifiers on the same input.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./preprocessed")
    parser.add_argument("--out_dir", type=str, default="./model_output")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
