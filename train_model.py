"""
Train KG-TransNet
==================
Experimental protocol:
  1. Load train_pool.npz (from preprocess_datasets.py) + cv_folds.json
  2. Map raw train-pool features into the shared concept space
  3. 5-fold stratified CV on the train portion -> report mean +/- std metrics
  4. Train a FINAL model on the full train portion (all 5 folds' data)
  5. Evaluate the final model on:
       - held-out internal test split (diabetes+CKD, unseen during training)
       - external transfer sets (heart_cleveland, heart_other_sites, mimic) --
         these go through their OWN concept-pooling (their own raw feature names)
         but the SAME trained shared model (no retraining) -- this is the
         zero-shot cross-institution transfer evaluation.
  6. Extract human-readable rules from a surrogate decision tree trained on
     concept-level scores vs. the model's predictions (explanation mechanism).

Run:
    pip install torch scikit-learn numpy --break-system-packages
    python train_model.py --data_dir "./preprocessed"
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, average_precision_score)
from sklearn.tree import DecisionTreeClassifier, export_text

from model import KGTransNet, build_concept_membership, pool_to_concepts, CONCEPTS

RANDOM_STATE = 42
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 128
EMB_DIM = 32


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_model(X_train, y_train, X_val, y_val, epochs=EPOCHS, verbose=False,
                     patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KGTransNet(num_concepts=len(CONCEPTS), emb_dim=EMB_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    # Class weighting for imbalance (readmission positive rate is low).
    # NOTE: an uncapped pos_weight (e.g. ~8x for an 11% positive rate) was tried
    # first and caused the model to collapse into predicting "positive" for
    # almost every patient (recall ~1.0, precision ~0.11, AUROC near chance) --
    # a degenerate shortcut rather than genuine discrimination. Capping the
    # weight keeps the imbalance correction useful without that collapse.
    pos_rate = max(y_train.mean(), 1e-6)
    raw_pos_weight = (1 - pos_rate) / pos_rate
    capped_pos_weight = min(raw_pos_weight, 3.0)
    pos_weight = torch.tensor([capped_pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    n = X_train_t.shape[0]
    best_val_auc = -1.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits, _, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
        scheduler.step(val_auc)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    epoch {epoch:3d}  loss={epoch_loss/n:.4f}  "
                  f"val_auc={val_auc:.4f}  lr={current_lr:.6f}")

        if epochs_since_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch} (no improvement for "
                      f"{patience} epochs)")
            break

    model.load_state_dict(best_state)
    return model, best_val_auc


def tune_threshold(y_true, probs):
    """Select the decision threshold that maximizes F1 on a validation set.
    NOTE: with a naive fixed 0.5 threshold under class imbalance, the model's
    predicted probabilities rarely cross 0.5 even when it IS ranking cases
    correctly (AUROC can look fine while F1/recall look collapsed at 0.5).
    Tuning the threshold separates 'is the model discriminating well' (AUROC)
    from 'what cutoff to act on' (F1/precision/recall), which is standard
    practice for imbalanced clinical prediction tasks."""
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


def evaluate(model, X, y, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits, concept_importance, _ = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auroc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": average_precision_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        "f1": f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "threshold_used": float(threshold),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
    return metrics, probs, concept_importance.cpu().numpy()


def run_cv(X_train_pool, y_train_pool, cv_folds):
    print("\n=== 5-FOLD CROSS-VALIDATION (train pool) ===")
    fold_metrics = []
    for fold in cv_folds:
        tr_idx, val_idx = np.array(fold["train_idx"]), np.array(fold["val_idx"])
        model, best_auc = train_one_model(
            X_train_pool[tr_idx], y_train_pool[tr_idx],
            X_train_pool[val_idx], y_train_pool[val_idx],
        )
        val_probs_for_thr = torch.sigmoid(
            model(torch.tensor(X_train_pool[val_idx], dtype=torch.float32,
                                device=next(model.parameters()).device))[0]
        ).detach().cpu().numpy()
        tuned_thr = tune_threshold(y_train_pool[val_idx], val_probs_for_thr)
        metrics, _, _ = evaluate(model, X_train_pool[val_idx], y_train_pool[val_idx],
                                   threshold=tuned_thr)
        fold_metrics.append(metrics)
        print(f"  Fold {fold['fold']}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
              f"F1={metrics['f1']:.4f} Prec={metrics['precision']:.4f} "
              f"Recall={metrics['recall']:.4f} (thr={tuned_thr:.2f})")

    print("\n  --- CV summary (mean +/- std) ---")
    summary = {}
    for key in ["auroc", "auprc", "f1", "precision", "recall"]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {key:10s}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
    return fold_metrics, summary


def extract_rules(model, X, y, concept_names=CONCEPTS, max_depth=4, threshold=0.5):
    """Surrogate-model distillation: train an interpretable decision tree on the
    concept-level pooled statistics to approximate the neural model's predictions.
    This produces human-readable IF-THEN rules over clinical concepts, which is
    the explanation mechanism (distinct from attention-weight or path-based
    explanations already common in the KG-XAI literature)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits, _, _ = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    surrogate_labels = (probs >= threshold).astype(int)

    stat_names = ["mean", "max", "abnormal_share"]
    expanded_names = [f"{c}_{s}" for c in concept_names for s in stat_names]

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE,
                                    class_weight="balanced")
    tree.fit(X, surrogate_labels)
    fidelity = tree.score(X, surrogate_labels)  # how well the tree mimics the NN
    rules_text = export_text(tree, feature_names=expanded_names)
    return rules_text, fidelity


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    set_seed()

    print("Loading preprocessed train pool...")
    train_pool = np.load(os.path.join(data_dir, "train_pool.npz"))
    X_train_raw, y_train = train_pool["X_train"], train_pool["y_train"]
    X_heldout_raw, y_heldout = train_pool["X_heldout_test"], train_pool["y_heldout_test"]

    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(data_dir, "cv_folds.json")) as f:
        cv_folds = json.load(f)

    # ---- Concept-pool the train pool features (shared concept space) ----
    membership = build_concept_membership(feature_names)
    X_train = pool_to_concepts(X_train_raw, membership)
    X_heldout = pool_to_concepts(X_heldout_raw, membership)
    print(f"Concept-pooled train shape: {X_train.shape} "
          f"(from {X_train_raw.shape[1]} raw features -> {len(CONCEPTS)} concepts)")

    # ---- 5-fold CV ----
    fold_metrics, cv_summary = run_cv(X_train, y_train, cv_folds)

    # ---- Train FINAL model on the full train portion ----
    print("\n=== Training FINAL model on full train portion ===")
    # Use a small internal split just for early-stopping selection during final fit
    n = len(X_train)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    split = int(0.9 * n)
    final_model, _ = train_one_model(
        X_train[perm[:split]], y_train[perm[:split]],
        X_train[perm[split:]], y_train[perm[split:]],
        verbose=True,
    )

    # Tune the decision threshold on the internal validation split used above
    # (NOT on the held-out test or transfer sets, to avoid leakage), then reuse
    # this single threshold consistently everywhere below.
    device = next(final_model.parameters()).device
    with torch.no_grad():
        val_probs = torch.sigmoid(final_model(
            torch.tensor(X_train[perm[split:]], dtype=torch.float32, device=device)
        )[0]).cpu().numpy()
    tuned_threshold = tune_threshold(y_train[perm[split:]], val_probs)
    print(f"\nTuned decision threshold (from internal validation): {tuned_threshold:.3f}")

    # ---- Evaluate on held-out internal test ----
    print("\n=== HELD-OUT INTERNAL TEST (diabetes+CKD, unseen) ===")
    heldout_metrics, _, _ = evaluate(final_model, X_heldout, y_heldout,
                                       threshold=tuned_threshold)
    for k, v in heldout_metrics.items():
        print(f"  {k}: {v}")

    # ---- Evaluate on external TRANSFER sets (zero-shot, own concept pooling) ----
    print("\n=== TRANSFER TEST (external datasets, zero-shot) ===")
    transfer_results = {}
    for name in ["heart_cleveland", "heart_other_sites", "mimic"]:
        npz_path = os.path.join(data_dir, f"transfer_{name}.npz")
        feat_path = os.path.join(data_dir, f"transfer_{name}_features.json")
        if not os.path.exists(npz_path):
            print(f"  {name}: file not found, skipping")
            continue
        transfer_data = np.load(npz_path)
        X_ext_raw, y_ext = transfer_data["X"], transfer_data["y"]
        with open(feat_path) as f:
            ext_feature_names = json.load(f)

        ext_membership = build_concept_membership(ext_feature_names)
        X_ext = pool_to_concepts(X_ext_raw, ext_membership)

        metrics, _, _ = evaluate(final_model, X_ext, y_ext, threshold=tuned_threshold)
        transfer_results[name] = metrics
        print(f"  {name}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
              f"F1={metrics['f1']:.4f} (n={metrics['n']}, pos_rate={metrics['positive_rate']:.3f})")

    # ---- Rule extraction (explanation) ----
    print("\n=== EXTRACTED RULES (surrogate decision tree over concept scores) ===")
    rules_text, fidelity = extract_rules(final_model, X_heldout, y_heldout,
                                           threshold=tuned_threshold)
    print(f"  Surrogate fidelity to neural model: {fidelity:.4f}")
    print(rules_text)

    # ---- Save everything ----
    torch.save(final_model.state_dict(), os.path.join(out_dir, "kg_transnet_final.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "tuned_threshold": tuned_threshold,
            "cv_fold_metrics": fold_metrics,
            "cv_summary": cv_summary,
            "heldout_test_metrics": heldout_metrics,
            "transfer_results": transfer_results,
            "surrogate_rule_fidelity": fidelity,
        }, f, indent=2)
    with open(os.path.join(out_dir, "extracted_rules.txt"), "w") as f:
        f.write(rules_text)

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./preprocessed")
    parser.add_argument("--out_dir", type=str, default="./model_output")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
