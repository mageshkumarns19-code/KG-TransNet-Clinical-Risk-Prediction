"""
Generate 17 figures for the KG-TransNet study
================================================
Run AFTER preprocess_datasets.py, train_model.py, train_baselines.py, and
ensemble_check.py have all been run once (this script reads their saved
outputs and, where needed, reloads the trained model / refits the fast
sklearn baselines to get raw probability scores for ROC/PR curves).

Figures produced (saved as numbered PNGs in --out_dir):
  01_dataset_sizes.png              - sample counts per dataset
  02_class_distribution.png         - positive/negative balance per dataset
  03_heart_missingness.png          - % missing values per heart-disease site
  04_data_split_diagram.png         - train/CV/held-out/transfer split sizes
  05_roc_heldout.png                - ROC curves, held-out test, all models
  06_pr_heldout.png                 - PR curves, held-out test, all models
  07_roc_transfer_cleveland.png     - ROC curves, Cleveland transfer
  08_roc_transfer_other_sites.png   - ROC curves, other heart sites transfer
  09_roc_transfer_mimic.png         - ROC curves, MIMIC transfer
  10_cv_fold_auroc_boxplot.png      - CV fold AUROC spread per model
  11_heldout_auroc_bar.png          - held-out AUROC, all models, bar chart
  12_transfer_auroc_grouped_bar.png - transfer AUROC, grouped by dataset/model
  13_training_curve.png             - KG-TransNet loss & val AUROC vs epoch
  14_concept_importance.png         - mean concept attention weights
  15_kg_concept_graph.png           - concept graph / adjacency diagram
  16_confusion_matrix.png           - KG-TransNet confusion matrix, held-out
  17_ensemble_comparison.png        - KG-TransNet vs RF vs ensemble, all sets

Run:
    pip install matplotlib scikit-learn torch numpy --break-system-packages
    python generate_figures.py --data_dir "./preprocessed" --model_dir "./model_output" --out_dir "./figures"
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from model import KGTransNet, build_concept_membership, pool_to_concepts, CONCEPTS, ADJACENCY

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def main(data_dir, model_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # ---- Load everything we need ----
    train_pool = np.load(os.path.join(data_dir, "train_pool.npz"))
    X_train_raw, y_train = train_pool["X_train"], train_pool["y_train"]
    X_heldout_raw, y_heldout = train_pool["X_heldout_test"], train_pool["y_heldout_test"]

    with open(os.path.join(data_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(data_dir, "cv_folds.json")) as f:
        cv_folds = json.load(f)
    with open(os.path.join(model_dir, "results.json")) as f:
        kg_results = json.load(f)
    with open(os.path.join(model_dir, "baseline_results.json")) as f:
        baseline_results = json.load(f)
    ensemble_path = os.path.join(model_dir, "ensemble_check_results.json")
    ensemble_results = None
    if os.path.exists(ensemble_path):
        with open(ensemble_path) as f:
            ensemble_results = json.load(f)

    membership = build_concept_membership(feature_names)
    X_train = pool_to_concepts(X_train_raw, membership)
    X_heldout = pool_to_concepts(X_heldout_raw, membership)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg_model = KGTransNet(num_concepts=len(CONCEPTS), emb_dim=32).to(device)
    kg_model.load_state_dict(torch.load(
        os.path.join(model_dir, "kg_transnet_final.pt"), map_location=device))
    kg_model.eval()

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    def kg_probs(X):
        with torch.no_grad():
            logits, concept_importance, _ = kg_model(
                torch.tensor(X, dtype=torch.float32, device=device))
            return (torch.sigmoid(logits).cpu().numpy(),
                    concept_importance.cpu().numpy())

    def load_transfer(name):
        npz_path = os.path.join(data_dir, f"transfer_{name}.npz")
        feat_path = os.path.join(data_dir, f"transfer_{name}_features.json")
        d = np.load(npz_path)
        with open(feat_path) as f:
            names = json.load(f)
        mem = build_concept_membership(names)
        X_pooled = pool_to_concepts(d["X"], mem)
        return X_pooled, d["y"]

    X_cleve, y_cleve = load_transfer("heart_cleveland")
    X_other, y_other = load_transfer("heart_other_sites")
    X_mimic, y_mimic = load_transfer("mimic")

    print("Generating figures...")

    # ================================================================
    # 01. Dataset sizes
    # ================================================================
    names = ["Diabetes-130", "CKD", "Heart\nCleveland", "Heart\nOther sites", "MIMIC-III\nDemo"]
    sizes = [101766, 400, 303, 617, 129]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, sizes, color=["#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#DD8452"])
    for b, s in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width()/2, s + max(sizes)*0.01, str(s),
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of samples")
    ax.set_title("Dataset sizes (blue = train pool, orange = transfer test)")
    savefig(fig, out_dir, "01_dataset_sizes.png")

    # ================================================================
    # 02. Class distribution per dataset
    # ================================================================
    datasets_y = {"Diabetes+CKD\n(train pool)": y_train, "Held-out test": y_heldout,
                  "Heart Cleveland": y_cleve, "Heart Other sites": y_other,
                  "MIMIC": y_mimic}
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(datasets_y.keys())
    pos_rates = [y.mean() for y in datasets_y.values()]
    neg_rates = [1 - p for p in pos_rates]
    x = np.arange(len(labels))
    ax.bar(x, neg_rates, label="Negative", color="#8DA0CB")
    ax.bar(x, pos_rates, bottom=neg_rates, label="Positive", color="#FC8D62")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Class balance across datasets")
    ax.legend()
    savefig(fig, out_dir, "02_class_distribution.png")

    # ================================================================
    # 03. Heart disease site missingness (from confirmed data audit)
    # ================================================================
    sites = ["Cleveland", "Hungarian", "Switzerland", "VA"]
    missing_pct = [6/303*100, 293/294*100, 123/123*100, 199/200*100]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sites, missing_pct, color="#E15759")
    for b, p in zip(bars, missing_pct):
        ax.text(b.get_x() + b.get_width()/2, p + 1, f"{p:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("% rows with missing values")
    ax.set_title("Missing-data rate by Heart Disease site")
    ax.set_ylim(0, 110)
    savefig(fig, out_dir, "03_heart_missingness.png")

    # ================================================================
    # 04. Data split diagram
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 4))
    split_labels = ["Train\n(CV pool)", "Held-out\ntest", "Heart\nCleveland\n(transfer)",
                     "Heart other\nsites (transfer)", "MIMIC\n(transfer)"]
    split_sizes = [len(X_train), len(X_heldout), len(X_cleve), len(X_other), len(X_mimic)]
    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#DD8452"]
    ax.barh(split_labels, split_sizes, color=colors)
    for i, s in enumerate(split_sizes):
        ax.text(s + max(split_sizes)*0.01, i, str(s), va="center", fontsize=9)
    ax.set_xlabel("Number of samples")
    ax.set_title("Experimental split sizes (blue = train-domain, orange = zero-shot transfer)")
    savefig(fig, out_dir, "04_data_split_diagram.png")

    # ================================================================
    # 05 & 06. ROC and PR curves on held-out test
    # ================================================================
    rf_probs_heldout = rf.predict_proba(X_heldout)[:, 1]
    kg_probs_heldout, _ = kg_probs(X_heldout)

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, probs in [("KG-TransNet", kg_probs_heldout), ("Random Forest", rf_probs_heldout)]:
        fpr, tpr, _ = roc_curve(y_heldout, probs)
        ax.plot(fpr, tpr, label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve - Held-out internal test")
    ax.legend()
    savefig(fig, out_dir, "05_roc_heldout.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, probs in [("KG-TransNet", kg_probs_heldout), ("Random Forest", rf_probs_heldout)]:
        prec, rec, _ = precision_recall_curve(y_heldout, probs)
        ax.plot(rec, prec, label=name, linewidth=2)
    ax.axhline(y_heldout.mean(), color="k", linestyle="--", linewidth=1, label="Baseline (prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve - Held-out internal test")
    ax.legend()
    savefig(fig, out_dir, "06_pr_heldout.png")

    # ================================================================
    # 07-09. ROC curves on each transfer set
    # ================================================================
    transfer_sets = [("heart_cleveland", X_cleve, y_cleve, "07_roc_transfer_cleveland.png",
                       "ROC curve - Heart Disease (Cleveland) transfer"),
                      ("heart_other_sites", X_other, y_other, "08_roc_transfer_other_sites.png",
                       "ROC curve - Heart Disease (other sites) transfer"),
                      ("mimic", X_mimic, y_mimic, "09_roc_transfer_mimic.png",
                       "ROC curve - MIMIC-III transfer")]
    for _, X_ext, y_ext, fname, title in transfer_sets:
        kg_p, _ = kg_probs(X_ext)
        rf_p = rf.predict_proba(X_ext)[:, 1]
        fig, ax = plt.subplots(figsize=(6, 6))
        for name, probs in [("KG-TransNet", kg_p), ("Random Forest", rf_p)]:
            fpr, tpr, _ = roc_curve(y_ext, probs)
            ax.plot(fpr, tpr, label=name, linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        savefig(fig, out_dir, fname)

    # ================================================================
    # 10. CV fold AUROC boxplot across models
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    model_cv_aurocs = {
        "KG-TransNet": [m["auroc"] for m in kg_results["cv_fold_metrics"]],
        "Logistic Reg.": [m["auroc"] for m in baseline_results["logistic_regression"]["cv_fold_metrics"]],
        "Random Forest": [m["auroc"] for m in baseline_results["random_forest"]["cv_fold_metrics"]],
        "Plain MLP": [m["auroc"] for m in baseline_results["plain_mlp_concept"]["cv_fold_metrics"]],
    }
    try:
        ax.boxplot(model_cv_aurocs.values(), tick_labels=list(model_cv_aurocs.keys()))
    except TypeError:
        ax.boxplot(model_cv_aurocs.values(), labels=list(model_cv_aurocs.keys()))
    ax.set_ylabel("AUROC")
    ax.set_title("5-fold CV AUROC distribution by model")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    savefig(fig, out_dir, "10_cv_fold_auroc_boxplot.png")

    # ================================================================
    # 11. Held-out AUROC bar chart, all models
    # ================================================================
    heldout_aurocs = {
        "KG-TransNet": kg_results["heldout_test_metrics"]["auroc"],
        "Logistic Reg.": baseline_results["logistic_regression"]["heldout_test_metrics"]["auroc"],
        "Random Forest": baseline_results["random_forest"]["heldout_test_metrics"]["auroc"],
        "Plain MLP": baseline_results["plain_mlp_concept"]["heldout_test_metrics"]["auroc"],
        "Raw MLP\n(not transferable)": baseline_results["raw_mlp_not_transferable"]["heldout_test_metrics"]["auroc"],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(heldout_aurocs.keys(), heldout_aurocs.values(),
                   color=["#4C72B0", "#8DA0CB", "#8DA0CB", "#8DA0CB", "#999999"])
    for b, v in zip(bars, heldout_aurocs.values()):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("AUROC")
    ax.set_title("Held-out internal test AUROC, all models")
    ax.set_ylim(0.4, max(heldout_aurocs.values()) + 0.08)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    savefig(fig, out_dir, "11_heldout_auroc_bar.png")

    # ================================================================
    # 12. Transfer AUROC grouped bar chart
    # ================================================================
    transfer_models = ["KG-TransNet", "Logistic Reg.", "Random Forest", "Plain MLP"]
    transfer_datasets_names = ["heart_cleveland", "heart_other_sites", "mimic"]
    transfer_matrix = np.zeros((len(transfer_models), len(transfer_datasets_names)))
    transfer_matrix[0] = [kg_results["transfer_results"][d]["auroc"] for d in transfer_datasets_names]
    transfer_matrix[1] = [baseline_results["logistic_regression"]["transfer_results"][d]["auroc"] for d in transfer_datasets_names]
    transfer_matrix[2] = [baseline_results["random_forest"]["transfer_results"][d]["auroc"] for d in transfer_datasets_names]
    transfer_matrix[3] = [baseline_results["plain_mlp_concept"]["transfer_results"][d]["auroc"] for d in transfer_datasets_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(transfer_datasets_names))
    width = 0.2
    for i, model_name in enumerate(transfer_models):
        ax.bar(x + i*width, transfer_matrix[i], width, label=model_name)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Heart Cleveland", "Heart Other sites", "MIMIC"])
    ax.set_ylabel("AUROC")
    ax.set_title("Zero-shot transfer AUROC by model and dataset")
    ax.legend()
    savefig(fig, out_dir, "12_transfer_auroc_grouped_bar.png")

    # ================================================================
    # 13. Training curve is not logged per-epoch in results.json, so we
    # retrain a quick instrumented copy just for plotting purposes.
    # ================================================================
    from train_model import train_one_model
    import io, contextlib
    n = len(X_train)
    perm = np.random.RandomState(RANDOM_STATE).permutation(n)
    split = int(0.9 * n)

    log_lines = []
    device_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tmp = KGTransNet(num_concepts=len(CONCEPTS), emb_dim=32).to(device_cpu)
    optimizer = torch.optim.Adam(model_tmp.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    pos_rate = max(y_train[perm[:split]].mean(), 1e-6)
    pos_weight = torch.tensor([min((1 - pos_rate) / pos_rate, 3.0)], device=device_cpu)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    X_tr_t = torch.tensor(X_train[perm[:split]], dtype=torch.float32, device=device_cpu)
    y_tr_t = torch.tensor(y_train[perm[:split]], dtype=torch.float32, device=device_cpu)
    X_val_t = torch.tensor(X_train[perm[split:]], dtype=torch.float32, device=device_cpu)
    y_val_arr = y_train[perm[split:]]

    from sklearn.metrics import roc_auc_score
    losses, val_aucs = [], []
    n_tr = X_tr_t.shape[0]
    best_auc, no_improve = -1, 0
    for epoch in range(150):
        model_tmp.train()
        perm_e = torch.randperm(n_tr)
        ep_loss = 0.0
        for start in range(0, n_tr, 128):
            idx = perm_e[start:start+128]
            optimizer.zero_grad()
            logits, _, _ = model_tmp(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        model_tmp.eval()
        with torch.no_grad():
            val_logits, _, _ = model_tmp(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_val_arr, val_probs) if len(np.unique(y_val_arr)) > 1 else 0.5
        scheduler.step(val_auc)
        losses.append(ep_loss / n_tr)
        val_aucs.append(val_auc)
        if val_auc > best_auc + 1e-4:
            best_auc, no_improve = val_auc, 0
        else:
            no_improve += 1
        if no_improve >= 15:
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(losses, color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_title("Training loss")
    ax2.plot(val_aucs, color="#DD8452")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUROC")
    ax2.set_title("Validation AUROC")
    fig.suptitle("KG-TransNet training curve (illustrative re-run)")
    savefig(fig, out_dir, "13_training_curve.png")

    # ================================================================
    # 14. Concept importance (mean attention pooling weights)
    # ================================================================
    _, concept_importance = kg_probs(X_heldout)
    mean_importance = concept_importance.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(mean_importance)[::-1]
    ax.bar(np.array(CONCEPTS)[order], mean_importance[order], color="#55A868")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Mean concept importance (held-out test)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    savefig(fig, out_dir, "14_concept_importance.png")

    # ================================================================
    # 15. KG concept graph diagram
    # ================================================================
    fig, ax = plt.subplots(figsize=(6, 6))
    n_c = len(CONCEPTS)
    angles = np.linspace(0, 2*np.pi, n_c, endpoint=False)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(CONCEPTS, angles)}
    for i in range(n_c):
        for j in range(i+1, n_c):
            if ADJACENCY[i, j] > 0:
                x1, y1 = pos[CONCEPTS[i]]
                x2, y2 = pos[CONCEPTS[j]]
                ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.3, linewidth=1, zorder=1)
    for c in CONCEPTS:
        x, y = pos[c]
        ax.add_patch(Circle((x, y), 0.13, color="#4C72B0", zorder=2))
        ax.text(x, y - 0.28, c, ha="center", va="center", color="black", fontsize=9, zorder=3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Shared concept graph (comorbidity adjacency)")
    savefig(fig, out_dir, "15_kg_concept_graph.png")

    # ================================================================
    # 16. Confusion matrix, KG-TransNet, held-out test (tuned threshold)
    # ================================================================
    thr = kg_results["heldout_test_metrics"]["threshold_used"]
    preds = (kg_probs_heldout >= thr).astype(int)
    cm = confusion_matrix(y_heldout, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(f"KG-TransNet confusion matrix (threshold={thr:.2f})")
    savefig(fig, out_dir, "16_confusion_matrix.png")

    # ================================================================
    # 17. Ensemble comparison (if available)
    # ================================================================
    if ensemble_results is not None:
        eval_sets = list(ensemble_results.keys())
        model_names = ["kg_transnet", "random_forest", "ensemble_avg"]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(eval_sets))
        width = 0.25
        for i, mname in enumerate(model_names):
            vals = [ensemble_results[ds][mname]["auroc"] for ds in eval_sets]
            ax.bar(x + i*width, vals, width, label=mname)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(x + width)
        ax.set_xticklabels(eval_sets, rotation=15, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_title("KG-TransNet vs Random Forest vs Ensemble (all evaluation sets)")
        ax.legend()
        savefig(fig, out_dir, "17_ensemble_comparison.png")
    else:
        print("  skipped 17_ensemble_comparison.png (ensemble_check_results.json not found)")

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./preprocessed")
    parser.add_argument("--model_dir", type=str, default="./model_output")
    parser.add_argument("--out_dir", type=str, default="./figures")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.out_dir)
