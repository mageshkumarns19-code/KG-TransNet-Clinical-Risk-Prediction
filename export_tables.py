"""
Export all results to CSV tables
===================================
Reads the JSON outputs already produced by train_model.py, train_baselines.py,
and ensemble_check.py, and writes clean, paper-ready CSV tables.

Run AFTER train_model.py, train_baselines.py, and ensemble_check.py:
    python export_tables.py --model_dir "./model_output" --out_dir "./tables"

Produces:
  table_01_cv_fold_metrics.csv       - per-fold metrics, every model
  table_02_cv_summary.csv            - mean +/- std CV metrics, every model
  table_03_heldout_test.csv          - held-out test metrics, every model
  table_04_transfer_results.csv      - transfer metrics, every model x dataset
  table_05_ensemble_comparison.csv   - KG-TransNet vs RF vs ensemble, all sets
  table_06_dataset_summary.csv       - dataset sizes / class balance overview
"""

import argparse
import json
import os
import csv


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved: {os.path.basename(path)}  ({len(rows)} rows)")


def main(model_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    kg_results = load_json(os.path.join(model_dir, "results.json"))
    baseline_results = load_json(os.path.join(model_dir, "baseline_results.json"))
    ensemble_path = os.path.join(model_dir, "ensemble_check_results.json")
    ensemble_results = load_json(ensemble_path) if os.path.exists(ensemble_path) else None

    # Combine KG-TransNet + baselines into one dict for uniform iteration
    all_models = {"KG-TransNet": kg_results}
    all_models.update({
        "Logistic Regression": baseline_results.get("logistic_regression"),
        "Random Forest": baseline_results.get("random_forest"),
        "Plain MLP (concept)": baseline_results.get("plain_mlp_concept"),
        "Raw MLP (not transferable)": baseline_results.get("raw_mlp_not_transferable"),
    })

    print("Exporting CSV tables...")

    # ================================================================
    # Table 1: per-fold CV metrics, every model
    # ================================================================
    rows = []
    for model_name, res in all_models.items():
        if res is None or "cv_fold_metrics" not in res:
            continue
        for m in res["cv_fold_metrics"]:
            rows.append({
                "model": model_name,
                "fold": m.get("fold", ""),
                "auroc": m.get("auroc", ""),
                "auprc": m.get("auprc", ""),
                "f1": m.get("f1", ""),
                "precision": m.get("precision", ""),
                "recall": m.get("recall", ""),
                "threshold_used": m.get("threshold_used", ""),
                "n": m.get("n", ""),
                "positive_rate": m.get("positive_rate", ""),
            })
    write_csv(rows, os.path.join(out_dir, "table_01_cv_fold_metrics.csv"),
              fieldnames=["model", "fold", "auroc", "auprc", "f1", "precision",
                          "recall", "threshold_used", "n", "positive_rate"])

    # ================================================================
    # Table 2: CV summary (mean +/- std), every model
    # ================================================================
    rows = []
    for model_name, res in all_models.items():
        if res is None or "cv_summary" not in res:
            continue
        summary = res["cv_summary"]
        row = {"model": model_name}
        for key in ["auroc", "auprc", "f1", "precision", "recall"]:
            if key in summary:
                row[f"{key}_mean"] = summary[key]["mean"]
                row[f"{key}_std"] = summary[key]["std"]
        rows.append(row)
    fieldnames = ["model"] + [f"{k}_{stat}" for k in ["auroc", "auprc", "f1", "precision", "recall"]
                               for stat in ["mean", "std"]]
    write_csv(rows, os.path.join(out_dir, "table_02_cv_summary.csv"), fieldnames=fieldnames)

    # ================================================================
    # Table 3: held-out test metrics, every model
    # ================================================================
    rows = []
    for model_name, res in all_models.items():
        if res is None or "heldout_test_metrics" not in res:
            continue
        m = res["heldout_test_metrics"]
        rows.append({
            "model": model_name,
            "auroc": m.get("auroc", ""),
            "auprc": m.get("auprc", ""),
            "f1": m.get("f1", ""),
            "precision": m.get("precision", ""),
            "recall": m.get("recall", ""),
            "threshold_used": m.get("threshold_used", ""),
            "n": m.get("n", ""),
            "positive_rate": m.get("positive_rate", ""),
        })
    write_csv(rows, os.path.join(out_dir, "table_03_heldout_test.csv"),
              fieldnames=["model", "auroc", "auprc", "f1", "precision", "recall",
                          "threshold_used", "n", "positive_rate"])

    # ================================================================
    # Table 4: transfer results, every model x dataset (long format)
    # ================================================================
    rows = []
    for model_name, res in all_models.items():
        if res is None:
            continue
        transfer = res.get("transfer_results", {})
        if not isinstance(transfer, dict):
            continue  # e.g. raw_mlp's "N/A" string
        for dataset_name, m in transfer.items():
            rows.append({
                "model": model_name,
                "transfer_dataset": dataset_name,
                "auroc": m.get("auroc", ""),
                "auprc": m.get("auprc", ""),
                "f1": m.get("f1", ""),
                "precision": m.get("precision", ""),
                "recall": m.get("recall", ""),
                "threshold_used": m.get("threshold_used", ""),
                "n": m.get("n", ""),
                "positive_rate": m.get("positive_rate", ""),
            })
    write_csv(rows, os.path.join(out_dir, "table_04_transfer_results.csv"),
              fieldnames=["model", "transfer_dataset", "auroc", "auprc", "f1",
                          "precision", "recall", "threshold_used", "n", "positive_rate"])

    # ================================================================
    # Table 5: ensemble comparison (if available)
    # ================================================================
    if ensemble_results is not None:
        rows = []
        for dataset_name, model_results in ensemble_results.items():
            for model_name, m in model_results.items():
                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "auroc": m.get("auroc", ""),
                    "auprc": m.get("auprc", ""),
                    "f1": m.get("f1", ""),
                })
        write_csv(rows, os.path.join(out_dir, "table_05_ensemble_comparison.csv"),
                  fieldnames=["dataset", "model", "auroc", "auprc", "f1"])
    else:
        print("  skipped table_05_ensemble_comparison.csv (ensemble_check_results.json not found)")

    # ================================================================
    # Table 6: dataset summary (confirmed counts from the data audit)
    # ================================================================
    rows = [
        {"dataset": "Diabetes-130 (train pool)", "n_samples": 101766,
         "n_unique_patients": 71518, "positive_rate": 0.1135, "role": "train/CV"},
        {"dataset": "Chronic Kidney Disease (train pool)", "n_samples": 400,
         "n_unique_patients": 400, "positive_rate": 0.625, "role": "train/CV"},
        {"dataset": "Heart Disease - Cleveland", "n_samples": 303,
         "n_unique_patients": 303, "positive_rate": 0.459, "role": "zero-shot transfer test"},
        {"dataset": "Heart Disease - Other sites (Hungarian+Switzerland+VA)",
         "n_samples": 617, "n_unique_patients": 617, "positive_rate": 0.600,
         "role": "zero-shot transfer test"},
        {"dataset": "MIMIC-III Demo", "n_samples": 129, "n_unique_patients": 100,
         "positive_rate": 0.085, "role": "zero-shot transfer test"},
    ]
    write_csv(rows, os.path.join(out_dir, "table_06_dataset_summary.csv"),
              fieldnames=["dataset", "n_samples", "n_unique_patients", "positive_rate", "role"])

    print(f"\nAll tables saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./model_output")
    parser.add_argument("--out_dir", type=str, default="./tables")
    args = parser.parse_args()
    main(args.model_dir, args.out_dir)
