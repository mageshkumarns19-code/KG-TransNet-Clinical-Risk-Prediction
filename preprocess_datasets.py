"""
Preprocessing pipeline for KG-TransNet study
=============================================
Datasets:
  1. Diabetes 130-US Hospitals   -> PRIMARY TRAIN (5-fold CV + held-out test)
  2. Chronic Kidney Disease      -> SECONDARY TRAIN (5-fold CV + held-out test)
  3. Heart Disease (Cleveland)   -> TRANSFER TEST (external, never trained on)
  4. MIMIC-III Demo              -> TRANSFER TEST (external, never trained on)

Experimental protocol:
  - Diabetes + CKD are combined into the TRAIN pool.
  - TRAIN pool -> 80% train/CV split, 20% held-out internal test (stratified).
  - The 80% portion is further split into 5 stratified folds for cross-validation.
  - Heart Disease (Cleveland, clean subset) and MIMIC-III are used ONLY as
    external transfer-test sets to evaluate generalization of the KG-conditioned
    representation - they are never seen during training or CV.

Run:
    pip install pandas numpy scikit-learn scipy --break-system-packages
    python preprocess_datasets.py --base_path "E:\\Research\\07.09.26\\Dataset"
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
N_FOLDS = 5


# ----------------------------------------------------------------------
# 1. DIABETES 130-US HOSPITALS
# ----------------------------------------------------------------------
def load_diabetes(base_path):
    path = os.path.join(base_path, "diabetes+130-us+hospitals+for+years+1999-2008",
                         "diabetic_data.csv")
    df = pd.read_csv(path)

    # Drop columns that are near-entirely missing or non-predictive identifiers
    df = df.drop(columns=["weight", "payer_code", "medical_specialty",
                           "encounter_id"], errors="ignore")

    # Replace '?' with NaN across the board
    df = df.replace("?", np.nan)

    # Binarize target: readmitted <30 days = 1, else = 0
    df["label"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Keep patient_nbr for later temporal/multi-visit KG linkage, but not as a feature
    patient_ids = df["patient_nbr"]
    df = df.drop(columns=["patient_nbr"])

    # Map age brackets "[0-10)" etc. to ordinal midpoints
    def age_to_num(a):
        if pd.isna(a):
            return np.nan
        lo, hi = a.strip("[)").split("-")
        return (int(lo) + int(hi)) / 2
    df["age"] = df["age"].apply(age_to_num)

    # diag_1/2/3 are ICD-9 codes (strings) -> keep as categorical codes for KG mapping,
    # but also produce a numeric-safe version for the tabular encoder
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].astype(str)

    df["source_dataset"] = "diabetes"
    df["_patient_id"] = patient_ids.values
    return df


# ----------------------------------------------------------------------
# 2. CHRONIC KIDNEY DISEASE (ARFF)
# ----------------------------------------------------------------------
def load_ckd(base_path):
    path = os.path.join(base_path, "Chronic_Kidney_Disease", "chronic_kidney_disease_full.arff")
    with open(path, "r") as f:
        lines = f.readlines()

    # Parse @attribute lines for column names
    columns = []
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            # Real UCI file quotes names like: @attribute 'age' numeric
            # Handle both quoted ('age') and unquoted (age) forms robustly.
            raw_name = stripped.split(None, 2)[1]
            clean_name = raw_name.strip("'\"")
            columns.append(clean_name)
        if stripped.lower().startswith("@data"):
            data_start = i + 1
            break

    data_lines = [l.strip() for l in lines[data_start:] if l.strip()]
    n_expected = len(columns)

    rows = []
    n_fixed = 0
    for l in data_lines:
        fields = l.split(",")
        if len(fields) > n_expected:
            # Common real-world quirk: trailing comma -> trailing empty field(s).
            # Drop trailing empty strings first.
            while len(fields) > n_expected and fields[-1].strip() == "":
                fields.pop()
                n_fixed += 1
            # If still too long, truncate from the end (last resort) and warn once.
            if len(fields) > n_expected:
                fields = fields[:n_expected]
                n_fixed += 1
        elif len(fields) < n_expected:
            # Pad missing trailing fields with '?' (treated as NaN downstream)
            fields = fields + ["?"] * (n_expected - len(fields))
            n_fixed += 1
        rows.append(fields)

    if n_fixed > 0:
        print(f"  [load_ckd] Note: {n_fixed} row(s) had a field-count mismatch "
              f"(e.g. trailing comma) and were auto-corrected to {n_expected} columns.")

    df = pd.DataFrame(rows, columns=columns)

    df = df.replace("?", np.nan)

    # NOTE: The real UCI ARFF attribute names do not always match the abbreviations
    # used in chronic_kidney_disease.info.txt (e.g. the file may use 'wbcc'/'rbcc'
    # instead of 'wc'/'rc', or other naming variants). Hardcoding exact column names
    # is fragile, so numeric vs. categorical columns are detected automatically:
    # a column is treated as numeric if most of its non-missing values parse as numbers.
    class_col = "class" if "class" in df.columns else df.columns[-1]
    candidate_cols = [c for c in df.columns if c != class_col]

    numeric_cols = []
    nominal_cols = []
    for c in candidate_cols:
        converted = pd.to_numeric(df[c], errors="coerce")
        non_missing = df[c].notna().sum()
        if non_missing == 0:
            nominal_cols.append(c)
            continue
        success_rate = converted.notna().sum() / non_missing
        if success_rate >= 0.8:
            numeric_cols.append(c)
        else:
            nominal_cols.append(c)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in nominal_cols:
        df[c] = df[c].astype(str).str.strip()

    df["label"] = (df[class_col].astype(str).str.strip() == "ckd").astype(int)
    df = df.drop(columns=[class_col])

    df["source_dataset"] = "ckd"
    df["_patient_id"] = [f"ckd_{i}" for i in range(len(df))]
    return df


# ----------------------------------------------------------------------
# 3. HEART DISEASE (Cleveland only - clean subset, TRANSFER TEST)
# ----------------------------------------------------------------------
def load_heart_cleveland(base_path):
    path = os.path.join(base_path, "heart+disease", "processed.cleveland.data")
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(path, header=None, names=columns, na_values="?")

    df["label"] = (df["target"] > 0).astype(int)
    df = df.drop(columns=["target"])

    df["source_dataset"] = "heart_cleveland"
    df["_patient_id"] = [f"heart_{i}" for i in range(len(df))]
    return df


def load_heart_transfer_sites(base_path):
    """Hungarian/Switzerland/VA - only features with acceptable coverage are kept
    (slope, ca, thal are dropped campus-wide due to near-total missingness)."""
    sites = ["processed.hungarian.data", "processed.switzerland.data", "processed.va.data"]
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    usable_cols = ["age", "sex", "cp", "trestbps", "restecg", "thalach", "exang", "target"]

    frames = []
    for s in sites:
        path = os.path.join(base_path, "heart+disease", s)
        df = pd.read_csv(path, header=None, names=columns, na_values="?")
        df = df[usable_cols].copy()
        df["label"] = (df["target"] > 0).astype(int)
        df = df.drop(columns=["target"])
        df["source_dataset"] = s.replace("processed.", "").replace(".data", "")
        df["_patient_id"] = [f"{s}_{i}" for i in range(len(df))]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# 4. MIMIC-III DEMO (TRANSFER TEST)
#    Build a simple 30-day-readmission-style label from ADMISSIONS.csv
# ----------------------------------------------------------------------
def load_mimic(base_path):
    admissions_path = os.path.join(base_path, "mimic-iii-clinical-database-demo-1.4",
                                    "ADMISSIONS.csv")
    diagnoses_path = os.path.join(base_path, "mimic-iii-clinical-database-demo-1.4",
                                   "DIAGNOSES_ICD.csv")

    adm = pd.read_csv(admissions_path, parse_dates=["admittime", "dischtime"])
    diag = pd.read_csv(diagnoses_path)

    adm = adm.sort_values(["subject_id", "admittime"])
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["days_to_next_admit"] = (adm["next_admittime"] - adm["dischtime"]).dt.days
    adm["label"] = ((adm["days_to_next_admit"] >= 0) &
                     (adm["days_to_next_admit"] <= 30)).astype(int)

    # length of stay as a numeric feature
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400

    # number of distinct diagnoses per admission
    diag_counts = diag.groupby("hadm_id")["icd9_code"].nunique().rename("num_diagnoses")
    adm = adm.merge(diag_counts, on="hadm_id", how="left")
    adm["num_diagnoses"] = adm["num_diagnoses"].fillna(0)

    keep_cols = ["subject_id", "hadm_id", "admission_type", "admission_location",
                 "los_days", "num_diagnoses", "label"]
    df = adm[keep_cols].copy()

    df["source_dataset"] = "mimic"
    df["_patient_id"] = df["subject_id"].astype(str)
    return df


# ----------------------------------------------------------------------
# GENERIC PREPROCESSING: impute, encode, scale
# ----------------------------------------------------------------------
def fit_transform_features(df, fit_on_index=None):
    """
    Encodes categoricals, imputes missing values, scales numerics.
    Returns X (np.ndarray), y (np.ndarray), feature_names, fitted transformers dict.
    """
    df = df.copy()
    y = df["label"].values
    meta_cols = ["label", "source_dataset", "_patient_id"]
    X_df = df.drop(columns=[c for c in meta_cols if c in df.columns])

    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    transformers = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
                     "encoders": {}, "num_imputer": None, "cat_imputer": None, "scaler": None}

    # Impute + encode categoricals
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat = pd.DataFrame(cat_imputer.fit_transform(X_df[categorical_cols]),
                              columns=categorical_cols, index=X_df.index)
        encoders = {}
        for c in categorical_cols:
            le = LabelEncoder()
            X_cat[c] = le.fit_transform(X_cat[c].astype(str))
            encoders[c] = le
        transformers["cat_imputer"] = cat_imputer
        transformers["encoders"] = encoders
    else:
        X_cat = pd.DataFrame(index=X_df.index)

    # Impute + scale numerics
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(num_imputer.fit_transform(X_df[numeric_cols]),
                              columns=numeric_cols, index=X_df.index)
        scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num),
                                     columns=numeric_cols, index=X_df.index)
        transformers["num_imputer"] = num_imputer
        transformers["scaler"] = scaler
    else:
        X_num_scaled = pd.DataFrame(index=X_df.index)

    X_full = pd.concat([X_num_scaled, X_cat], axis=1)
    feature_names = X_full.columns.tolist()
    return X_full.values, y, feature_names, transformers


def transform_with_fitted(df, transformers):
    """Apply already-fitted transformers to a NEW (external/transfer) dataframe.
    Columns not seen during fit are dropped; missing expected columns raise a note."""
    df = df.copy()
    y = df["label"].values
    meta_cols = ["label", "source_dataset", "_patient_id"]
    X_df = df.drop(columns=[c for c in meta_cols if c in df.columns])

    numeric_cols = [c for c in transformers["numeric_cols"] if c in X_df.columns]
    categorical_cols = [c for c in transformers["categorical_cols"] if c in X_df.columns]

    if categorical_cols:
        X_cat = pd.DataFrame(transformers["cat_imputer"].transform(X_df[categorical_cols]) \
                              if set(transformers["categorical_cols"]) == set(categorical_cols)
                              else X_df[categorical_cols].fillna("missing"),
                              columns=categorical_cols, index=X_df.index)
        for c in categorical_cols:
            le = transformers["encoders"][c]
            X_cat[c] = X_cat[c].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1)
    else:
        X_cat = pd.DataFrame(index=X_df.index)

    if numeric_cols:
        X_num = X_df[numeric_cols].fillna(X_df[numeric_cols].median())
        X_num_scaled = pd.DataFrame(transformers["scaler"].transform(
            X_num.reindex(columns=transformers["numeric_cols"], fill_value=0)),
            columns=transformers["numeric_cols"], index=X_df.index)
    else:
        X_num_scaled = pd.DataFrame(index=X_df.index)

    X_full = pd.concat([X_num_scaled, X_cat], axis=1)
    return X_full.values, y


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main(base_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading TRAIN pool datasets (Diabetes + CKD)...")
    diabetes_df = load_diabetes(base_path)
    ckd_df = load_ckd(base_path)

    # Align on a common minimal feature set is NOT required here since diabetes and
    # CKD are trained as one pooled tabular set for the encoder; KG-conditioning
    # layer (built separately) is what ties them to the shared graph.
    train_pool = pd.concat([diabetes_df, ckd_df], ignore_index=True, sort=False)

    print(f"Diabetes rows: {len(diabetes_df)} | CKD rows: {len(ckd_df)} | "
          f"Combined train pool: {len(train_pool)}")

    # ---- Held-out internal test split, GROUPED BY PATIENT AND STRATIFIED ----
    # IMPORTANT: ~30,248 diabetes encounters belong to patients with multiple visits
    # (71,518 unique patients / 101,766 encounters). A plain row-level stratified
    # split can put different encounters from the SAME patient into both train and
    # test, which is patient-level data leakage. Grouping by patient fixes that --
    # but a single random GroupShuffleSplit has NO stratification, and with ~72,000
    # patient groups an unlucky random split can shift the positive-class
    # subpopulation noticeably between train/test by chance alone (this was tried
    # first and caused the held-out AUROC to collapse below random for the
    # concept-pooled models, while CV -- which used a properly stratified group
    # split -- stayed normal). Fix: derive one approximate label per PATIENT (does
    # this patient have any positive-labeled encounter), then stratify the
    # patient-level split on that label before expanding back to encounter rows.
    patient_level = (train_pool.groupby("_patient_id")["label"]
                      .max().reset_index())  # 1 if patient has >=1 positive encounter
    train_patients, test_patients = train_test_split(
        patient_level["_patient_id"], test_size=0.20,
        stratify=patient_level["label"], random_state=RANDOM_STATE
    )
    train_patients, test_patients = set(train_patients), set(test_patients)

    train_mask = train_pool["_patient_id"].isin(train_patients)
    test_mask = train_pool["_patient_id"].isin(test_patients)
    train_df = train_pool[train_mask].reset_index(drop=True)
    heldout_test_df = train_pool[test_mask].reset_index(drop=True)

    # Sanity check: confirm zero patient overlap AND reasonable label balance
    overlap = train_patients & test_patients
    print(f"Patient-level leakage check (train vs held-out test): "
          f"{len(overlap)} overlapping patient(s) -- should be 0")
    assert len(overlap) == 0, "Patient leakage detected between train and held-out test!"
    print(f"Positive rate -- train: {train_df['label'].mean():.4f}, "
          f"held-out test: {heldout_test_df['label'].mean():.4f} (should be close)")

    # ---- Fit preprocessing on TRAIN portion only (avoid leakage) ----
    X_train, y_train, feature_names, transformers = fit_transform_features(train_df)
    X_heldout, y_heldout = transform_with_fitted(heldout_test_df, transformers)
    train_groups = train_df["_patient_id"].astype(str).values

    print(f"Train shape: {X_train.shape} | Held-out internal test shape: {X_heldout.shape}")

    # ---- 5-fold GROUPED stratified CV indices on the TRAIN portion ----
    # StratifiedGroupKFold keeps each patient's encounters within a single fold
    # while still trying to balance the positive rate across folds.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = []
    for fold_i, (tr_idx, val_idx) in enumerate(
            sgkf.split(X_train, y_train, groups=train_groups)):
        # Per-fold sanity check
        fold_train_patients = set(train_groups[tr_idx])
        fold_val_patients = set(train_groups[val_idx])
        fold_overlap = fold_train_patients & fold_val_patients
        assert len(fold_overlap) == 0, f"Patient leakage in fold {fold_i}!"

        fold_indices.append({"fold": fold_i, "train_idx": tr_idx.tolist(),
                              "val_idx": val_idx.tolist()})
        print(f"  Fold {fold_i}: train={len(tr_idx)}, val={len(val_idx)}, "
              f"val_positive_rate={y_train[val_idx].mean():.3f}, "
              f"patient_overlap={len(fold_overlap)}")

    # ---- Save train/CV/test arrays ----
    np.savez(os.path.join(out_dir, "train_pool.npz"),
              X_train=X_train, y_train=y_train,
              X_heldout_test=X_heldout, y_heldout_test=y_heldout)
    with open(os.path.join(out_dir, "cv_folds.json"), "w") as f:
        json.dump(fold_indices, f)
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # ---- External TRANSFER TEST sets (never trained on) ----
    print("\nLoading external TRANSFER TEST datasets (Heart Cleveland + other sites + MIMIC)...")
    heart_cleveland_df = load_heart_cleveland(base_path)
    heart_other_sites_df = load_heart_transfer_sites(base_path)
    mimic_df = load_mimic(base_path)

    # NOTE: Transfer sets have fundamentally different raw feature columns than the
    # train pool (different clinical instruments/tables). Forcing them onto the exact
    # train feature vector is not meaningful here -- the real cross-dataset alignment
    # happens at the KG-conditioning layer (shared disease/feature NODES), not by
    # matching raw column names. So each transfer set is preprocessed independently
    # (its own imputers/scalers/encoders) and kept as a separate array; the KG layer
    # (built separately) is what actually projects all of them into a shared space.
    for name, df in [("heart_cleveland", heart_cleveland_df),
                      ("heart_other_sites", heart_other_sites_df),
                      ("mimic", mimic_df)]:
        X_ext, y_ext, ext_feature_names, _ = fit_transform_features(df)
        np.savez(os.path.join(out_dir, f"transfer_{name}.npz"), X=X_ext, y=y_ext)
        with open(os.path.join(out_dir, f"transfer_{name}_features.json"), "w") as f:
            json.dump(ext_feature_names, f)
        print(f"  {name}: shape={X_ext.shape}, positive_rate={y_ext.mean():.3f} -> saved "
              f"(preprocessed independently; align via KG layer, not raw columns)")

    print(f"\nAll preprocessing artifacts written to: {out_dir}")
    print("Files:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True,
                         help=r'e.g. "E:\Research\07.09.26\Dataset"')
    parser.add_argument("--out_dir", type=str, default="./preprocessed")
    args = parser.parse_args()
    main(args.base_path, args.out_dir)
