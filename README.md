# KG-TransNet: A Knowledge-Guided Concept Attention Framework for Explainable and Transferable Clinical Risk Prediction Across Heterogeneous Healthcare Datasets

This repository contains the full implementation, preprocessing pipeline, and experimental scripts for the study *"A Knowledge-Guided Concept Attention Framework for Explainable and Transferable Clinical Risk Prediction Across Heterogeneous Healthcare Datasets,"* submitted to *Knowledge-Based Systems*.

## Overview

KG-TransNet projects heterogeneous electronic health record (EHR) features from four independent public datasets into a shared, clinically grounded concept space, refines this representation using a knowledge-guided graph-attention mechanism, and distills predictions into human-readable rules via a surrogate decision-tree module. The framework is trained on Diabetes-130 and Chronic Kidney Disease data and evaluated zero-shot — without retraining — on Heart Disease and MIMIC-III, testing whether the learned knowledge structure generalizes across diseases and institutions.

## Repository Structure

```
├── preprocess_datasets.py   # Data loading, cleaning, and leakage-free patient-stratified splitting
├── model.py                  # KG-TransNet architecture: concept pooling, graph attention, classifier
├── train_model.py            # 5-fold CV training, held-out and transfer evaluation, rule extraction
├── train_baselines.py        # Logistic Regression, Random Forest, Plain MLP, Raw MLP baselines
├── ensemble_check.py         # KG-TransNet + Random Forest ensemble evaluation
├── generate_figures.py       # Generates all manuscript figures (ROC/PR curves, bar charts, etc.)
├── export_tables.py          # Exports all results to CSV tables
└── README.md
```

## Datasets

All datasets are publicly available and are **not included** in this repository (see download links below). The preprocessing script expects them organized under a single base directory.

| Dataset | Source |
|---|---|
| Diabetes 130-US Hospitals | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| Chronic Kidney Disease | https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease |
| Heart Disease | https://archive.ics.uci.edu/dataset/45/heart+disease |
| MIMIC-III Clinical Database Demo | https://physionet.org/content/mimiciii-demo/1.4/ |

Expected folder layout:
```
Dataset/
├── diabetes+130-us+hospitals+for+years+1999-2008/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── Chronic_Kidney_Disease/
│   └── chronic_kidney_disease_full.arff
├── heart+disease/
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   └── processed.va.data
└── mimic-iii-clinical-database-demo-1.4/
    ├── ADMISSIONS.csv
    └── DIAGNOSES_ICD.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline in order:

```bash
# 1. Preprocess all datasets with leakage-free, patient-stratified splitting
python preprocess_datasets.py --base_path "./Dataset" --out_dir "./preprocessed"

# 2. Train KG-TransNet: 5-fold CV, held-out test, zero-shot transfer, rule extraction
python train_model.py --data_dir "./preprocessed" --out_dir "./model_output"

# 3. Train baseline models for comparison
python train_baselines.py --data_dir "./preprocessed" --out_dir "./model_output"

# 4. Evaluate the KG-TransNet + Random Forest ensemble
python ensemble_check.py --data_dir "./preprocessed" --model_dir "./model_output"

# 5. Generate all manuscript figures
python generate_figures.py --data_dir "./preprocessed" --model_dir "./model_output" --out_dir "./figures"

# 6. Export all results as CSV tables
python export_tables.py --model_dir "./model_output" --out_dir "./tables"
```

## Key Methodological Notes

- **Patient-level data leakage prevention**: The Diabetes-130 dataset contains multiple encounters per patient. All train/test and cross-validation splits are performed at the patient level (via `StratifiedGroupKFold` and a patient-level stratified split) to prevent the same patient's records from appearing in both training and evaluation partitions.
- **Zero-shot transfer evaluation**: Heart Disease and MIMIC-III are never used during training or hyperparameter selection; they are evaluated only after the final model is trained on Diabetes-130 + CKD.
- **Concept pooling**: Raw features from each dataset (which differ in dimensionality) are deterministically mapped into a shared 6-concept space via keyword matching, enabling the same trained model to be applied across datasets with entirely different raw feature schemas.

## Results Summary

| Model | CV AUROC | Held-out AUROC | Cleveland Transfer | Other-sites Transfer | MIMIC Transfer |
|---|---|---|---|---|---|
| KG-TransNet | 0.609 | 0.594 | **0.767** | 0.594 | 0.468 |
| Logistic Regression | 0.608 | 0.597 | 0.342 | 0.639 | 0.508 |
| Random Forest | 0.617 | 0.523 | 0.630 | 0.712 | 0.589 |
| Plain MLP (concept) | 0.610 | 0.593 | 0.614 | 0.720 | 0.474 |

## Citation

If you use this code, please cite:

```
[Insert full citation once the paper is published/accepted]
```

## License

This repository is released under the MIT License (see `LICENSE`).

## Contact

For questions, please open an issue on this repository or contact [your email].
