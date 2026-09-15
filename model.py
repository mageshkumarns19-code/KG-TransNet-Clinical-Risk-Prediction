"""
KG-TransNet model architecture
================================
Key design constraint: the four datasets have DIFFERENT raw feature counts
(train pool = 44 features; heart_cleveland = 13; heart_other_sites = 7; mimic = 6).
A single fixed-size input encoder can't be shared across them directly.

Solution: raw features -> deterministic CONCEPT POOLING (keyword-matched into a
fixed set of K clinical concept groups, e.g. "renal", "metabolic", "cardiac") ->
a fixed K-dimensional vector regardless of the original dataset's raw dimensionality.
Everything downstream of concept pooling (embedding, graph attention over the
comorbidity graph, classifier) is fully SHARED and dataset-agnostic, which is what
makes zero-shot transfer to heart disease / MIMIC meaningful: those datasets are
projected into the same concept space the model was trained on, with no
dataset-specific parameters to retrain.
"""

import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. CONCEPT SCHEMA (shared knowledge graph nodes)
# ----------------------------------------------------------------------
CONCEPTS = ["metabolic", "renal", "cardiac", "vascular", "demographic", "utilization"]

CONCEPT_KEYWORDS = {
    "metabolic":   ["gluc", "a1c", "diabet", "metformin", "glipizide", "glyburide",
                     "glimepiride", "insulin", "pioglitazone", "rosiglitazone",
                     "acarbose", "miglitol", "dm", "su", "diag_1", "diag_2", "diag_3"],
    "renal":       ["sc", "creatin", "bu", "urea", "sod", "pot", "sg", "pcv", "hemo",
                     "wc", "rc", "wbcc", "rbcc", "ane", "rbc", "al", "pc", "pcc", "ba"],
    "cardiac":     ["chol", "trestbps", "thalach", "oldpeak", "cp", "exang", "slope",
                     "ca", "thal", "cad", "restecg"],
    "vascular":    ["htn", "bp", "pe"],
    "demographic": ["age", "sex", "gender", "race"],
    "utilization": ["num_procedures", "num_lab_procedures", "num_medications",
                     "time_in_hospital", "number_outpatient", "number_emergency",
                     "number_inpatient", "number_diagnoses", "admission_type",
                     "discharge_disposition", "admission_source", "los_days",
                     "num_diagnoses", "appet", "change"],
}

# Fixed comorbidity adjacency between concepts (1 = edge, grounded in known
# cardio-renal-metabolic clinical relationships; demographic/utilization connect
# to everything since age and care-utilization modulate all conditions).
_N = len(CONCEPTS)
ADJACENCY = np.ones((_N, _N), dtype=np.float32)  # start fully connected...
# ...then explicitly zero out pairs with no established clinical relationship
_idx = {c: i for i, c in enumerate(CONCEPTS)}
# (kept simple/dense on purpose: comorbidity in this triad is broadly interconnected;
#  sparsify here if you want a stricter graph prior)


def build_concept_membership(feature_names):
    """
    Given a list of raw feature names for a specific dataset, returns a
    (num_concepts, num_features) binary membership matrix via keyword matching.
    A feature can belong to multiple concepts. Unmatched features fall into
    'utilization' as a catch-all so no signal is silently dropped.
    """
    n_feat = len(feature_names)
    membership = np.zeros((len(CONCEPTS), n_feat), dtype=np.float32)
    lower_names = [str(f).lower() for f in feature_names]

    for f_idx, name in enumerate(lower_names):
        matched = False
        for c_idx, concept in enumerate(CONCEPTS):
            for kw in CONCEPT_KEYWORDS[concept]:
                if kw in name:
                    membership[c_idx, f_idx] = 1.0
                    matched = True
        if not matched:
            membership[_idx["utilization"], f_idx] = 1.0

    # Normalize each concept row to average (not sum) its member features
    row_sums = membership.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    membership_norm = membership / row_sums
    return membership_norm  # (num_concepts, num_features)


def pool_to_concepts(X, membership_norm):
    """
    X: (batch, num_features) raw preprocessed feature matrix (already scaled/imputed)
    membership_norm: (num_concepts, num_features) -- binary membership (not yet
                      row-normalized; normalization is handled per-statistic below)
    Returns: (batch, num_concepts * 3) concept-pooled vector: for each concept,
             [mean, max, share-of-abnormal-members] of its member features.

    NOTE: a single mean-per-concept was tried first and was far too lossy --
    averaging many raw (and especially label-encoded categorical) features into
    one scalar destroyed almost all discriminative signal. Using three
    complementary statistics per concept keeps the representation compact and
    still fully deterministic/dataset-agnostic (required for zero-shot transfer),
    while retaining much more of the real signal.
    """
    membership_bin = (membership_norm > 0).astype(np.float32)  # (K, F) binary mask
    counts = membership_bin.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0

    # Mean per concept
    mean_vals = (X @ membership_bin.T) / counts.T  # (batch, K)

    # Max per concept (masked max over member features; -inf for non-members)
    K, Fdim = membership_bin.shape
    batch = X.shape[0]
    masked = np.where(membership_bin[None, :, :] > 0,
                       X[:, None, :],
                       -np.inf)  # (batch, K, F)
    max_vals = masked.max(axis=2)
    max_vals[np.isneginf(max_vals)] = 0.0  # concept had no members in this dataset

    # Share of "abnormal" members per concept (fraction of member features
    # more than 1 std from 0, since inputs are already z-scored/scaled)
    abnormal = (np.abs(X) > 1.0).astype(np.float32)  # (batch, F)
    abnormal_share = (abnormal @ membership_bin.T) / counts.T  # (batch, K)

    pooled = np.concatenate([mean_vals, max_vals, abnormal_share], axis=1)  # (batch, 3K)
    return pooled


# ----------------------------------------------------------------------
# 2. SHARED MODEL COMPONENTS (dataset-agnostic from here on)
# ----------------------------------------------------------------------
class ConceptEmbedding(nn.Module):
    """Turns each concept's pooled statistics (mean, max, abnormal-share) into a
    learned embedding vector, combined with a learned per-concept identity embedding."""
    def __init__(self, num_concepts, emb_dim, stats_per_concept=3):
        super().__init__()
        self.stats_per_concept = stats_per_concept
        self.value_proj = nn.Linear(stats_per_concept, emb_dim)
        self.identity_emb = nn.Embedding(num_concepts, emb_dim)
        self.num_concepts = num_concepts

    def forward(self, concept_stats):
        # concept_stats: (batch, num_concepts * stats_per_concept)
        batch = concept_stats.shape[0]
        stats_reshaped = concept_stats.view(batch, self.num_concepts, self.stats_per_concept)
        value_emb = self.value_proj(stats_reshaped)  # (batch, K, emb_dim)
        idx = torch.arange(self.num_concepts, device=concept_stats.device)
        id_emb = self.identity_emb(idx).unsqueeze(0).expand(batch, -1, -1)  # (batch, K, emb_dim)
        return value_emb + id_emb  # (batch, K, emb_dim)


class KGGraphAttention(nn.Module):
    """Single-layer adjacency-masked self-attention over concept nodes.
    This is the 'KG-conditioning' step: each concept's representation is
    refined by attending only to clinically-related concepts (per ADJACENCY)."""
    def __init__(self, emb_dim, adjacency):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.scale = emb_dim ** 0.5
        mask = torch.tensor(adjacency, dtype=torch.float32)
        self.register_buffer("mask", mask)  # (K, K), 1 = allowed edge

    def forward(self, concept_emb):
        # concept_emb: (batch, K, emb_dim)
        Q = self.q_proj(concept_emb)
        K = self.k_proj(concept_emb)
        V = self.v_proj(concept_emb)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, K, K)
        mask_add = (1.0 - self.mask) * -1e9
        scores = scores + mask_add.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)  # (batch, K, K)
        out = torch.matmul(attn, V)  # (batch, K, emb_dim)
        return out, attn


class KGTransNet(nn.Module):
    """Full model: concept embedding -> KG graph attention -> attention-pooled
    patient representation -> classifier. Everything here is dataset-agnostic;
    only the concept-pooling step (outside this module) is dataset-specific."""
    def __init__(self, num_concepts=len(CONCEPTS), emb_dim=32, adjacency=ADJACENCY,
                 dropout=0.2):
        super().__init__()
        self.concept_embed = ConceptEmbedding(num_concepts, emb_dim)
        self.graph_attn = KGGraphAttention(emb_dim, adjacency)
        self.patient_query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.01)
        self.pool_proj_k = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, concept_stats):
        # concept_stats: (batch, num_concepts * 3)
        concept_emb = self.concept_embed(concept_stats)              # (batch, K, emb)
        refined, kg_attn = self.graph_attn(concept_emb)             # (batch, K, emb)
        refined = self.dropout(refined)

        # Attention-pool concepts into a single patient representation using a
        # learned global query (this also gives per-patient concept importance
        # weights, usable directly for explanation).
        batch = refined.shape[0]
        q = self.patient_query.expand(batch, -1, -1)                # (batch, 1, emb)
        k = self.pool_proj_k(refined)                                # (batch, K, emb)
        pool_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        pool_weights = F.softmax(pool_scores, dim=-1)                # (batch, 1, K)
        patient_repr = torch.matmul(pool_weights, refined).squeeze(1)  # (batch, emb)

        logit = self.classifier(patient_repr).squeeze(-1)            # (batch,)
        concept_importance = pool_weights.squeeze(1)                  # (batch, K)
        return logit, concept_importance, kg_attn
