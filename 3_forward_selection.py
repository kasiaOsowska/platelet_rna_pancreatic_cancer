"""Greedy forward selection of the final gene panel from the stable gene pool.

Methodology: reuses the split and label encoding of 2_stable_selection_with_enet.py.
The stable genes are taken from the raw (not covariate-corrected) matrix and
standardized per column. Candidates are the
genes that passed stability selection. At each step the gene whose addition maximizes
the validation-set AUC of a logistic regression fitted on the whole training set is
appended to the panel. Holdout test AUC is reported alongside but never used for
selection. The procedure stops at TOP_K_FINAL genes.
"""

import warnings

from utilz.multi_residual_bootstrap import CovariateResidualizer, build_covariates

warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).stem
Path(OUT_DIR).mkdir(exist_ok=True)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED  = 2137

STABLE_GENES_CSV = "2_stable_selection_with_enet/stability_all_stable_genes.csv"
TOP_K_FINAL      = 12

OUT_CSV_PATH = f"{OUT_DIR}/forward_selection_genes.csv"
OUT_PNG_PATH = f"{OUT_DIR}/forward_selection_auc.png"


def holdout_auc(X_tr, y_tr, X_ho, y_ho, idx, seed=BASE_SEED):
    mdl = LogisticRegression(
        max_iter=20000, class_weight='balanced', random_state=seed,
    ).fit(X_tr[:, idx], y_tr)
    return float(roc_auc_score(y_ho, mdl.predict_proba(X_ho[:, idx])[:, 1]))


def forward_select(X_tr, y_tr, X_va, y_va, candidate_idx, gene_names, k_max,
                   seed=BASE_SEED, X_te=None, y_te=None):
    selected = []
    remaining = list(candidate_idx)
    rows = []
    k_max = min(k_max, len(remaining))

    for step in range(1, k_max + 1):
        best_col, best_auc = None, -np.inf
        for col in remaining:
            trial = selected + [col]
            auc_va = holdout_auc(X_tr, y_tr, X_va, y_va, trial, seed)
            if auc_va > best_auc:
                best_col, best_auc = col, auc_va

        selected.append(best_col)
        remaining.remove(best_col)

        row = {
            'step':      step,
            'gene':      gene_names[best_col],
            'auc_valid': best_auc,
        }
        msg = (f"  +{step:>2}  {gene_names[best_col]:<18}"
               f"  valid AUC = {best_auc:.4f}")
        if X_te is not None and y_te is not None:
            row['auc_test'] = holdout_auc(X_tr, y_tr, X_te, y_te, selected, seed)
            msg += f"  | test AUC = {row['auc_test']:.4f}"
        rows.append(row)
        print(msg)

    return selected, pd.DataFrame(rows)


ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
y_enc = pd.Series(LabelEncoder().fit_transform(ds.y), index=ds.y.index)

X_tr_raw, X_te_raw, X_va_raw, y_train, y_test, y_valid = ds.get_train_test_valid_split(
    ds.X, y_enc, test_size=TEST_SIZE, valid_size=VALID_SIZE,
    random_state=BASE_SEED,
)
print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
print(f"Valid: {len(X_va_raw)}  cancer={int(y_valid.sum())}  ctrl={int((y_valid==0).sum())}")
print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

stable_df = pd.read_csv(STABLE_GENES_CSV)
stable_genes = stable_df['gene'].tolist()
genes = [g for g in stable_genes if g in X_tr_raw.columns]
missing = [g for g in stable_genes if g not in X_tr_raw.columns]
if missing:
    print(f"[!] {len(missing)} stable genes missing from the data (skipped): {missing}")
print(f"[forward] candidate pool: {len(genes)} genes  -> selecting {TOP_K_FINAL}")

X_tr_df = X_tr_raw[genes]
X_va_df = X_va_raw[genes]
X_te_df = X_te_raw[genes]
cov = build_covariates(ds.meta)

scale_pipe = Pipeline([
    ('multi_resid', CovariateResidualizer(
        covariates=cov, labels=y_train,
        n_bootstrap=1000, min_r2=0.05, cv_threshold_pct=30.0,
    )),
    ('scaler', StandardScaler()),
])
X_tr_z = scale_pipe.fit_transform(X_tr_df)
X_va_z = scale_pipe.transform(X_va_df)
X_te_z = scale_pipe.transform(X_te_df)
y_tr_np = y_train.values
y_va_np = y_valid.values
y_te_np = y_test.values

print(f"\n=== forward selection (AUC on validation set) ===")
candidate_idx = list(range(len(genes)))
selected_idx, fs_df = forward_select(
    X_tr_z, y_tr_np, X_va_z, y_va_np, candidate_idx, genes, k_max=TOP_K_FINAL,
    seed=BASE_SEED, X_te=X_te_z, y_te=y_te_np,
)
selected_genes = [genes[i] for i in selected_idx]
print(f"\nSelected {len(selected_genes)} genes (in order of selection):")
print(selected_genes)

fs_df['rank'] = np.arange(1, len(fs_df) + 1)
fs_df.to_csv(OUT_CSV_PATH, index=False)
print(f"\n[OK] order + AUC -> {OUT_CSV_PATH}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(fs_df['step'], fs_df['auc_valid'],
        marker='D', color='tab:orange', label='Validation AUC (selection)')
if 'auc_test' in fs_df.columns:
    ax.plot(fs_df['step'], fs_df['auc_test'],
            marker='s', linestyle='--', color='tab:green',
            label='Holdout test AUC')
ax.set(xlabel='Liczba genow (forward selection)', ylabel='AUC',
       title='Greedy forward selection na stabilnych genach')
ax.set_xticks(fs_df['step'])
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG_PATH, dpi=140)
plt.show()
print(f"[OK] plot -> {OUT_PNG_PATH}")


