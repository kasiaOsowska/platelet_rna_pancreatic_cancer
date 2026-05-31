"""
Greedy forward selection na stabilnych genach (wynik bootstrap_enet.py).

Zamiast dodawac geny w sztywnym rankingu (freq * mean_abs_coef, jak KROK 4
w bootstrap_enet.py), na kazdym kroku dokladamy ten gen z puli kandydatow,
ktory NAJBARDZIEJ podnosi AUC na zbiorze treningowym (CV AUC, stratyfikowane
foldy). Powtarzamy az do TOP_K_FINAL genow.

Pula kandydatow = geny z stability_all_stable_genes.csv.
Split, kodowanie etykiet i skalowanie - identyczne jak w bootstrap_enet.py,
wiec X_tr_z dla podzbioru genow jest dokladnie taki sam (StandardScaler
dziala per-kolumna).
"""

import warnings
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

# split (musi byc taki sam jak w bootstrap_enet.py, zeby trafic w ten sam train)
TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED  = 2137

# forward selection
STABLE_GENES_CSV = "3_stable_enet_selection/stability_all_stable_genes.csv"
TOP_K_FINAL      = 12      # ile genow dobrac
SELECT_CV_FOLDS  = 10      # foldy do oceny kandydata (capowane min. liczebnoscia klasy)

OUT_CSV_PATH = f"{OUT_DIR}/forward_selection_genes.csv"
OUT_PNG_PATH = f"{OUT_DIR}/forward_selection_auc.png"


def cv_auc_train(X, y, idx, folds, seed=BASE_SEED):
    """Sredni CV AUC na zbiorze treningowym dla kolumn `idx`.
    `folds` to lista par (train_idx, test_idx) z ds.get_stratified_kfold."""
    aucs = []
    Xi = X[:, idx]
    for tr, va in folds:
        mdl = LogisticRegression(
            max_iter=20000, class_weight='balanced', random_state=seed,
        ).fit(Xi[tr], y[tr])
        aucs.append(roc_auc_score(y[va], mdl.predict_proba(Xi[va])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def test_auc(X_tr, y_tr, X_te, y_te, idx, seed=BASE_SEED):
    """Holdout test AUC: model uczony na calym train, oceniany na test."""
    mdl = LogisticRegression(
        max_iter=20000, class_weight='balanced', random_state=seed,
    ).fit(X_tr[:, idx], y_tr)
    return float(roc_auc_score(y_te, mdl.predict_proba(X_te[:, idx])[:, 1]))


def forward_select(X_tr, y_tr, candidate_idx, gene_names, k_max,
                   folds, seed=BASE_SEED,
                   X_te=None, y_te=None):
    """Chciwy dobor: na kazdym kroku dokladamy gen maks. CV AUC na train."""
    selected = []
    remaining = list(candidate_idx)
    rows = []
    k_max = min(k_max, len(remaining))

    for step in range(1, k_max + 1):
        best_col, best_auc, best_std = None, -np.inf, np.nan
        for col in remaining:
            trial = selected + [col]
            auc_mean, auc_std = cv_auc_train(X_tr, y_tr, trial, folds, seed)
            if auc_mean > best_auc:
                best_col, best_auc, best_std = col, auc_mean, auc_std

        selected.append(best_col)
        remaining.remove(best_col)

        row = {
            'step':     step,
            'gene':     gene_names[best_col],
            'auc_mean': best_auc,
            'auc_std':  best_std,
        }
        msg = (f"  +{step:>2}  {gene_names[best_col]:<18}"
               f"  CV AUC(train) = {best_auc:.4f}+-{best_std:.4f}")
        if X_te is not None and y_te is not None:
            row['auc_test'] = test_auc(X_tr, y_tr, X_te, y_te, selected, seed)
            msg += f"  | test AUC = {row['auc_test']:.4f}"
        rows.append(row)
        print(msg)

    return selected, pd.DataFrame(rows)


def main():
    # --- dane + split (identyczny z bootstrap_enet.py) ---
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})
    y_enc = pd.Series(LabelEncoder().fit_transform(ds.y), index=ds.y.index)

    X_tr_raw, X_te_raw, X_va_raw, y_train, y_test, y_valid = ds.get_train_test_valid_split(
        ds.X, y_enc, test_size=TEST_SIZE, valid_size=VALID_SIZE,
        random_state=BASE_SEED,
    )
    X_te_raw = pd.concat([X_te_raw, X_va_raw])
    y_test   = pd.concat([y_test, y_valid])
    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    # --- pula kandydatow = stabilne geny ---
    stable_df = pd.read_csv(STABLE_GENES_CSV)
    stable_genes = stable_df['gene'].tolist()
    genes = [g for g in stable_genes if g in X_tr_raw.columns]
    missing = [g for g in stable_genes if g not in X_tr_raw.columns]
    if missing:
        print(f"[!] {len(missing)} stabilnych genow nie ma w danych (pomijam): {missing}")
    print(f"[forward] pula kandydatow: {len(genes)} genow  -> dobieram {TOP_K_FINAL}")

    # --- skalowanie (per-kolumna, jak w bootstrap_enet.py) ---
    X_tr_df = X_tr_raw[genes]
    X_te_df = X_te_raw[genes]
    scaler = StandardScaler().fit(X_tr_df.values)
    X_tr_z = scaler.transform(X_tr_df.values)
    X_te_z = scaler.transform(X_te_df.values)
    y_tr_np = y_train.values
    y_te_np = y_test.values

    # --- greedy forward selection ---
    print(f"\n=== forward selection (CV AUC na train, {SELECT_CV_FOLDS}-fold) ===")
    candidate_idx = list(range(len(genes)))
    # foldy z wlasnej, wielokryterialnej stratyfikacji (ds.get_stratified_kfold), liczone raz
    folds = ds.get_stratified_kfold(X_tr_df, y_train, n_splits=SELECT_CV_FOLDS, random_state=BASE_SEED)
    selected_idx, fs_df = forward_select(
        X_tr_z, y_tr_np, candidate_idx, genes, k_max=TOP_K_FINAL,
        folds=folds, seed=BASE_SEED,
        X_te=X_te_z, y_te=y_te_np,
    )
    selected_genes = [genes[i] for i in selected_idx]
    print(f"\nWybrane {len(selected_genes)} genow (kolejnosc doboru):")
    print(selected_genes)

    # --- zapis CSV ---
    fs_df['rank'] = np.arange(1, len(fs_df) + 1)
    fs_df.to_csv(OUT_CSV_PATH, index=False)
    print(f"\n[OK] kolejnosc + AUC -> {OUT_CSV_PATH}")

    # --- wykres ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(fs_df['step'], fs_df['auc_mean'], yerr=fs_df['auc_std'],
                marker='o', capsize=3,
                label=f'CV AUC ({SELECT_CV_FOLDS}-fold, train)')
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
    print(f"[OK] wykres            -> {OUT_PNG_PATH}")


if __name__ == '__main__':
    main()
