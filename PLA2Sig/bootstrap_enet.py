"""
3. Stability-based gene selection (Elastic Net + bootstrap, PLA2Sig-style).

Schemat (wzorowany na 4_pla2sig_gene_selection.py):
  KROK 1: DEG (log2FC + Mann-Whitney + multi-covariate residual bootstrap)
  KROK 2: Tuning hiperparametrow ENet (C, l1_ratio) - GridSearchCV na train
  KROK 3: Bootstrap stability - N_ITER iteracji ENet (tuned params) na
          losowych podprobkach train; gen przechodzi jesli wybrany w
          >= STABILITY_THRESHOLD iteracji
  KROK 4: Inkrementalne top-k CV AUC + holdout test AUC -> wybor k*
  KROK 5: Final GLM (binomial) na k* genach + zapis do CSV
"""

import warnings

from utilz.multi_residual_bootstrap import (
    MultiCovariateResidualBootstrapTransformer, build_covariates,
)

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import loguniform, uniform

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, Log2FCReductor, MannWhitneyReductor, AnovaFdrReductor, WithinGroupVarianceReductor,
    MeanExpressionReductor,
)

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

# split / DEG
TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED           = 2137
LOG2FC_THRESHOLD    = np.log2(1.2)
DEG_PVAL            = 0.05
ANOVA_FDR_THRESHOLD = 0.05
MEAN_THRESHOLD = 5

# tuning ENet (random search)
TUNE_CV_FOLDS       = 50
TUNE_N_ITER         = 30
TUNE_C_DIST         = loguniform(1e-2, 5)
TUNE_L1_DIST        = uniform(loc=0.4, scale=0.6)

# bootstrap stability
N_ITER              = 100
INNER_SUBSAMPLE     = 0.6       # frakcja train uzyta w kazdej iteracji
STABILITY_THRESHOLD = 0.8       # gen niezerowy w >= 80% iteracji

# incremental top-k
TOP_K_FINAL         = 12   # ile genów zapisac (sztywne, bez plateau)
INCR_CV_FOLDS       = 50

N_JOBS              = -1

OUT_CSV_GENES       = "bootstrap_enet_selected_genes.csv"
OUT_PNG_INCR        = "stability_enet_incremental.png"
OUT_CSV_THR         = "stability_threshold_counts.csv"
OUT_CSV_STATS       = "stability_summary.csv"
OUT_CSV_STABLE      = "stability_all_stable_genes.csv"


# ---------------------------------------------------------------------------
# KROK 2: tuning
# ---------------------------------------------------------------------------
def tune_enet(X, y, seed=BASE_SEED):
    n_folds = min(TUNE_CV_FOLDS, int(np.bincount(y.astype(int)).min()))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    estimator = LogisticRegression(
        penalty='elasticnet', solver='saga',
        max_iter=20000, class_weight='balanced',
        random_state=seed,
    )
    rs = RandomizedSearchCV(
        estimator,
        param_distributions={'C': TUNE_C_DIST, 'l1_ratio': TUNE_L1_DIST},
        n_iter=TUNE_N_ITER,
        scoring='roc_auc', cv=cv, n_jobs=N_JOBS, refit=False,
        return_train_score=False, verbose=0,
        random_state=seed,
    ).fit(X, y)
    print(f"[tune] {n_folds}-fold CV; best CV AUC={rs.best_score_:.4f}  "
          f"params={rs.best_params_}")
    return rs.best_params_['C'], rs.best_params_['l1_ratio']


# ---------------------------------------------------------------------------
# KROK 3: bootstrap stability
# ---------------------------------------------------------------------------
def bootstrap_stability(X, y, enet_C, enet_l1_ratio,
                        n_iter=N_ITER, subsample=INNER_SUBSAMPLE,
                        seed=BASE_SEED):
    """Subsample-based stability selection (Meinshausen-Buhlmann 2010).
    Stratyfikowane podprobki bez zwracania.
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    selected_count = np.zeros(p, dtype=int)
    abs_coef_sums  = np.zeros(p, dtype=float)
    signed_coef_sums = np.zeros(p, dtype=float)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos_sub = max(2, int(len(idx_pos) * subsample))
    n_neg_sub = max(2, int(len(idx_neg) * subsample))

    progress_every = max(1, n_iter // 10)
    for i in range(n_iter):
        sub_pos = rng.choice(idx_pos, size=n_pos_sub, replace=False)
        sub_neg = rng.choice(idx_neg, size=n_neg_sub, replace=False)
        sub_idx = np.concatenate([sub_pos, sub_neg])

        enet = LogisticRegression(
            penalty='elasticnet', solver='saga',
            l1_ratio=enet_l1_ratio, C=enet_C,
            class_weight='balanced', max_iter=20000,
            random_state=seed + i + 1,
        ).fit(X[sub_idx], y[sub_idx])

        coefs = enet.coef_[0]
        nz = coefs != 0
        selected_count += nz.astype(int)
        abs_coef_sums  += np.abs(coefs)
        signed_coef_sums += coefs
        if (i + 1) % progress_every == 0 or i + 1 == n_iter:
            print(f"  [boot {i+1:>4}/{n_iter}]  n_selected={int(nz.sum())}")

    freq = selected_count / n_iter
    mean_abs_coef = np.where(
        selected_count > 0,
        abs_coef_sums / np.maximum(selected_count, 1),
        0.0,
    )
    mean_signed_coef = np.where(
        selected_count > 0,
        signed_coef_sums / np.maximum(selected_count, 1),
        0.0,
    )
    return freq, mean_abs_coef, mean_signed_coef


# ---------------------------------------------------------------------------
# KROK 4: incremental top-k (taki sam jak w 4_pla2sig)
# ---------------------------------------------------------------------------
def incremental_topk_eval(X, y, ranked_idx, k_max=TOP_K_FINAL,
                          n_folds=INCR_CV_FOLDS, seed=BASE_SEED,
                          X_test=None, y_test=None):
    n_folds = min(n_folds, int(np.bincount(y.astype(int)).min()))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for k in range(1, min(k_max, len(ranked_idx)) + 1):
        idx = ranked_idx[:k]
        aucs = []
        for tr, va in skf.split(X, y):
            mdl = LogisticRegression(
                max_iter=20000, class_weight='balanced', random_state=seed,
            ).fit(X[np.ix_(tr, idx)], y[tr])
            aucs.append(roc_auc_score(
                y[va], mdl.predict_proba(X[np.ix_(va, idx)])[:, 1]))
        row = {'k': k, 'auc_mean': float(np.mean(aucs)),
               'auc_std': float(np.std(aucs, ddof=1))}
        if X_test is not None and y_test is not None:
            mdl_full = LogisticRegression(
                max_iter=20000, class_weight='balanced', random_state=seed,
            ).fit(X[:, idx], y)
            row['auc_test'] = float(roc_auc_score(
                y_test, mdl_full.predict_proba(X_test[:, idx])[:, 1]))
            msg = (f"  top-{k:>2}  CV AUC = {row['auc_mean']:.4f}+-{row['auc_std']:.4f}"
                   f"  | test AUC = {row['auc_test']:.4f}")
        else:
            msg = f"  top-{k:>2}  CV AUC = {row['auc_mean']:.4f}+-{row['auc_std']:.4f}"
        rows.append(row)
        print(msg)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})
    y_enc = pd.Series(LabelEncoder().fit_transform(ds.y), index=ds.y.index)

    X_tr_raw, X_te_raw, X_va_raw, y_train, y_test, y_valid = ds.get_train_test_valid_split(
        ds.X, y_enc, test_size=TEST_SIZE, valid_size=VALID_SIZE,
        random_state=BASE_SEED,
    )
    # walidacyjny niepotrzebny -> doklejamy do test setu (split deterministyczny)
    X_te_raw = pd.concat([X_te_raw, X_va_raw])
    y_test   = pd.concat([y_test, y_valid])
    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    # --- KROK 1: DEG (identyczny jak w 4_pla2sig) ---
    print("\n=== KROK 1: DEG ===")
    cov = build_covariates(ds.meta)
    deg_pipe = Pipeline([
        ('ConstantExpressionReductor', ConstantExpressionReductor()),

        ('AnovaFDRReductor', AnovaFdrReductor(alpha=ANOVA_FDR_THRESHOLD)),
        #('Log2FCReductor', Log2FCReductor(min_abs_log2fc=LOG2FC_THRESHOLD)),
        ('MeanExpressionReductor', MeanExpressionReductor(MEAN_THRESHOLD)),
        ('multi_resid', MultiCovariateResidualBootstrapTransformer(
            covariates=cov, labels=y_train,
            n_bootstrap=1000, fdr_alpha=0.05, min_r2=0.05, cv_threshold_pct=30.0,
        )),
    ])
    X_tr_deg_df = deg_pipe.fit_transform(X_tr_raw, y_train)
    X_te_deg_df = deg_pipe.transform(X_te_raw)
    deg_genes = list(X_tr_deg_df.columns)

    scaler = StandardScaler().fit(X_tr_deg_df.values)
    X_tr_z = scaler.transform(X_tr_deg_df.values)
    X_te_z = scaler.transform(X_te_deg_df.values)
    y_tr_np = y_train.values
    y_te_np = y_test.values

    # --- KROK 2: tuning hiperparametrow ENet ---
    print("\n=== KROK 2: tuning ENet ===")
    print(f"[tune] random search: n_iter={TUNE_N_ITER}, "
          f"C~loguniform[0.01,0.2], l1_ratio~uniform[0.4,0.9]")
    enet_C, enet_l1 = tune_enet(X_tr_z, y_tr_np, seed=BASE_SEED)

    # --- KROK 3: bootstrap stability ---
    print(f"\n=== KROK 3: bootstrap stability "
          f"(N_ITER={N_ITER}, subsample={INNER_SUBSAMPLE:.0%}, "
          f"threshold={STABILITY_THRESHOLD:.0%}) ===")
    freq, mean_abs_coef, mean_signed_coef = bootstrap_stability(
        X_tr_z, y_tr_np, enet_C, enet_l1,
        n_iter=N_ITER, subsample=INNER_SUBSAMPLE, seed=BASE_SEED,
    )

    stable_mask = freq >= STABILITY_THRESHOLD
    stable_df = (pd.DataFrame({
        'gene':             [deg_genes[i] for i in np.where(stable_mask)[0]],
        'freq':             freq[stable_mask],
        'mean_abs_coef':    mean_abs_coef[stable_mask],
        'mean_signed_coef': mean_signed_coef[stable_mask],
        'score':            freq[stable_mask] * mean_abs_coef[stable_mask],
    }).sort_values('score', ascending=False).reset_index(drop=True))

    print(f"\n[stability] {int(stable_mask.sum())} / {len(deg_genes)} genow "
          f"przeszlo prog freq >= {STABILITY_THRESHOLD:.0%}")
    print(stable_df.head(20).to_string())

    if stable_mask.sum() == 0:
        print("\n[!] zaden gen nie przeszedl progu stability -- konczy.")
        return

    # --- statystyki stabilnosci (zapis do CSV) ---
    thresholds = [0.5, 0.7, 0.8, 0.9, 1.0]
    thr_counts = pd.DataFrame({
        'prog_freq': thresholds,
        'n_genow':   [int((freq >= t).sum()) for t in thresholds],
    })
    thr_counts.to_csv(OUT_CSV_THR, index=False)
    print("\n[stability] licznosc genow przy progach freq:")
    print(thr_counts.to_string(index=False))

    n_up   = int((stable_df['mean_signed_coef'] > 0).sum())
    n_down = int((stable_df['mean_signed_coef'] < 0).sum())
    summary = pd.DataFrame([{
        'n_stable':         int(stable_mask.sum()),
        'n_input':          len(deg_genes),
        'frac_stable':      round(float(stable_mask.sum()) / len(deg_genes), 4),
        'freq_median':      round(float(np.median(freq[stable_mask])), 4),
        'freq_min':         round(float(freq[stable_mask].min()), 4),
        'freq_max':         round(float(freq[stable_mask].max()), 4),
        'n_up_regulated':   n_up,
        'n_down_regulated': n_down,
    }])
    summary.to_csv(OUT_CSV_STATS, index=False)
    stable_df.to_csv(OUT_CSV_STABLE, index=False)
    print("\n[stability] podsumowanie:")
    print(summary.to_string(index=False))
    print(f"  up (dodatni wspolczynnik):  {n_up}")
    print(f"  down (ujemny wspolczynnik): {n_down}")
    print(f"[OK] statystyki -> {OUT_CSV_THR}, {OUT_CSV_STATS}, {OUT_CSV_STABLE}")

    # --- KROK 4: incremental top-k ---
    print("\n=== KROK 4: top-k ===")
    g2c = {g: i for i, g in enumerate(deg_genes)}
    ranked_idx = [g2c[g] for g in stable_df['gene'].tolist()]
    n_available = min(TOP_K_FINAL, len(ranked_idx))
    incr_df = incremental_topk_eval(
        X_tr_z, y_tr_np, ranked_idx,
        k_max=n_available,
        X_test=X_te_z, y_test=y_te_np,
    )
    minimal_genes = stable_df['gene'].head(n_available).tolist()
    print(f"\nzapisuje top-{n_available} genow: {minimal_genes}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(incr_df['k'], incr_df['auc_mean'], yerr=incr_df['auc_std'],
                marker='o', capsize=3,
                label=f'CV AUC ({INCR_CV_FOLDS}-fold, train)')
    if 'auc_test' in incr_df.columns:
        ax.plot(incr_df['k'], incr_df['auc_test'],
                marker='s', linestyle='--', color='tab:green',
                label='Holdout test AUC')
    ax.set(xlabel='Top-k genes', ylabel='AUC',
           title=f'Stability-ENet (C={enet_C}, l1={enet_l1}): top-k AUC')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG_INCR, dpi=140)
    plt.show()

    # --- KROK 5: final GLM + zapis ---
    print("\n=== KROK 5: GLM ===")
    idx_min = [g2c[g] for g in minimal_genes]
    glm = LogisticRegression(
        penalty=None, solver='lbfgs', max_iter=20000,
        class_weight='balanced', random_state=BASE_SEED,
    ).fit(X_tr_z[:, idx_min], y_tr_np)

    auc_tr = roc_auc_score(y_tr_np, glm.predict_proba(X_tr_z[:, idx_min])[:, 1])
    auc_te = roc_auc_score(y_te_np, glm.predict_proba(X_te_z[:, idx_min])[:, 1])
    print(f"coef:       {dict(zip(minimal_genes, glm.coef_.ravel().round(4)))}")
    print(f"intercept:  {glm.intercept_[0]:.4f}")
    print(f"Train AUC:  {auc_tr:.4f}")
    print(f"Holdout AUC:{auc_te:.4f}")

    stable_idx = stable_df.set_index('gene')
    out_df = pd.DataFrame({
        'gene':              minimal_genes,
        'enet_freq':         [stable_idx.loc[g, 'freq'] for g in minimal_genes],
        'enet_mean_abs_coef':[stable_idx.loc[g, 'mean_abs_coef'] for g in minimal_genes],
        'glm_coef':          glm.coef_.ravel(),
        'rank':              np.arange(1, len(minimal_genes) + 1),
    })
    out_df.loc[len(out_df)] = {
        'gene': '__intercept__',
        'enet_freq': np.nan,
        'enet_mean_abs_coef': np.nan,
        'glm_coef': float(glm.intercept_[0]),
        'rank': 0,
    }
    out_df.to_csv(OUT_CSV_GENES, index=False)
    print(f"\n[OK] zapisano {len(minimal_genes)} genow + intercept -> {OUT_CSV_GENES}")
    print(f"[OK] wykres incremental                                 -> {OUT_PNG_INCR}")


if __name__ == '__main__':
    main()
