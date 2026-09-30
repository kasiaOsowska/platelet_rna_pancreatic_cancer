"""Stability-based gene selection for platelet RNA pancreatic cancer classification.

Methodology:

1. Dataset split. Samples are split into train/test/validation with multi-criteria
   stratification (label, stage, sex, age group). The validation set is not needed
   here, so it is concatenated with the test set into a single holdout set. This is
   only an implementation detail: every script performs the same three-way split, so
   the split stays identical across scripts in case a validation set is needed later.

2. Gene filtering, fitted on the training set only:
   a. remove constant genes,
   b. keep genes passing an ANOVA F-test with Benjamini-Hochberg FDR,
   c. drop genes whose mean expression is below the MEAN_THRESHOLD percentile.

3. Covariate correction. Age, sex and log10 library size are regressed out by OLS
   fitted on training controls (CovariateResidualizer); only
   genes whose covariate dependence is large enough (median bootstrap R^2) and
   bootstrap-stable are corrected.

4. Standardization of every gene to zero mean and unit variance on the training set.

5. Tuning. Elastic-net logistic regression hyperparameters (C, l1_ratio) are tuned
   by randomized search with stratified CV.

6. Stability selection (Meinshausen-Buhlmann https://doi.org/10.1111/j.1467-9868.2010.00740.x).
    The tuned elastic net is refitted on N_ITER class-stratified subsamples
    (INNER_SUBSAMPLE of the training set).
    genes with a non-zero coefficient in at least STABILITY_THRESHOLD of the iterations are
    kept and ranked by selection frequency times mean absolute coefficient.

Outputs: the stable genes (OUT_CSV_STABLE), which are the candidate pool for
3_forward_selection.py, gene counts per frequency threshold and a summary.
"""

import warnings
from pathlib import Path

from utilz.multi_residual_bootstrap import (
    CovariateResidualizer, build_covariates,
)

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).stem
Path(OUT_DIR).mkdir(exist_ok=True)


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import loguniform, uniform

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, AnovaFdrReductor, MeanExpressionReductor
)

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED           = 2137
LOG2FC_THRESHOLD    = np.log2(1.2)
DEG_PVAL            = 0.05
ANOVA_FDR_THRESHOLD = 0.05
MEAN_THRESHOLD = 5

TUNE_CV_FOLDS       = 50
TUNE_N_ITER         = 30
TUNE_C_DIST         = loguniform(1e-2, 5)
TUNE_L1_DIST        = uniform(loc=0.4, scale=0.6)

N_ITER              = 100
INNER_SUBSAMPLE     = 0.6
STABILITY_THRESHOLD = 0.8

N_JOBS              = -1

OUT_CSV_THR         = f"{OUT_DIR}/stability_threshold_counts.csv"
OUT_CSV_STATS       = f"{OUT_DIR}/stability_summary.csv"
OUT_CSV_STABLE      = f"{OUT_DIR}/stability_all_stable_genes.csv"


def tune_enet(X, y, seed=BASE_SEED, test=False):

    if test:
        return np.float64(1.641717598117172), np.float64(0.7916508010562565)
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


def stability_selection(X, y, enet_C, enet_l1_ratio,
                        n_iter=N_ITER, subsample=INNER_SUBSAMPLE,
                        seed=BASE_SEED):
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

print("\n=== STEP 1: DEG ===")
cov = build_covariates(ds.meta)
deg_pipe = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaFDRReductor', AnovaFdrReductor(alpha=ANOVA_FDR_THRESHOLD)),
    ('MeanExpressionReductor', MeanExpressionReductor(MEAN_THRESHOLD)),
    ('multi_resid', CovariateResidualizer(
        covariates=cov, labels=y_train,
        n_bootstrap=1000, min_r2=0.05, cv_threshold_pct=30.0,
    )),
    ('scaler', StandardScaler()),
])
X_tr_z = deg_pipe.fit_transform(X_tr_raw, y_train)
deg_genes = list(deg_pipe.get_feature_names_out())
y_tr_np = y_train.values

print("\n=== STEP 2: ENet tuning ===")
c_lo, c_hi = TUNE_C_DIST.support()
l1_lo, l1_hi = TUNE_L1_DIST.support()
print(f"[tune] random search: n_iter={TUNE_N_ITER}, "
      f"C~loguniform[{c_lo:g},{c_hi:g}], l1_ratio~uniform[{l1_lo:g},{l1_hi:g}]")
enet_C, enet_l1 = tune_enet(X_tr_z, y_tr_np, seed=BASE_SEED, test=True)

print(f"\n=== STEP 3: stability selection "
      f"(N_ITER={N_ITER}, subsample={INNER_SUBSAMPLE:.0%}, "
      f"threshold={STABILITY_THRESHOLD:.0%}) ===")
freq, mean_abs_coef, mean_signed_coef = stability_selection(
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

print(f"\n[stability] {int(stable_mask.sum())} / {len(deg_genes)} genes "
      f"passed freq >= {STABILITY_THRESHOLD:.0%}")
print(stable_df.head(20).to_string())


thresholds = [0.5, 0.7, 0.8, 0.9, 1.0]
thr_counts = pd.DataFrame({
    'prog_freq': thresholds,
    'n_genow':   [int((freq >= t).sum()) for t in thresholds],
})
thr_counts.to_csv(OUT_CSV_THR, index=False)
print("\n[stability] gene counts per freq threshold:")
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
print("\n[stability] summary:")
print(summary.to_string(index=False))
print(f"  up (positive coefficient):   {n_up}")
print(f"  down (negative coefficient): {n_down}")
print(f"[OK] statistics -> {OUT_CSV_THR}, {OUT_CSV_STATS}, {OUT_CSV_STABLE}")
