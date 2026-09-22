"""Replication of the PLA2Sig gene selection method (Ji et al., BJC 2025).

Methodology: same stratified train/test split as the other scripts. Differential
expression filtering on the training set removes constant genes and keeps genes with
|log2 fold change| above a threshold and a Mann-Whitney U p-value below 0.05. The
surviving genes are standardized and passed to L1-penalized logistic regression tuned
by 100-fold cross-validation, where the penalty is chosen by the one-standard-error
rule on log loss; genes with non-zero coefficients are ranked by absolute coefficient.
Incremental top-k evaluation reports CV and holdout AUC for growing panels, k* is the
smallest panel within DELTA_AUC_SIG of the best CV AUC, and an unpenalized binomial GLM
is fitted on the saved panel.
"""

import warnings;
from pathlib import Path

from utilz.multi_residual_bootstrap import MultiCovariateResidualBootstrapTransformer, build_covariates

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).stem
Path(OUT_DIR).mkdir(exist_ok=True)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, Log2FCReductor, MannWhitneyReductor,
)

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED         = 2137
LOG2FC_THRESHOLD  = np.log2(1.2)
DEG_PVAL          = 0.05
LASSO_CV_FOLDS    = 100
LASSO_N_LAMBDAS   = 50
TOP_K_MAX         = 12
INCR_CV_FOLDS     = 20
DELTA_AUC_SIG     = 0.005


def lasso_cv_lambda_1se(X, y, n_folds=LASSO_CV_FOLDS,
                        n_lambdas=LASSO_N_LAMBDAS, seed=BASE_SEED):
    n_folds = min(n_folds, np.bincount(y.astype(int)).min())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv = LogisticRegressionCV(
        Cs=n_lambdas, cv=skf, penalty='l1', solver='saga',
        scoring='neg_log_loss', class_weight='balanced',
        max_iter=20000, n_jobs=-1, random_state=seed, refit=False,
    ).fit(X, y)

    losses = -cv.scores_[1]
    mean_loss = losses.mean(axis=0)
    se_loss = losses.std(axis=0, ddof=1) / np.sqrt(n_folds)
    k_min = int(np.argmin(mean_loss))
    threshold = mean_loss[k_min] + se_loss[k_min]
    k_1se = int(np.where(mean_loss <= threshold)[0].min())
    C_1se = float(cv.Cs_[k_1se])
    print(f"[LASSO] {n_folds}-fold, lambda.1se -> C={C_1se:.5g} "
          f"loss={mean_loss[k_1se]:.4f} (lambda.min loss={mean_loss[k_min]:.4f}+-{se_loss[k_min]:.4f})")

    final = LogisticRegression(
        penalty='l1', solver='saga', C=C_1se, max_iter=20000,
        class_weight='balanced', random_state=seed,
    ).fit(X, y)
    return final.coef_.ravel()


def incremental_topk_eval(X, y, ranked_idx, k_max=TOP_K_MAX,
                          n_folds=INCR_CV_FOLDS, seed=BASE_SEED,
                          X_test=None, y_test=None):
    n_folds = min(n_folds, np.bincount(y.astype(int)).min())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for k in range(1, min(k_max, len(ranked_idx)) + 1):
        idx = ranked_idx[:k]
        aucs = []
        for tr, va in skf.split(X, y):
            mdl = LogisticRegression(
                max_iter=20000, class_weight='balanced', random_state=seed,
            ).fit(X[np.ix_(tr, idx)], y[tr])
            aucs.append(roc_auc_score(y[va], mdl.predict_proba(X[np.ix_(va, idx)])[:, 1]))
        row = {'k': k, 'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs, ddof=1)}
        if X_test is not None and y_test is not None:
            mdl_full = LogisticRegression(
                max_iter=20000, class_weight='balanced', random_state=seed,
            ).fit(X[:, idx], y)
            row['auc_test'] = roc_auc_score(
                y_test, mdl_full.predict_proba(X_test[:, idx])[:, 1]
            )
            msg = (f"  top-{k:>2}  CV AUC = {row['auc_mean']:.4f}+-{row['auc_std']:.4f}"
                   f"  | test AUC = {row['auc_test']:.4f}")
        else:
            msg = f"  top-{k:>2}  CV AUC = {row['auc_mean']:.4f}+-{row['auc_std']:.4f}"
        rows.append(row)
        print(msg)
    return pd.DataFrame(rows)


def pick_minimal_k(incr_df, delta=DELTA_AUC_SIG):
    auc = incr_df['auc_mean'].values
    return int(incr_df['k'].values[auc >= auc.max() - delta].min())


def main():
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

    print("\n=== KROK 1: DEG ===")
    deg_pipe = Pipeline([
        ('const',  ConstantExpressionReductor()),
        ('log2fc', Log2FCReductor(min_abs_log2fc=LOG2FC_THRESHOLD)),
        ('pval',   MannWhitneyReductor(alpha=DEG_PVAL)),
    ])
    X_tr_deg_df = deg_pipe.fit_transform(X_tr_raw, y_train)
    X_te_deg_df = deg_pipe.transform(X_te_raw)
    deg_genes = list(X_tr_deg_df.columns)

    print("\n=== KROK 2: LASSO ===")
    scaler = StandardScaler().fit(X_tr_deg_df.values)
    X_tr_z = scaler.transform(X_tr_deg_df.values)
    X_te_z = scaler.transform(X_te_deg_df.values)
    coefs = lasso_cv_lambda_1se(X_tr_z, y_train.values)
    print(f"[LASSO] niezerowe wsp.: {(coefs != 0).sum()} / {len(coefs)}")

    rank_df = (pd.DataFrame({'gene': deg_genes, 'coef': coefs})
               .assign(abs_coef=lambda d: d['coef'].abs())
               .query('coef != 0')
               .sort_values('abs_coef', ascending=False)
               .reset_index(drop=True))
    print("\nTop 20 wg |coef|:")
    print(rank_df.head(20)[['gene', 'coef']].to_string())

    print("\n=== KROK 3: top-k ===")
    g2c = {g: i for i, g in enumerate(deg_genes)}
    ranked_idx = [g2c[g] for g in rank_df['gene'].tolist()]
    incr_df = incremental_topk_eval(
        X_tr_z, y_train.values, ranked_idx,
        X_test=X_te_z, y_test=y_test.values,
    )
    k_star = pick_minimal_k(incr_df)
    selected_genes = rank_df['gene'].head(TOP_K_MAX).tolist()
    if len(selected_genes) < TOP_K_MAX:
        print(f"[UWAGA] tylko {len(selected_genes)} genow z niezerowym wsp. LASSO "
              f"(< TOP_K_MAX={TOP_K_MAX})")
    print(f"\nk* (informacyjnie) = {k_star}; zapis TOP_K_MAX={TOP_K_MAX}; "
          f"geny: {selected_genes}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(incr_df['k'], incr_df['auc_mean'], yerr=incr_df['auc_std'],
                marker='o', capsize=3, label=f'CV AUC ({INCR_CV_FOLDS}-fold, train)')
    if 'auc_test' in incr_df.columns:
        ax.plot(incr_df['k'], incr_df['auc_test'],
                marker='s', linestyle='--', color='tab:green',
                label='Holdout test AUC')
    ax.set(xlabel='Top-k genes', ylabel='AUC',
           title=f'Top-k AUC (train CV {INCR_CV_FOLDS} folds vs holdout test)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    print("\n=== KROK 4: GLM ===")
    idx_sel = [g2c[g] for g in selected_genes]
    glm = LogisticRegression(
        penalty=None, solver='lbfgs', max_iter=20000,
        class_weight='balanced', random_state=BASE_SEED,
    ).fit(X_tr_z[:, idx_sel], y_train.values)

    auc_tr = roc_auc_score(y_train.values, glm.predict_proba(X_tr_z[:, idx_sel])[:, 1])
    auc_te = roc_auc_score(y_test.values,  glm.predict_proba(X_te_z[:, idx_sel])[:, 1])
    print(f"coef:       {dict(zip(selected_genes, glm.coef_.ravel().round(4)))}")
    print(f"intercept:  {glm.intercept_[0]:.4f}")
    print(f"Train AUC:  {auc_tr:.4f}")
    print(f"Holdout AUC:{auc_te:.4f}")

    out_path = f"{OUT_DIR}/pla2sig_selected_genes.csv"
    glm_coefs = glm.coef_.ravel()
    out_df = (pd.DataFrame({
        'gene':      selected_genes,
        'lasso_coef': [rank_df.set_index('gene').loc[g, 'coef'] for g in selected_genes],
        'glm_coef':   glm_coefs,
        'rank':       np.arange(1, len(selected_genes) + 1),
    }))
    out_df.loc[len(out_df)] = {
        'gene': '__intercept__',
        'lasso_coef': np.nan,
        'glm_coef':   float(glm.intercept_[0]),
        'rank':       0,
    }
    out_df.attrs.update({
        'k_star':       k_star,
        'top_k_max':    int(TOP_K_MAX),
        'train_auc':    float(auc_tr),
        'holdout_auc':  float(auc_te),
        'n_train':      int(len(y_train)),
        'n_test':       int(len(y_test)),
    })
    out_df.to_csv(out_path, index=False)
    print(f"\n[OK] zapisano {len(selected_genes)} genow + intercept -> {out_path}")


if __name__ == '__main__':
    main()
