"""Removal of covariate-driven expression variation by multi-covariate residual bootstrap.

Methodology: covariates are age, sex and log10 library size. For every gene an OLS
model of expression on the covariates is fitted and its R^2 tested with an F test;
p-values are corrected by Benjamini-Hochberg FDR. Genes that pass the FDR test are
re-fitted on n_bootstrap resamples of the samples, and only genes whose bootstrap
R^2 distribution has a median of at least min_r2 and a coefficient of variation below
cv_threshold_pct are accepted as stably covariate-dependent. The transformer fits the
OLS coefficients on control samples only (labels == 0) to avoid removing
disease-related signal, and at transform time subtracts the predicted covariate
contribution from the accepted genes, leaving the other genes untouched. Samples with
missing covariates are left uncorrected.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests


def _ols_r2(X_cov, Y):
    n = Y.shape[0]
    Xc = X_cov - X_cov.mean(axis=0)
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Y
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        return None, np.zeros(Y.shape[1])
    ess = (XtY * beta).sum(axis=0)
    y_mean = Y.mean(axis=0)
    tss = np.einsum('ij,ij->j', Y, Y) - n * y_mean * y_mean
    return None, np.where(tss > 0, ess / tss, 0.0)


def find_stable_multivariate_genes(
    X, covariates,
    fdr_alpha=0.05, n_bootstrap=500, min_r2=0.05,
    cv_threshold_pct=30.0, random_state=2137,
):
    valid = covariates.dropna(how='any').index.intersection(X.index)
    X = X.loc[valid]
    X_cov = covariates.loc[valid].values.astype(float)
    Y = X.values
    n, p = X_cov.shape

    _, r2 = _ols_r2(X_cov, Y)
    F = (r2 / p) / ((1 - r2) / (n - p - 1))
    pvals = stats.f.sf(F, p, n - p - 1)
    rejected, _, _, _ = multipletests(pvals, alpha=fdr_alpha, method='fdr_bh')
    Y_cand = Y[:, rejected]
    cand_names = X.columns[rejected].tolist()

    rng = np.random.default_rng(random_state)
    r2_boot = np.empty((n_bootstrap, len(cand_names)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        _, r2_boot[b] = _ols_r2(X_cov[idx], Y_cand[idx])

    median_r2 = np.median(r2_boot, axis=0)
    cv_pct = np.std(r2_boot, axis=0, ddof=1) / np.abs(median_r2) * 100
    keep = (median_r2 >= min_r2) & (cv_pct < cv_threshold_pct)

    return pd.DataFrame({
        'gene':      np.array(cand_names)[keep],
        'r2_median': median_r2[keep],
        'r2_cv_pct': cv_pct[keep],
    }).reset_index(drop=True)


class MultiCovariateResidualBootstrapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, covariates, labels=None,
                 fdr_alpha=0.05, n_bootstrap=500, min_r2=0.05,
                 cv_threshold_pct=30.0, random_state=2137):
        self.covariates = covariates
        self.labels = labels
        self.fdr_alpha = fdr_alpha
        self.n_bootstrap = n_bootstrap
        self.min_r2 = min_r2
        self.cv_threshold_pct = cv_threshold_pct
        self.random_state = random_state
        self.selected_genes_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y=None):
        cov = self.covariates.reindex(X.index).astype(float)
        base = self.labels[self.labels == 0].index if self.labels is not None else X.index
        valid = cov.dropna(how='any').index.intersection(base)

        stable = find_stable_multivariate_genes(
            X.loc[valid], cov.loc[valid],
            fdr_alpha=self.fdr_alpha, n_bootstrap=self.n_bootstrap,
            min_r2=self.min_r2, cv_threshold_pct=self.cv_threshold_pct,
            random_state=self.random_state,
        )
        self.selected_genes_ = stable['gene'].tolist()

        lr = LinearRegression().fit(
            cov.loc[valid].values, X.loc[valid, self.selected_genes_].values
        )
        self.coef_ = pd.DataFrame(lr.coef_, index=self.selected_genes_,
                                  columns=cov.columns)
        self.intercept_ = pd.Series(lr.intercept_, index=self.selected_genes_)
        print(f"  [{type(self).__name__}] fit OLS on {len(valid)} probs "
              f"for {len(self.selected_genes_)} stable genes "
              f"({len(cov.columns)} covariates)")
        return self

    def transform(self, X):
        cov = self.covariates.reindex(X.index).astype(float)
        nan_mask = cov.isna().any(axis=1).values
        predicted = cov.fillna(0.0).values @ self.coef_.values.T + self.intercept_.values
        predicted[nan_mask, :] = 0.0
        result = X.copy()
        result[self.selected_genes_] = X[self.selected_genes_].values - predicted
        print(f"data shape after {type(self).__name__}: {result.shape}")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_covariates(meta, age='Age', sex='Sex', libsize='Lib.size', ptprc='PTPRC'):
    return pd.DataFrame({
        'age':           meta[age].astype(float),
        'sex':           meta[sex].map({'F': 0, 'M': 1}).astype(float),
        'log10_libsize': np.log10(meta[libsize].astype(float)),
    }, index=meta.index)
