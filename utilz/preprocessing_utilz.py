"""Scikit-learn transformers for gene-level filtering before model fitting.

Each transformer learns the set of retained genes on the training data only and
subsets the columns at transform time. ConstantExpressionReductor drops genes with a
single unique value. MeanExpressionReductor keeps genes whose mean expression exceeds a
given percentile of the mean expression distribution. Log2FCReductor keeps genes whose
absolute log2 fold change between the two classes exceeds a threshold.
MannWhitneyReductor keeps genes with a two-sided Mann-Whitney U p-value below alpha
(p < 0.05 in PLA2Sig). AnovaFdrReductor keeps genes whose ANOVA F-test p-value survives
Benjamini-Hochberg FDR correction. WithinGroupVarianceReductor pools the within-class
variance of each gene, compares it against the median gene variance with a chi-square
test and keeps genes whose FDR-corrected p-value is at most alpha.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import f_classif
from scipy import stats


class AnovaFdrReductor(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.selected_genes_ = None

    def fit(self, X, y=None):
        F, p = f_classif(X, y)
        _, p_corrected, _, _ = multipletests(p, alpha=self.alpha, method='fdr_bh')
        self.selected_genes_ = X.columns[p_corrected < self.alpha]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after AnovaFdrReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class MeanExpressionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=25):
        self.selected_genes_ = None
        self.percentile = percentile

    def fit(self, X, y=None):
        mean_per_gene = X.mean(axis=0)
        threshold = np.percentile(mean_per_gene, self.percentile)
        self.selected_genes_ = mean_per_gene[mean_per_gene > threshold].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after MeanExpressionReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class ConstantExpressionReductor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_genes_ = None

    def fit(self, X, y=None):
        num_unique = X.nunique(dropna=True)
        self.selected_genes_ = num_unique[num_unique > 1].index
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after ConstantExpressionReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class Log2FCReductor(BaseEstimator, TransformerMixin):

    def __init__(self, min_abs_log2fc=1.0):
        self.min_abs_log2fc = min_abs_log2fc
        self.selected_genes_ = None
        self.log2fc_ = None
        self.abs_log2fc_ = None
        self.classes_ = None

    def fit(self, X, y=None):
        group_means = X.groupby(y).mean()
        class0, class1 = group_means.index
        self.classes_ = (class0, class1)
        self.log2fc_ = group_means.loc[class1] - group_means.loc[class0]
        self.abs_log2fc_ = self.log2fc_.abs()
        self.selected_genes_ = X.columns[self.abs_log2fc_ > self.min_abs_log2fc]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after Log2FCReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class MannWhitneyReductor(BaseEstimator, TransformerMixin):

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.selected_genes_ = None
        self.pvalues_ = None

    def fit(self, X, y):
        X0 = X.loc[y == 0].values
        X1 = X.loc[y == 1].values
        _, p = stats.mannwhitneyu(X0, X1, alternative='two-sided', axis=0)
        self.pvalues_ = pd.Series(p, index=X.columns)
        self.selected_genes_ = X.columns[self.pvalues_ < self.alpha]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print("data shape after MannWhitneyReductor: ", X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)


class WithinGroupVarianceReductor(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.selected_genes_ = None
        self.wgv_ = None
        self.pvalues_ = None
        self.pvalues_corrected_ = None
        self.sigma2_ref_ = None

    def fit(self, X: pd.DataFrame, y=None):
        y_s = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) \
              else y.reindex(X.index)
        classes = y_s.unique()
        N, K = len(X), len(classes)
        df = N - K
        wgv = pd.Series(0.0, index=X.columns)
        for cls in classes:
            mask = y_s == cls
            n_k = int(mask.sum())
            if n_k > 1:
                wgv += (n_k - 1) * X.loc[mask].var(ddof=1)
        wgv /= df
        self.wgv_ = wgv
        sigma2_ref = float(wgv.median())
        self.sigma2_ref_ = sigma2_ref

        T = df * wgv.values / sigma2_ref
        pvals = stats.chi2.sf(T, df)
        self.pvalues_ = pd.Series(pvals, index=X.columns)
        _, p_corr, _, _ = multipletests(pvals, alpha=self.alpha,
                                        method='fdr_bh')
        self.pvalues_corrected_ = pd.Series(p_corr, index=X.columns)
        self.selected_genes_ = X.columns[self.pvalues_corrected_ <= self.alpha]
        return self

    def transform(self, X):
        X = X[self.selected_genes_]
        print(f"data shape after WithinGroupVarianceReductor: {X.shape} "
              f"sigma^2_ref = {self.sigma2_ref_:.4g}, alpha = {self.alpha})")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_genes_, dtype=object)
