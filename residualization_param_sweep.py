"""
Przeszukiwanie parametrow MultiCovariateResidualBootstrapTransformer pod katem
usuwania sygnalu wieku. Im luzniejsze filtry (nizsze min_r2, wyzsze
cv_threshold_pct, wyzsze fdr_alpha), tym wiecej genow podlega korekcji, wiec
przewidywalnosc wieku po korekcji powinna w koncu zaczac spadac.

Sonda: RidgeCV (szybka dzieki LOO). R^2 "przed" liczony raz.
Uruchom:  python residualization_param_sweep.py
"""
import io, contextlib, itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from utilz.Dataset import load_dataset
from utilz.constans import HEALTHY, DISEASE, CANCER
from utilz.preprocessing_utilz import ConstantExpressionReductor
from utilz.multi_residual_bootstrap import (
    MultiCovariateResidualBootstrapTransformer, build_covariates,
)

# --- dane i podzial (jak w notatniku) ---
ds = load_dataset("../data/counts_pancreatic.csv", "../data/samples_pancreatic.xlsx", label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.2, valid_size=0.2)
)
cov = build_covariates(ds.meta)

# --- usuniecie genow stalych (raz) ---
const_red = ConstantExpressionReductor().fit(X_train, y_train)
Xtr_raw = const_red.transform(X_train)
Xte_raw = const_red.transform(X_test)

# --- cel: wiek ---
age = ds.age.dropna().astype(float)
tr = Xtr_raw.index.intersection(age.index)
te = Xte_raw.index.intersection(age.index)


def probe_r2(Xtr, Xte):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', RidgeCV(alphas=np.logspace(1, 4, 7)))])
    pipe.fit(Xtr.loc[tr], age.loc[tr])
    return r2_score(age.loc[te], pipe.predict(Xte.loc[te]))


r2_before = probe_r2(Xtr_raw, Xte_raw)
print(f"genow po usunieciu stalych: {Xtr_raw.shape[1]}")
print(f"R^2 wieku PRZED korekcja: {r2_before:+.4f}\n")

# --- siatka parametrow (od luznych do scislych) ---
grid_fdr = [0.2, 0.05]
grid_min_r2 = [0.0, 0.01, 0.05]
grid_cv = [1000.0, 100.0, 30.0]
N_BOOT = 200  # mniej niz docelowe 1000, dla szybkosci przeszukiwania

rows = []
for fdr, mr2, cvp in itertools.product(grid_fdr, grid_min_r2, grid_cv):
    with contextlib.redirect_stdout(io.StringIO()):
        resid = MultiCovariateResidualBootstrapTransformer(
            covariates=cov, labels=y_train,
            n_bootstrap=N_BOOT, fdr_alpha=fdr, min_r2=mr2, cv_threshold_pct=cvp,
        ).fit(Xtr_raw, y_train)
        Xtr_c = resid.transform(Xtr_raw)
        Xte_c = resid.transform(Xte_raw)
    n_corr = len(resid.selected_genes_)
    r2_after = probe_r2(Xtr_c, Xte_c)
    delta = r2_after - r2_before
    rows.append((fdr, mr2, cvp, n_corr, r2_after, delta))
    flaga = "  <-- SPADEK" if delta < 0 else ""
    print(f"fdr={fdr:<4} min_r2={mr2:<4} cv%={cvp:<6} skorygowanych={n_corr:>6}  "
          f"R2_po={r2_after:+.4f}  delta={delta:+.4f}{flaga}")

res = pd.DataFrame(rows, columns=["fdr_alpha", "min_r2", "cv_pct", "n_corrected", "r2_age_after", "delta_vs_before"])
res = res.sort_values("r2_age_after")
print("\nPosortowane (najnizsze R^2 wieku na gorze):")
print(res.to_string(index=False))
res.to_csv("residualization_param_sweep.csv", index=False)
print("\nzapisano: residualization_param_sweep.csv")
