"""Calibrated stacking ensemble on the selected gene panel.

Methodology: same split, covariate correction and gene panel as
5_alternative_models.py. Elastic-net logistic regression, XGBoost and an RBF SVM are
first tuned by grid search over multi-criteria stratified folds, then combined in a
stacking classifier with a logistic regression meta-learner (plain StratifiedKFold
inside the stack, since cross_val_predict requires a partition), and the whole stack is
wrapped in CalibratedClassifierCV with sigmoid calibration. Reliability is assessed by
comparing calibration curves and Brier scores of the raw and calibrated stack plus a
histogram of predicted probabilities per class. The Youden-optimal threshold is taken
from the training set and applied to the holdout test set, reported as a classification
report, a 2x2 confusion matrix, metadata of misclassified samples, and a 3x2 confusion
matrix of the three original groups against the two predicted classes.
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).stem
Path(OUT_DIR).mkdir(exist_ok=True)

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, brier_score_loss,
)

from xgboost import XGBClassifier

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY, CANCER
from utilz.helpers import show_report
from utilz.multi_residual_bootstrap import (
    MultiCovariateResidualBootstrapTransformer, build_covariates,
)

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"
GENES_CSV = "4_forward_selection/forward_selection_genes.csv"

OUT_PNG_CALIB  = f"{OUT_DIR}/stacking_calibration.png"
OUT_PNG_CM_3x2 = f"{OUT_DIR}/stacking_confusion_3x2.png"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED  = 2137

GRID_CV_FOLDS = 30
N_JOBS        = -1

CALIB_METHOD = 'sigmoid'
CALIB_CV     = 10


def get_model_grids(seed):
    return {
        'logreg_elasticnet': {
            'estimator': LogisticRegression(
                penalty='elasticnet', solver='saga',
                max_iter=20000, class_weight='balanced',
                random_state=seed,
            ),
            'param_grid': {
                'C':        [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                'l1_ratio': [0, 0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        'xgboost': {
            'estimator': XGBClassifier(
                objective='binary:logistic', eval_metric='logloss',
                tree_method='hist', random_state=seed, n_jobs=1,
                verbosity=0,
            ),
            'param_grid': {
                'n_estimators':     [50, 100, 200, 400],
                'max_depth':        [1, 2, 3, 6],
                'learning_rate':    [0.01, 0.03, 0.1, 0.2, 0.3],
                'min_child_weight': [1, 5, 10, 20, 50],
                'reg_lambda':       [0.5, 1.0, 2.0],
                'subsample':        [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            },
        },
        'svm_rbf': {
            'estimator': SVC(
                kernel='rbf', probability=True, class_weight='balanced',
                random_state=seed,
            ),
            'param_grid': {
                'C':     [0.1, 1.0, 10.0, 100.0, 200],
                'gamma': ['scale', 0.001, 0.01, 0.1],
            },
        },
    }


def tune_base(name, spec, X_tr, y_tr, cv):
    print(f"\n--- tuning {name} ---")
    if name == 'xgboost':
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spec['estimator'].set_params(scale_pos_weight=neg / max(pos, 1))
    gs = GridSearchCV(
        spec['estimator'], spec['param_grid'],
        scoring='roc_auc', cv=cv, n_jobs=N_JOBS, refit=True,
        return_train_score=False, verbose=0,
    ).fit(X_tr, y_tr)
    print(f"  best params : {gs.best_params_}")
    print(f"  best CV AUC : {gs.best_score_:.4f}")
    return gs.best_estimator_


def youden_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def plot_calibration(y_true, proba_before, proba_after,
                     out_png="stacking_calibration.png",
                     n_bins=5, hist_bins=20):
    y_true = np.asarray(y_true)
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]})

    ax.plot([0, 1], [0, 1], 'k--', label='idealna kalibracja')
    for proba, name in [(proba_before, 'przed kalibracją'),
                         (proba_after, 'po kalibracji')]:
        frac_pos, mean_pred = calibration_curve(
            y_true, proba, n_bins=n_bins, strategy='quantile')
        bs = brier_score_loss(y_true, proba)
        ax.plot(mean_pred, frac_pos, marker='o', label=f'{name} (Brier={bs:.3f})')
    ax.set_ylabel('Obserwowana częstość nowotworu (accuracy)')
    ax.set_title('Krzywa niezawodności stacking (przed vs po kalibracji)')
    ax.legend(); ax.grid(alpha=0.3)

    bins = np.linspace(0, 1, hist_bins + 1)
    axh.hist(proba_after[y_true == 0], bins=bins, alpha=0.6,
             color='tab:blue', label='kontrola')
    axh.hist(proba_after[y_true == 1], bins=bins, alpha=0.6,
             color='tab:red', label='nowotwór')
    axh.set_xlabel('Przewidziane prawdopodobieństwo nowotworu (confidence)')
    axh.set_ylabel('Liczba próbek')
    axh.legend(); axh.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.show()
    print(f"[OK] krzywa kalibracji + histogram -> {out_png}")


def confusion_3x2(y_pred, y_index, ds, le, out_png="stacking_confusion_3x2.png"):
    true_group = ds.meta.loc[y_index, 'Group']
    pred_label = pd.Series(
        np.where(np.asarray(y_pred) == 1, le.classes_[1], le.classes_[0]),
        index=y_index, name='przewidziana',
    )
    row_order = [HEALTHY, DISEASE, CANCER]
    col_order = [le.classes_[0], le.classes_[1]]
    cm = (pd.crosstab(true_group, pred_label)
            .reindex(index=row_order, columns=col_order, fill_value=0))

    print("\n=== Macierz pomylek 3x2 (prawdziwe 3 klasy x przewidziane 2) ===")
    print(cm.to_string())

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.imshow(cm.values, cmap='Blues')
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=15, ha='right')
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    vmax = cm.values.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm.values[i, j])
            ax.text(j, i, v, ha='center', va='center',
                    color='white' if v > vmax / 2 else 'black')
    ax.set_xlabel('Klasa przewidziana')
    ax.set_ylabel('Klasa prawdziwa')
    ax.set_title('Macierz pomylek (3 klasy prawdziwe, 2 przewidziane)')
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.show()
    print(f"[OK] macierz 3x2 -> {out_png}")
    return cm


def build_stacking(base_estimators, seed, cv_stack):
    return StackingClassifier(
        estimators=[(name, clone(est)) for name, est in base_estimators],
        final_estimator=LogisticRegression(
            max_iter=20000, class_weight='balanced', random_state=seed,
        ),
        stack_method='predict_proba',
        cv=cv_stack, n_jobs=N_JOBS, passthrough=False,
    )


def main():
    if not os.path.exists(GENES_CSV):
        raise FileNotFoundError(f"Brak {GENES_CSV} - uruchom najpierw selekcje genow.")
    genes_df = pd.read_csv(GENES_CSV)
    selected_genes = genes_df.loc[genes_df['gene'] != '__intercept__', 'gene'].tolist()
    print(f"[INFO] wczytano {len(selected_genes)} genow z {GENES_CSV}")

    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    X_tr_raw, X_te_raw, X_va_raw, y_train, y_test, y_valid = ds.get_train_test_valid_split(
        ds.X, y_enc, test_size=TEST_SIZE, valid_size=VALID_SIZE,
        random_state=BASE_SEED,
    )
    X_te_raw = pd.concat([X_te_raw, X_va_raw])
    y_test   = pd.concat([y_test, y_valid])

    missing = [g for g in selected_genes if g not in X_tr_raw.columns]
    if missing:
        raise ValueError(f"Geny z CSV nie sa dostepne: {missing[:5]}...")
    X_tr_raw = X_tr_raw[selected_genes]
    X_te_raw = X_te_raw[selected_genes]

    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    cov = build_covariates(ds.meta)
    deg_pipe = Pipeline([
        ('multi_resid', MultiCovariateResidualBootstrapTransformer(
            covariates=cov, labels=y_train,
            n_bootstrap=2, fdr_alpha=1, min_r2=0, cv_threshold_pct=1e9,
        )),
    ])
    X_tr_deg_df = deg_pipe.fit_transform(X_tr_raw, y_train)
    X_te_deg_df = deg_pipe.transform(X_te_raw)

    scaler = StandardScaler().fit(X_tr_deg_df.values)
    X_tr_z = scaler.transform(X_tr_deg_df.values)
    X_te_z = scaler.transform(X_te_deg_df.values)
    y_tr_np = y_train.values
    y_te_np = y_test.values

    cv_grid = ds.get_stratified_kfold(
        X_tr_raw, y_train, n_splits=GRID_CV_FOLDS, random_state=BASE_SEED,
    )
    cv_stack = StratifiedKFold(
        n_splits=min(GRID_CV_FOLDS, int(np.bincount(y_tr_np).min())),
        shuffle=True, random_state=BASE_SEED,
    )
    cv_calib = ds.get_stratified_kfold(
        X_tr_raw, y_train, n_splits=CALIB_CV, random_state=BASE_SEED,
    )

    grids = get_model_grids(BASE_SEED)
    base_estimators = []
    for name, spec in grids.items():
        best = tune_base(name, spec, X_tr_z, y_tr_np, cv_grid)
        base_estimators.append((name, best))

    stack_calib = CalibratedClassifierCV(
        build_stacking(base_estimators, BASE_SEED, cv_stack),
        method=CALIB_METHOD, cv=cv_calib,
    )
    stack_calib.fit(X_tr_z, y_tr_np)

    proba_tr = stack_calib.predict_proba(X_tr_z)[:, 1]
    proba_te = stack_calib.predict_proba(X_te_z)[:, 1]
    auc_te = roc_auc_score(y_te_np, proba_te)
    print(f"\n[stacking skalibrowany] kalibracja={CALIB_METHOD}, cv={CALIB_CV}")
    print(f"Bazowe modele: {[n for n, _ in base_estimators]}")
    print(f"Holdout test AUC: {auc_te:.4f}")

    raw_stack = build_stacking(base_estimators, BASE_SEED, cv_stack).fit(X_tr_z, y_tr_np)
    proba_before = raw_stack.predict_proba(X_te_z)[:, 1]
    plot_calibration(y_te_np, proba_before, proba_te, out_png=OUT_PNG_CALIB)

    thr = youden_threshold(y_tr_np, proba_tr)
    y_pred = (proba_te >= thr).astype(int)
    print(f"Prog Youdena (z train): {thr:.4f}")

    print("\n=== classification_report (test) ===")
    print(classification_report(y_te_np, y_pred, target_names=le.classes_, digits=3))
    print("Macierz pomylek [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_te_np, y_pred))

    print("\n=== show_report (metadane probek FN/FP) ===")
    show_report(y_pred, y_test, ds, le)

    confusion_3x2(y_pred, y_test.index, ds, le, out_png=OUT_PNG_CM_3x2)


if __name__ == '__main__':
    main()
