"""Comparison of alternative classifiers on the selected gene panel.

Methodology: loads the panel produced by forward selection and reuses the same
train/test split and covariate correction (multi-covariate residual bootstrap on
age, sex and log10 library size, applied to all panel genes) followed by
standardization. Four models are compared: elastic-net logistic regression, XGBoost,
RBF SVM - each tuned by grid search over multi-criteria stratified folds on the
training set - and a stacking ensemble of the three tuned models with a logistic
regression meta-learner (stacking falls back to plain StratifiedKFold because
cross_val_predict requires a partition). Evaluation on the holdout test set: AUC bar
plot, ROC curves, 2x2 confusion matrices at the Youden threshold determined on the
training set, and 3x2 confusion matrices that keep the original three groups as true
labels to show how patients with benign pancreatic disease are classified.
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
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY, CANCER
from utilz.preprocessing_utilz import (
    ConstantExpressionReductor, Log2FCReductor, MannWhitneyReductor,
)
from utilz.multi_residual_bootstrap import (
    MultiCovariateResidualBootstrapTransformer, build_covariates,
)

meta_path        = r"../data/samples_pancreatic.xlsx"
data_path        = r"../data/counts_pancreatic.csv"
GENES_CSV        = "4_forward_selection/forward_selection_genes.csv"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED         = 2137

GRID_CV_FOLDS     = 30
N_JOBS            = -1

OUT_PNG_BARS      = f"{OUT_DIR}/alt_models_test_auc.png"
OUT_PNG_ROC       = f"{OUT_DIR}/alt_models_roc.png"
OUT_PNG_CM        = f"{OUT_DIR}/alt_models_confusion.png"
OUT_CSV_CM        = f"{OUT_DIR}/alt_models_confusion.csv"
OUT_PNG_CM_3x2    = f"{OUT_DIR}/alt_models_confusion_3x2.png"
OUT_CSV_CM_3x2    = f"{OUT_DIR}/alt_models_confusion_3x2.csv"

PL_LABELS = {
    HEALTHY: "zdrowe",
    DISEASE: "choroby trzustki",
    CANCER:  "nowotwór trzustki",
}


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


def fit_and_eval(name, spec, X_tr, y_tr, X_te, y_te, cv, seed):
    print(f"\n--- {name} ---")
    fit_params = {}
    if name == 'xgboost':
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spec['estimator'].set_params(scale_pos_weight=neg / max(pos, 1))

    gs = GridSearchCV(
        spec['estimator'], spec['param_grid'],
        scoring='roc_auc', cv=cv, n_jobs=N_JOBS, refit=True,
        return_train_score=False, verbose=0,
    ).fit(X_tr, y_tr, **fit_params)

    proba_tr = gs.best_estimator_.predict_proba(X_tr)[:, 1]
    proba_te = gs.best_estimator_.predict_proba(X_te)[:, 1]
    auc_cv   = gs.best_score_
    auc_tr   = roc_auc_score(y_tr, proba_tr)
    auc_te   = roc_auc_score(y_te, proba_te)

    print(f"  best params : {gs.best_params_}")
    print(f"  CV AUC (train, {GRID_CV_FOLDS}-fold): {auc_cv:.4f}")
    print(f"  Train AUC (refit on full train)    : {auc_tr:.4f}")
    print(f"  Holdout test AUC                   : {auc_te:.4f}")

    return {
        'model':       name,
        'best_params': gs.best_params_,
        'cv_auc':      float(auc_cv),
        'train_auc':   float(auc_tr),
        'test_auc':    float(auc_te),
        'proba_tr':    proba_tr,
        'proba_te':    proba_te,
        'estimator':   gs.best_estimator_,
    }


def fit_and_eval_stacking(base_results, X_tr, y_tr, X_te, y_te, cv, seed):
    print(f"\n--- stacking ---")
    estimators = [(r['model'], clone(r['estimator'])) for r in base_results]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=20000, class_weight='balanced', random_state=seed,
        ),
        stack_method='predict_proba',
        cv=cv, n_jobs=N_JOBS, passthrough=False,
    )

    auc_cv = float(np.mean(cross_val_score(
        clone(stack), X_tr, y_tr, scoring='roc_auc', cv=cv, n_jobs=N_JOBS,
    )))
    stack.fit(X_tr, y_tr)
    proba_tr = stack.predict_proba(X_tr)[:, 1]
    proba_te = stack.predict_proba(X_te)[:, 1]
    auc_tr   = roc_auc_score(y_tr, proba_tr)
    auc_te   = roc_auc_score(y_te, proba_te)

    print(f"  base models : {[r['model'] for r in base_results]}")
    print(f"  CV AUC (train, {GRID_CV_FOLDS}-fold): {auc_cv:.4f}")
    print(f"  Train AUC (refit on full train)    : {auc_tr:.4f}")
    print(f"  Holdout test AUC                   : {auc_te:.4f}")

    return {
        'model':       'stacking',
        'best_params': {'final_estimator': 'logreg', 'base': [r['model'] for r in base_results]},
        'cv_auc':      auc_cv,
        'train_auc':   float(auc_tr),
        'test_auc':    float(auc_te),
        'proba_tr':    proba_tr,
        'proba_te':    proba_te,
        'estimator':   stack,
    }


def youden_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def confusion_3x2_single(y_pred, y_index, ds, le):
    true_group = ds.meta.loc[y_index, 'Group']
    pred_label = pd.Series(
        np.where(np.asarray(y_pred) == 1, le.classes_[1], le.classes_[0]),
        index=y_index, name='przewidziana',
    )
    row_order = [HEALTHY, DISEASE, CANCER]
    col_order = [le.classes_[0], le.classes_[1]]
    return (pd.crosstab(true_group, pred_label)
              .reindex(index=row_order, columns=col_order, fill_value=0))


def main():
    if not os.path.exists(GENES_CSV):
        raise FileNotFoundError(
            f"Brak {GENES_CSV} - uruchom najpierw 4_pla2sig_gene_selection.py"
        )
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
        raise ValueError(f"Geny z CSV nie sa dostepne w danych: {missing[:5]}...")
    X_tr_raw = X_tr_raw[selected_genes]
    X_te_raw = X_te_raw[selected_genes]

    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")


    print("\n=== korekcja na 12 genach panelu ===")
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

    cv = ds.get_stratified_kfold(X_tr_raw, y_train, n_splits=GRID_CV_FOLDS, random_state=BASE_SEED)
    cv_stack = StratifiedKFold(
        n_splits=min(GRID_CV_FOLDS, int(np.bincount(y_tr_np).min())),
        shuffle=True, random_state=BASE_SEED,
    )
    grids = get_model_grids(BASE_SEED)

    results = []
    for name, spec in grids.items():
        results.append(fit_and_eval(
            name, spec, X_tr_z, y_tr_np, X_te_z, y_te_np, cv, BASE_SEED,
        ))

    results.append(fit_and_eval_stacking(
        results, X_tr_z, y_tr_np, X_te_z, y_te_np, cv_stack, BASE_SEED,
    ))

    summary = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ('proba_te', 'estimator')}
                            for r in results])
    print("\n=== PODSUMOWANIE ===")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(results))
    w = 0.35
    ax.bar(x - w/2, summary['cv_auc'],   w, label=f'CV AUC ({GRID_CV_FOLDS}-fold, train)')
    ax.bar(x + w/2, summary['test_auc'], w, label='Holdout test AUC',
           color='tab:green')
    for i, (cv_v, te_v) in enumerate(zip(summary['cv_auc'], summary['test_auc'])):
        ax.text(i - w/2, cv_v + 0.005, f'{cv_v:.3f}', ha='center', fontsize=8)
        ax.text(i + w/2, te_v + 0.005, f'{te_v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(summary['model'])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('AUC')
    ax.set_title(f'Modele na {len(selected_genes)} genach PLA2Sig')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(OUT_PNG_BARS, dpi=140); plt.show()
    print(f"[OK] bar plot -> {OUT_PNG_BARS}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_te_np, r['proba_te'])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', ls='--', alpha=0.5)
    ax.set(xlabel='FPR', ylabel='TPR', title='ROC - holdout test')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_PNG_ROC, dpi=140); plt.show()
    print(f"[OK] ROC plot  -> {OUT_PNG_ROC}")

    print("\n=== MACIERZE POMYLEK (prog Youdena wyznaczony na train) ===")
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    cm_rows = []
    for ax, r in zip(axes, results):
        thr = youden_threshold(y_tr_np, r['proba_tr'])
        y_pred = (r['proba_te'] >= thr).astype(int)
        cm = confusion_matrix(y_te_np, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        acc  = (tp + tn) / cm.sum()
        cm_rows.append({
            'model': r['model'], 'prog_youden': round(thr, 4),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'czulosc': round(float(sens), 4),
            'swoistosc': round(float(spec), 4),
            'dokladnosc': round(float(acc), 4),
        })
        print(f"  {r['model']:<20} prog={thr:.4f}  TN={tn} FP={fp} FN={fn} TP={tp}  "
              f"czulosc={sens:.3f} swoistosc={spec:.3f} dokladnosc={acc:.3f}")
        ConfusionMatrixDisplay(cm, display_labels=['kontrola', 'nowotwor']).plot(
            ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"{r['model']}\nprog={thr:.2f}")
    plt.tight_layout(); plt.savefig(OUT_PNG_CM, dpi=140); plt.show()
    pd.DataFrame(cm_rows).to_csv(OUT_CSV_CM, index=False)
    print(f"[OK] confusion matrices -> {OUT_PNG_CM}")
    print(f"[OK] confusion summary  -> {OUT_CSV_CM}")

    print("\n=== MACIERZE POMYLEK 3x2 (3 klasy prawdziwe x 2 przewidziane) ===")
    n = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    cm3_rows = []
    for ax, r in zip(axes, results):
        thr = youden_threshold(y_tr_np, r['proba_tr'])
        y_pred = (r['proba_te'] >= thr).astype(int)
        cm = confusion_3x2_single(y_pred, y_test.index, ds, le)

        print(f"\n--- {r['model']} (prog={thr:.4f}) ---")
        print(cm.to_string())
        for grp in cm.index:
            cm3_rows.append({
                'model': r['model'], 'prog_youden': round(thr, 4),
                'prawdziwa_grupa': grp,
                f'pred_{cm.columns[0]}': int(cm.loc[grp, cm.columns[0]]),
                f'pred_{cm.columns[1]}': int(cm.loc[grp, cm.columns[1]]),
            })

        ax.imshow(cm.values, cmap='Blues')
        ax.set_xticks(range(cm.shape[1]))
        ax.set_xticklabels([PL_LABELS.get(c, c) for c in cm.columns],
                           rotation=15, ha='right')
        ax.set_yticks(range(cm.shape[0]))
        ax.set_yticklabels([PL_LABELS.get(r_, r_) for r_ in cm.index])
        vmax = cm.values.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = int(cm.values[i, j])
                ax.text(j, i, v, ha='center', va='center',
                        color='white' if v > vmax / 2 else 'black')
        ax.set_xlabel('Klasa przewidziana')
        ax.set_ylabel('Klasa prawdziwa')
        ax.set_title(f"{r['model']}\nprog={thr:.2f}")

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout(h_pad=3.0); plt.savefig(OUT_PNG_CM_3x2, dpi=140); plt.show()
    pd.DataFrame(cm3_rows).to_csv(OUT_CSV_CM_3x2, index=False)
    print(f"[OK] macierze 3x2     -> {OUT_PNG_CM_3x2}")
    print(f"[OK] podsumowanie 3x2 -> {OUT_CSV_CM_3x2}")


if __name__ == '__main__':
    main()
