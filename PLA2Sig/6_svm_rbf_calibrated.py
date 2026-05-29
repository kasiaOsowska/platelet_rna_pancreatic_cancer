"""
Samodzielny skrypt: SVM RBF na panelu genow PLA2Sig, skalibrowany
(CalibratedClassifierCV) tak, aby zwracal prawdopodobienstwa przynaleznosci
do klasy. Uzywa tego samego splitu, DEG-preprocessingu i panelu genow co
5_alternative_models.py. Raport klasyfikacji przez show_report, krzywa
niezawodnosci przed i po kalibracji oraz niekwadratowa macierz pomylek.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, brier_score_loss

from utilz.Dataset import load_dataset
from utilz.constans import DISEASE, HEALTHY, CANCER
from utilz.helpers import show_report
from utilz.multi_residual_bootstrap import (
    MultiCovariateResidualBootstrapTransformer, build_covariates,
)

# ---------------------------------------------------------------------------
# Konfiguracja - zgodna z 5_alternative_models.py
# ---------------------------------------------------------------------------
meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"
GENES_CSV = "forward_selection_genes.csv"

TEST_SIZE  = 0.2
VALID_SIZE = 0.2
BASE_SEED  = 2137

SVM_C        = 1.0
SVM_GAMMA    = 0.01
CALIB_METHOD = 'sigmoid'
CALIB_CV     = 10


def make_svm():
    return SVC(
        kernel='rbf', C=SVM_C, gamma=SVM_GAMMA,
        class_weight='balanced', random_state=BASE_SEED,
    )


def youden_threshold(y_true, proba):
    """Prog decyzyjny maksymalizujacy indeks Youdena J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def plot_calibration(y_true, proba_before, proba_after,
                     out_png="svm_calibration.png", n_bins=5, hist_bins=20):
    """Krzywa niezawodnosci (confidence vs accuracy) przed i po kalibracji
    plus histogram przewidzianych prawdopodobienstw w rozbiciu na klasy.
    Gora: os X to przewidziane prawdopodobienstwo, os Y to obserwowana
    czestosc nowotworu, idealna kalibracja to przekatna. Dol: ile probek
    kontroli i nowotworu trafia do kazdego kubelka prawdopodobienstwa,
    co pokazuje kubelki zawierajace wylacznie nowotwor."""
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
    ax.set_title('Krzywa niezawodności SVM RBF (przed vs po kalibracji)')
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


def confusion_3x2(y_pred, y_index, ds, le, out_png="svm_confusion_3x2.png"):
    """Niekwadratowa macierz pomylek: prawdziwe 3 klasy (zdrowi, choroby
    trzustki, nowotwor) wzgledem 2 klas przewidzianych (kontrola, nowotwor).
    Pacjenci z chorobami trzustki w treningu naleza do klasy kontrolnej,
    wiec ten widok pokazuje, ilu z nich model blednie wskazuje jako nowotwor."""
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


def main():
    # === panel genow ===
    if not os.path.exists(GENES_CSV):
        raise FileNotFoundError(f"Brak {GENES_CSV} - uruchom najpierw selekcje genow.")
    genes_df = pd.read_csv(GENES_CSV)
    selected_genes = genes_df.loc[genes_df['gene'] != '__intercept__', 'gene'].tolist()
    print(f"[INFO] wczytano {len(selected_genes)} genow z {GENES_CSV}")

    # === dane + split (taki sam jak w skrypcie 5) ===
    ds = load_dataset(data_path, meta_path, label_col="Group")
    ds.y = ds.y.replace({DISEASE: HEALTHY})
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    X_tr_raw, X_te_raw, X_va_raw, y_train, y_test, y_valid = ds.get_train_test_valid_split(
        ds.X, y_enc, test_size=TEST_SIZE, valid_size=VALID_SIZE
    )
    X_te_raw = pd.concat([X_te_raw, X_va_raw])
    y_test   = pd.concat([y_test, y_valid])

    X_tr_raw = X_tr_raw[selected_genes]
    X_te_raw = X_te_raw[selected_genes]

    print(f"Train: {len(X_tr_raw)}  cancer={int(y_train.sum())} ctrl={int((y_train==0).sum())}")
    print(f"Test:  {len(X_te_raw)}  cancer={int(y_test.sum())}  ctrl={int((y_test==0).sum())}")

    # === DEG preprocessing (taki sam jak w skrypcie 5) ===
    cov = build_covariates(ds.meta)
    deg_pipe = Pipeline([
        ('multi_resid', MultiCovariateResidualBootstrapTransformer(
            covariates=cov, labels=y_train,
            n_bootstrap=2, fdr_alpha=1, min_r2=0, cv_threshold_pct=1e9,
        )),
    ])
    X_tr_deg_df = deg_pipe.fit_transform(X_tr_raw, y_train)
    X_te_deg_df = deg_pipe.transform(X_te_raw)

    missing = [g for g in selected_genes if g not in X_tr_deg_df.columns]
    if missing:
        raise ValueError(f"Geny z CSV nie sa dostepne: {missing[:5]}...")

    scaler = StandardScaler()
    scaler.fit(X_tr_deg_df.values)
    X_tr_z = scaler.transform(X_tr_deg_df.values)
    X_te_z = scaler.transform(X_te_deg_df.values)
    y_tr_np = y_train.values
    y_te_np = y_test.values

    # === SVM RBF + kalibracja ===
    clf = CalibratedClassifierCV(make_svm(), method=CALIB_METHOD, cv=CALIB_CV)
    clf.fit(X_tr_z, y_tr_np)

    proba_tr = clf.predict_proba(X_tr_z)[:, 1]
    proba_te = clf.predict_proba(X_te_z)[:, 1]
    auc_te = roc_auc_score(y_te_np, proba_te)
    print(f"\n[SVM RBF skalibrowany] C={SVM_C}, gamma={SVM_GAMMA}, kalibracja={CALIB_METHOD}")
    print(f"Holdout test AUC: {auc_te:.4f}")

    # === krzywa niezawodnosci: przed vs po kalibracji ===
    # przed kalibracja: surowy SVM, decision_function przeskalowany do [0,1]
    # zakresem z treningu (SVM nie ma natywnego predict_proba bez probability=True)
    raw_svm = make_svm().fit(X_tr_z, y_tr_np)
    dec_tr = raw_svm.decision_function(X_tr_z)
    lo, hi = float(dec_tr.min()), float(dec_tr.max())
    proba_before = np.clip(
        (raw_svm.decision_function(X_te_z) - lo) / (hi - lo), 0.0, 1.0)
    plot_calibration(y_te_np, proba_before, proba_te)

    # === prog decyzyjny z indeksu Youdena (wyznaczony na train) ===
    thr = youden_threshold(y_tr_np, proba_tr)
    y_pred = (proba_te >= thr).astype(int)
    print(f"Prog Youdena (z train): {thr:.4f}")

    # === raport klasyfikacji ===
    print("\n=== classification_report (test) ===")
    print(classification_report(y_te_np, y_pred, target_names=le.classes_, digits=3))
    print("Macierz pomylek [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_te_np, y_pred))

    print("\n=== show_report (metadane probek FN/FP) ===")
    show_report(y_pred, y_test, ds, le)

    # === niekwadratowa macierz pomylek: 3 klasy prawdziwe x 2 przewidziane ===
    confusion_3x2(y_pred, y_test.index, ds, le)


if __name__ == '__main__':
    main()
