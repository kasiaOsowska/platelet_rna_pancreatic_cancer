<br />
<div align="center">
  <h3 align="center">Pancreatic cancer data analysis</h3>
</div>

Detecting pancreatic cancer from the RNA profile of blood platelets (tumour-educated
platelets, a liquid biopsy), with the emphasis on confounder control, stable gene
selection and a clinically realistic control group.

Reference method: Ji et al. (2025), *Transcriptomic profiling of blood platelets identifies
a diagnostic signature for pancreatic cancer*, British Journal of Cancer 132:937-946,
[doi.org/10.1038/s41416-025-02980-z](https://doi.org/10.1038/s41416-025-02980-z).
Their two-gene signature PLA2Sig (SCN1B, MAGOHB) reached AUC 0.783 to 0.900 across cohorts.

## The question

Classifying pancreatic cancer from platelet RNA-seq has already been solved to a high AUC,
including by Ji et al. This project does not aim to repeat that result. It targets three
things those studies left open.

**1. Do confounders inflate the reported accuracy?**
Ji et al. ruled out sex but did not analyse age. Preliminary analysis here shows age is
associated with the expression of exactly the genes the classifiers rely on. If the age
distribution differs between the cancer and control groups, part of the measured AUC is
age, not cancer. Sequencing depth is a third candidate.

**2. Is the selected gene set reproducible?**
A single training run yields a single gene list. Which of those genes survive if the
training subsample changes?

**3. Does the control group reflect the clinic?**
Standard studies use healthy donors as the only controls. Here, patients with non-cancerous
pancreatic disease are placed in the non-cancer class. That forces the model to learn
cancer-specific expression rather than pancreas-disease-general expression, and it matches
the actual clinical question: a patient referred on suspicion of pancreatic cancer who
turns out to have a different pancreatic condition.

Concrete task: binary classification of a platelet RNA-seq sample as cancer or not-cancer.

## Approach

**Data.** A counts matrix of 42,630 genes by 2,558 samples with an accompanying metadata
sheet. 1,973 samples are dropped for missing metadata and one is dropped as internally
inconsistent (labelled as control or pancreatic disease while carrying stage IV), leaving
**583 samples**. Three metadata groups: asymptomatic controls, pancreatic diseases,
pancreatic cancer. The last two are merged into a binary target with pancreatic diseases
counted as non-cancer.

**Split.** Not a plain stratified split. `Dataset._get_strata` builds the strata from the
joint of label, tumour stage (I and II merged), sex and age tertile, so the training and
test sets match on all four at once. Strata with fewer than two members cannot be split and
are appended to train. A leakage assertion compares the index sets afterwards and raises if
they intersect. Result: **356 train** (81 cancer, 275 control) and **227 test** (43 cancer,
184 control).

**Gene reduction**, as a scikit-learn pipeline fitted on train only:

| step | genes remaining |
|---|---|
| input | 30,588 |
| `ConstantExpressionReductor` (drop zero-variance) | 30,588 |
| `AnovaFdrReductor` (ANOVA F test, Benjamini-Hochberg, alpha 0.05) | 4,006 |
| `MeanExpressionReductor` (drop the lowest-expressed percentile) | 3,805 |
| `CovariateResidualizer` | 3,805 |

**Confounder removal** is the methodological core and answers question 1.
`CovariateResidualizer` regresses each gene on age, sex and
log10 library size, then subtracts the fitted value, leaving the residual. Two details make
it more than a plain regression:

- It is **fitted on control samples only** (`labels == 0`). Fitting on everything would
  absorb part of the cancer signal into the covariate fit and remove it.
- A gene is residualised only if the covariate dependence is **stable under bootstrap**:
  1,000 bootstrap resamples, keep genes whose median R-squared is at least 0.05 and whose
  coefficient of variation across resamples is below 30 percent. 1,052 of 3,805 genes met
  this and were corrected; the rest are passed through untouched.

**Stable gene selection** answers question 2. Elastic Net logistic regression is tuned by
randomised search over C and l1_ratio, then refitted on 100 stratified subsamples of 60
percent of train. A gene is kept if it receives a non-zero coefficient in at least 80
percent of those runs (Meinshausen-Buhlmann stability selection).

**Panel construction and models.** Greedy forward selection over the stable genes, then four
classifiers on the resulting panel.

## Results

Holdout test set, 227 samples, seed 2137.

**Stability selection** (stage 2). Elastic Net tuning: 50-fold CV AUC 0.8467 at C = 1.64,
l1_ratio = 0.79 (result of the randomised search, now fixed in `tune_enet`). How many genes
survive at each stability threshold:

| selection frequency | genes |
|---|---|
| >= 50 % | 229 |
| >= 70 % | 87 |
| >= 80 % | **54** |
| >= 90 % | 28 |
| = 100 % | 0 |

54 of 3,805 genes (1.4 percent) clear the 80 percent threshold: 24 up-regulated, 30
down-regulated in cancer, selection frequency 0.80 to 0.99. No gene is selected in every
single run, which is itself the answer to question 2: the gene list from one training run is
not reproducible without this step.

**Forward selection** (stage 3) builds the panel from the 54 stable genes. Each step adds
the gene that most raises CV AUC on train (10-fold, logistic regression):

| genes | CV AUC (train) | test AUC |
|---|---|---|
| 1 (PLD4) | 0.754 | 0.658 |
| 3 (+ ITGB3BP, KALRN) | 0.831 | 0.768 |
| 6 (+ PRELID2, TMEM63B, SYT17) | 0.884 | 0.750 |
| 9 (+ H2BC15, FABP4, FUT8) | 0.915 | 0.793 |
| **12 (+ ACVR1, MAOB, FUCA2)** | **0.942** | **0.816** |

The resulting 12-gene panel (`3_forward_selection/forward_selection_genes.csv`, gene
symbols in `utilz/constans.py` as `BIOMARKER_PANEL`):
PLD4, ITGB3BP, KALRN, PRELID2, TMEM63B, SYT17, H2BC15, FABP4, FUT8, ACVR1, MAOB, FUCA2.

**Models on the panel** (stage 4). Panel genes are corrected with `CovariateResidualizer`
(same criteria as above), hyperparameters are tuned by 30-fold stratified CV on train:

| model | CV AUC (train) | test AUC | sensitivity | specificity | accuracy |
|---|---|---|---|---|---|
| Logistic regression, Elastic Net | 0.944 | 0.825 | 0.744 | 0.777 | 0.771 |
| SVM RBF | 0.942 | 0.815 | 0.698 | 0.761 | 0.749 |
| XGBoost | 0.885 | 0.817 | 0.326 | 0.935 | 0.819 |
| **Stacking (LogReg + SVM + XGB)** | 0.932 | **0.830** | 0.605 | 0.848 | 0.802 |

Sensitivity and specificity are taken at the Youden-optimal threshold computed on train.
XGBoost's threshold lands at 0.704, which buys accuracy by predicting the majority class:
its accuracy is the highest in the table and its sensitivity the lowest by a wide margin.
Accuracy is the wrong summary at this class balance.

**The three-group breakdown** (stage 4, `4_predictive_models/alt_models_confusion_3x2.csv`)
answers question 3. Share of each true group predicted as cancer:

| true group | LogReg ENet | SVM RBF | XGBoost | Stacking |
|---|---|---|---|---|
| asymptomatic controls (160) | 31 (19 %) | 33 (21 %) | 7 (4 %) | 19 (12 %) |
| pancreatic diseases (24) | 10 (42 %) | 11 (46 %) | 5 (21 %) | 9 (38 %) |
| pancreatic cancer (43) | 32 (74 %) | 30 (70 %) | 14 (33 %) | 26 (60 %) |

For every model, pancreatic disease patients are called cancer at two to five times the rate
of healthy donors. Part of the signal the models use is pancreatic disease in general, not
cancer specifically. This is the cost of the harder control group, and it is only visible
because those patients were kept as a separate reported stratum.

## What this resolves

- **Confounders were worth removing.** 1,052 of 3,805 candidate genes have a stable,
  bootstrap-confirmed dependence on age, sex or library depth. Reporting a classifier on
  this data without residualising them would mix those effects into the AUC.
- **One training run does not give a reproducible gene list.** No gene out of 3,805 is
  selected in all 100 subsamples, and only 54 reach 80 percent.
- **The harder control group costs measurable specificity** on exactly the patients it was
  added for. That trade is the point of the design, not a defect, but it should be reported
  as the three-group table rather than hidden inside a binary confusion matrix.
- **Accuracy is misleading here.** XGBoost has the best accuracy and the worst sensitivity in
  the same row.

## Layout

```
utilz/constans.py                 labels, known marker genes, the final BIOMARKER_PANEL
utilz/Dataset.py                  loading, multi-criteria stratified splits, leakage assert
utilz/preprocessing_utilz.py      gene reduction transformers (ANOVA FDR, mean, log2FC,
                                  Mann-Whitney, within-group variance)
utilz/multi_residual_bootstrap.py CovariateResidualizer: covariate residualisation with
                                  bootstrap stability
utilz/helpers.py                  reports and figures

1_data_preparation.ipynb          data overview, split balance, PCA, residualisation tests
2_stable_selection_with_enet.py   gene filtering, covariate correction, ENet stability selection
3_forward_selection.py            greedy forward selection over the stable genes
4_predictive_models.py            LogReg / SVM RBF / XGBoost / stacking on the panel
5_biomarker_boxplots.py           panel expression boxplots by group
PLA2Sig/                          replication of the Ji et al. selection method
```

Each script writes its outputs into a directory named after itself. Stages 3 to 5 consume
the CSV produced by the preceding stages, so they must be run in order.

## Running

```bash
pip install -r requirements.txt
```

```bash
python 2_stable_selection_with_enet.py
```

```bash
python 3_forward_selection.py
```

```bash
python 4_predictive_models.py
```

```bash
python 5_biomarker_boxplots.py
```

Scripts expect `../data/counts_pancreatic.csv` and `../data/samples_pancreatic.xlsx`
relative to the project directory. The data is not part of this repository.

## Known limitations

- **A single holdout split.** Everything is measured on one train/test partition at seed 2137. There is no external cohort and no repeated-split estimate, so the test AUCs carry no
confidence interval.
- **Test AUC is printed during selection.** Stage 3 selects on CV AUC over train, which is correct, but it also displays the holdout AUC at every step. Choosing the panel size by looking at that column would make the reported test AUC optimistic.
- **Forward selection runs on uncorrected expression.** Stage 3 standardises the raw values of the stable genes, while stage 4 trains the models on covariate-corrected values of the same genes.
- **43 cancer cases in the test set.** One sample is 2.3 percentage points of sensitivity. Differences of a few points between models are not separable.
- **Pancreatic disease patients are a small stratum**, 24 samples in test. The 38 to 46 percent false-positive rate for that group (logistic regression, SVM, stacking) rests on 9 to 11 patients.
