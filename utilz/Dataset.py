"""Dataset loading and leakage-free, multi-criteria stratified splitting.

load_dataset reads the count matrix and the sample metadata, keeps only samples present
in both, and drops samples whose group label contradicts their tumour stage. The
Dataset class builds strata by combining the class label, tumour stage (I and II
merged, missing coded as none), sex and an age tertile; strata with fewer than two
samples cannot be stratified and their samples are always assigned to the training set.
get_train_test_valid_split performs a stratified train/rest split followed by a
stratified test/validation split and asserts that no sample index occurs in more than
one split. get_stratified_kfold returns fold index pairs stratified on the same
composite strata, with the non-stratifiable remainder added to the training part of
every fold.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from utilz.constans import CANCER, HEALTHY, DISEASE


class Dataset:
    def __init__(self, X, meta, y=None):
        self.X = X
        self.meta = meta
        self.y = y
        self.age = self.meta['Age']
        self.sex = self.meta['Sex']
        self.institution = meta['RealLocation']
        self.stage = None

    def _get_strata(self, X, y):
        meta = self.meta.loc[X.index]

        df = pd.DataFrame(index=X.index)
        df["y"] = y
        df["sex"] = meta["Sex"]
        df["age_group"] = (
            pd.qcut(meta["Age"], q=3, labels=["young", "mid", "old"], duplicates="drop")
            .astype(object)
            .fillna("unknown")
        )
        stage = meta["Stage"].replace("NA", "none").fillna("none")
        stage = stage.replace({"I": "I_II", "II": "I_II"})
        self.stage = stage
        df["stage"] = stage

        X = X.loc[df.index]
        y = y.loc[df.index]

        strata = df[["y", "stage", "sex", "age_group"]].astype(str).agg("_".join, axis=1)

        counts = strata.value_counts()
        valid = counts[counts >= 2].index
        mask = strata.isin(valid)

        X_stratifiable = X[mask]
        y_stratifiable = y[mask]
        strata_stratifiable = strata[mask]

        X_remainder = X[~mask]
        y_remainder = y[~mask]

        return X_stratifiable, y_stratifiable, strata_stratifiable, X_remainder, y_remainder

    def get_train_test_valid_split(self, X, y, test_size=0.2, valid_size=0.2, random_state=2137, return_valid=True):
        X_strat, y_strat, strata, X_rem, y_rem = self._get_strata(X, y)

        X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
            X_strat, y_strat, strata,
            test_size=(test_size + valid_size),
            random_state=random_state,
            stratify=strata
        )

        if len(X_rem) > 0:
            print(f"[INFO] {len(X_rem)} samples with unique strata added to train set")
            X_train = pd.concat([X_train, X_rem])
            y_train = pd.concat([y_train, y_rem])

        if not return_valid:
            return X_train, X_temp, y_train, y_temp

        relative_valid_size = valid_size / (test_size + valid_size)

        X_temp_strat, y_temp_strat, strata_temp, X_rem2, y_rem2 = self._get_strata(X_temp, y_temp)

        X_test, X_valid, y_test, y_valid = train_test_split(
            X_temp_strat, y_temp_strat,
            test_size=relative_valid_size,
            random_state=random_state,
            stratify=strata_temp
        )

        if len(X_rem2) > 0:
            print(f"[INFO] {len(X_rem2)} samples with unique strata (2nd split) added to train set")
            X_train = pd.concat([X_train, X_rem2])
            y_train = pd.concat([y_train, y_rem2])

        _assert_no_leakage(
            X_train.index, X_test.index, X_valid.index,
            names=["Train", "Test", "Valid"]
        )

        return X_train, X_test, X_valid, y_train, y_test, y_valid


    def get_stratified_kfold(self, X, y, n_splits=5, random_state=2137):
        X_strat, y_strat, strata, X_rem, y_rem = self._get_strata(X, y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(skf.split(X_strat, strata))
        global_idx = X.index.get_indexer(X_strat.index)
        rem_idx = X.index.get_indexer(X_rem.index) if len(X_rem) > 0 else np.array([])

        all_folds = []
        for train_strat, test_strat in folds:
            train_global = global_idx[train_strat]
            test_global = global_idx[test_strat]
            train_global = np.concatenate([train_global, rem_idx])
            train_global = np.sort(train_global)
            test_global = np.sort(test_global)

            all_folds.append((train_global, test_global))

        print(f"[INFO] Generated {len(all_folds)} folds. Remainder ({len(rem_idx)}) "
              f"added to training set in each fold.")
        return all_folds


def load_dataset(path_csv, path_xlsx, label_col=None, separate_stage_iv = False):
    df_features = pd.read_csv(path_csv, sep=";", decimal=",", index_col=0)
    df_features = df_features.T
    df_metadata = pd.read_excel(path_xlsx, index_col=0)
    intersect = df_features.index.intersection(df_metadata.index)

    if len(df_features.index) != len(intersect):
        print(f"[INFO] skipped {len(df_features.index) - len(intersect)} probs due to missing metadata")

    df_features = df_features.loc[intersect].sort_index()
    meta = df_metadata.loc[intersect].sort_index()

    y = meta[label_col] if label_col is not None else meta["Group"]

    if separate_stage_iv:
        y = y.mask((y == CANCER) & (meta["Stage"] == "IV"), "cancer_IV")

    mask = ((y == HEALTHY) | (y == DISEASE)) & (meta["Stage"] == "IV")
    print("Dropping inconsistent sample:")
    print(meta.loc[mask])

    df_features = df_features.loc[~mask]
    meta = meta.loc[~mask]
    y = y.loc[~mask]

    return Dataset(X=df_features, meta=meta, y=y)

def _assert_no_leakage(*splits: pd.Index, names: list[str] = None) -> None:
    if names is None:
        names = [f"Split_{i}" for i in range(len(splits))]

    ok = True
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = splits[i].intersection(splits[j])
            if len(overlap) > 0:
                print(f"LEAKAGE: {names[i]} ∩ {names[j]} = {len(overlap)}")
                print(f"   Indexes: {overlap.tolist()}")
                ok = False

    if ok:
        print("\n[ASSERTION PASSED] No leakage detected between splits.")
    else:
        raise AssertionError("LEAKAGE")
