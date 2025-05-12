from sklearn.model_selection._split import _BaseKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class StratifiedGroupKFold(_BaseKFold):
    """
    Stratified K-Folds cross-validator with non-overlapping groups.

    Each fold maintains the class distribution (stratification) at the group level.
    Groups are not split across folds.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups):
        X, y, groups = np.array(X), np.array(y), np.array(groups)

        # Step 1: group -> class mapping
        df = pd.DataFrame({"group": groups, "label": y})
        group_labels = df.groupby("group")["label"].first()
        unique_groups = group_labels.index.values
        unique_labels = group_labels.values

        # Step 2: stratified split on group-level labels
        skf = StratifiedKFold(
            self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        for group_train_idx, group_test_idx in skf.split(unique_groups, unique_labels):
            train_groups = unique_groups[group_train_idx]
            test_groups = unique_groups[group_test_idx]

            # Step 3: mask original samples by group
            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]
            yield train_idx, test_idx
