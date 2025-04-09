from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CorrelationFeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        # Compute absolute correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        corr_matrix = np.abs(corr_matrix)

        # Upper triangle mask, excluding diagonal
        upper = np.triu(corr_matrix, k=1)

        # Find columns to drop
        to_drop = set()
        for i in range(upper.shape[1]):
            if any(upper[:i, i] > self.threshold):
                to_drop.add(i)

        # Save indices of columns to keep
        self.cols_to_keep_ = [i for i in range(X.shape[1]) if i not in to_drop]
        return self

    def transform(self, X):
        return X[:, self.cols_to_keep_]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
