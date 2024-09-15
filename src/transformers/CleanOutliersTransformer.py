from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.stats import zscore


class CleanOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, method="zscore", threshold=3.0):
        """
        Parameters
        ----------
        columns : list
            List of columns to clean outliers from.
        method : str, optional
            Method to use for outlier detection ('zscore' or 'iqr', default is 'zscore').
        threshold : float, optional
            Threshold for outlier detection. For 'zscore', it is the z-score value (default is 3.0).
            For 'iqr', it is the multiplier for the IQR (default is 1.5).
        """
        self.columns = columns
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        if self.method == "zscore":
            z_scores = np.abs(zscore(X[self.columns]))
            filtered_entries = (z_scores < self.threshold).all(axis=1)
        elif self.method == "iqr":
            Q1 = X[self.columns].quantile(0.25)
            Q3 = X[self.columns].quantile(0.75)
            IQR = Q3 - Q1
            filtered_entries = ~(
                (X[self.columns] < (Q1 - self.threshold * IQR))
                | (X[self.columns] > (Q3 + self.threshold * IQR))
            ).any(axis=1)
        else:
            raise ValueError("Method must be 'zscore' or 'iqr'")

        X = X[filtered_entries]
        return X
