from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class RemoveHighlyCorrelatedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target, threshold=0.9, exclude_features=None):
        self.target = target
        self.threshold = threshold
        self.exclude_features = exclude_features if exclude_features is not None else []

    def fit(self, X, y=None):
        self.to_drop = []
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        target_corr = X.corr()[self.target].abs()

        for column in upper.columns:
            if column in self.exclude_features:
                continue
            correlated_features = upper.index[upper[column] > self.threshold].tolist()
            if correlated_features:
                correlated_features = [
                    feat
                    for feat in correlated_features
                    if feat not in self.exclude_features
                ]
                if not correlated_features:
                    continue
                to_remove = (
                    correlated_features[0]
                    if target_corr[correlated_features[0]] < target_corr[column]
                    else column
                )
                if to_remove not in self.to_drop:
                    self.to_drop.append(to_remove)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.drop(columns=self.to_drop, inplace=True)
        return X
