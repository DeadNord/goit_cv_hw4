from sklearn.base import BaseEstimator, TransformerMixin


class ValueRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_cols, target_col):
        self.groupby_cols = groupby_cols
        self.target_col = target_col
        self.grouped_means = None

    def fit(self, X, y=None):
        self.grouped_means = X.groupby(self.groupby_cols)[self.target_col].mean()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[f"{self.target_col}_Ratio"] = X.groupby(self.groupby_cols)[
            self.target_col
        ].transform(lambda x: x / self.grouped_means[x.name])
        return X
