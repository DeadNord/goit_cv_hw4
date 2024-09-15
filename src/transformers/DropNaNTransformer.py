from sklearn.base import BaseEstimator, TransformerMixin


class DropNaNTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.dropna(inplace=True)
        return X
