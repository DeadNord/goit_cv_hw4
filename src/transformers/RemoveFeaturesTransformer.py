from sklearn.base import BaseEstimator, TransformerMixin


class RemoveFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if len(self.features) > 0:
            X.drop(columns=self.features, inplace=True)
        return X
