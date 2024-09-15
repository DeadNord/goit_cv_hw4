from sklearn.base import BaseEstimator, TransformerMixin


class AddCountFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, material_columns, new_feature_name="Count"):
        self.material_columns = material_columns
        self.new_feature_name = new_feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.new_feature_name] = X[self.material_columns].astype(bool).sum(axis=1)
        return X
