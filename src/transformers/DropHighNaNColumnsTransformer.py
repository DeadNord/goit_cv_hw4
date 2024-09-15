from sklearn.base import BaseEstimator, TransformerMixin


class DropHighNaNColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        # Рассчитать долю пропусков в каждом столбце
        missing_ratios = X.isnull().mean()
        # Определить столбцы для удаления
        self.columns_to_drop = missing_ratios[
            missing_ratios > self.threshold
        ].index.tolist()
        return self

    def transform(self, X, y=None):
        # Удалить выбранные столбцы
        return X.drop(columns=self.columns_to_drop)
