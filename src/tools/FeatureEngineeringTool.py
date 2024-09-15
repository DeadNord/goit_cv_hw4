import numpy as np
from scipy.stats import zscore


class FeatureEngineeringTool:
    """
    A class to perform feature engineering on a dataset.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset to be transformed.

    Methods
    -------
    remove_highly_correlated_feature(target, threshold=0.9, exclude_features=None):
        Removes features with high cross-correlation above the given threshold.

    remove_features(features):
        Removes the specified features from the dataset.

    add_features(feature_func_dict):
        Adds new features to the dataset based on the provided dictionary of feature functions.

    transform_features(transform_func_dict):
        Transforms features in the dataset based on the provided dictionary of transformation functions.

    drop_nan():
        Drops rows with any NaN values from the dataset.

    clean_outliers(columns):
        Removes outliers from the specified columns based on z-scores.

    add_count_feature(material_columns, new_feature_name='Count'):
        Adds a new feature that counts the number of non-zero materials in the specified columns.
    """

    def __init__(self, df):
        """
        Constructs all the necessary attributes for the FeatureEngineering object.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to be transformed.
        """
        self.df = df

    def remove_highly_correlated_feature(
        self, target, threshold=0.9, exclude_features=None
    ):
        """
        Removes features with high cross-correlation above the given threshold.

        Parameters
        ----------
        target : str
            The target column name for correlation reference.
        threshold : float, optional
            The correlation threshold to use for removing features (default is 0.9).
        exclude_features : list, optional
            List of features to exclude from dropping (default is None).
        """
        if exclude_features is None:
            exclude_features = []

        corr_matrix = self.df.corr().abs()  # Compute correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )  # Select upper triangle of correlation matrix
        to_drop = []  # List to track which features to drop
        target_corr = self.df.corr()[target].abs()  # Correlation with the target

        for column in upper.columns:
            if column in exclude_features:
                continue
            correlated_features = upper.index[upper[column] > threshold].tolist()
            if correlated_features:
                # Filter out features that are in the exclude list
                correlated_features = [
                    feat for feat in correlated_features if feat not in exclude_features
                ]
                if not correlated_features:
                    continue
                # Determine the feature with the lower correlation with the target
                to_remove = (
                    correlated_features[0]
                    if target_corr[correlated_features[0]] < target_corr[column]
                    else column
                )
                if to_remove not in to_drop:
                    to_drop.append(to_remove)

        self.df.drop(columns=to_drop, inplace=True)
        print(f"Removed features due to high correlation: {to_drop}")

    def remove_features(self, features):
        """
        Removes the specified features from the dataset.

        Parameters
        ----------
        features : list
            List of feature names to be removed.
        """
        if len(features) == 0:
            print(f"No features removed")
            return
        self.df.drop(columns=features, inplace=True)
        print(f"Removed features: {features}")

    def add_features(self, feature_func_dict):
        """
        Adds new features to the dataset based on the provided dictionary of feature functions.

        Parameters
        ----------
        feature_func_dict : dict
            Dictionary where keys are new feature names and values are functions to calculate the feature.
        """
        for feature_name, func in feature_func_dict.items():
            self.df[feature_name] = self.df.apply(func, axis=1)
        print(f"Added features: {list(feature_func_dict.keys())}")

    def transform_features(self, transform_func_dict):
        """
        Transforms features in the dataset based on the provided dictionary of transformation functions.

        Parameters
        ----------
        transform_func_dict : dict
            Dictionary where keys are existing feature names and values are functions to transform the feature.
        """
        for feature_name, func in transform_func_dict.items():
            self.df[feature_name] = self.df[feature_name].apply(func)
        print(f"Transformed features: {list(transform_func_dict.keys())}")

    def drop_nan(self):
        """
        Drops rows with any NaN values from the dataset.
        """
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        print(
            f"Dropped {initial_shape[0] - self.df.shape[0]} rows containing NaN values."
        )

    def clean_outliers(self, columns):
        """
        Removes outliers from the specified columns based on z-scores.

        Parameters
        ----------
        columns : list
            List of columns to clean outliers from.
        """
        z_scores = np.abs(
            zscore(self.df[columns])
        )  # Compute z-scores for specified columns
        filtered_entries = (z_scores < 3).all(
            axis=1
        )  # Filter out rows with any z-score >= 3
        self.df = self.df[filtered_entries]  # Update the dataframe
        print(f"Removed outliers from columns: {columns}")
        return self.df

    def add_count_feature(self, material_columns, new_feature_name="Count"):
        """
        Adds a new feature that counts the number of non-zero materials in the specified columns.

        Parameters
        ----------
        material_columns : list
            List of columns that represent the materials.
        new_feature_name : str, optional
            The name of the new feature column (default is 'Count').
        """
        self.df[new_feature_name] = self.df[material_columns].astype(bool).sum(axis=1)
        print(
            f"Added new feature '{new_feature_name}' with counts of non-zero materials."
        )
