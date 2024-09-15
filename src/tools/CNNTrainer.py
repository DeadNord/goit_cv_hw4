import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tqdm import tqdm


class CNNTrainer:
    """
    A class to train PyTorch CNN models with grid search for hyperparameter tuning.
    Supports only classification tasks.
    """

    def __init__(self, device="cpu"):
        """
        Initialize the trainer with a device.

        Parameters
        ----------
        device : str, optional
            The device to train the model on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.device = device
        self.best_estimators = {}
        self.best_params = {}
        self.best_scores = {}
        self.best_model_name = None
        self.best_model_score = float("-inf")

    def train(
        self,
        train_dataset,  # Transformed training dataset (e.g. ImageFolder)
        test_dataset,  # Transformed test dataset (e.g. ImageFolder)
        pipelines,  # Dictionary of model pipelines
        param_grids,  # Dictionary of hyperparameter grids for grid search
        scoring="accuracy",
        cv=5,
        verbose=0,
        n_jobs=-1,
        error_score=np.nan,
        use_progress_bar=True,
    ):
        """
        Train the PyTorch CNN models using grid search and cross-validation.

        Parameters
        ----------
        train_dataset : torchvision.datasets.ImageFolder
            The preprocessed training dataset (e.g. ImageFolder).
        test_dataset : torchvision.datasets.ImageFolder
            The preprocessed test dataset (e.g. ImageFolder).
        pipelines : dict
            A dictionary of model pipelines.
        param_grids : dict
            A dictionary of hyperparameter grids for grid search.
        scoring : str or None, optional
            Scoring metric for grid search. Default is 'accuracy'.
        cv : int, optional
            The number of cross-validation folds (default is 5).
        verbose : int, optional
            The verbosity level (default is 0).
        n_jobs : int, optional
            The number of jobs to run in parallel (default is -1).
        error_score : str, optional
            How to handle errors during fitting (default is "raise").
        use_progress_bar : bool, optional
            If True, a progress bar and callback are used (default is True).
        """
        print(f"Using scoring metric: {scoring}")
        print(f"Training on device: {self.device}")

        total_fits = sum(
            len(list(ParameterGrid(param_grids[model_name]))) * cv
            for model_name in pipelines
        )

        if use_progress_bar:
            pbar = tqdm(total=total_fits, desc="Total Progress")
        else:
            pbar = None

        for model_name, pipeline in pipelines.items():
            # Set the device for the model
            pipeline.set_params(classifier__device=self.device)

            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                n_jobs=n_jobs,
                error_score=error_score,
            )

            if use_progress_bar:
                # Callback function to update the progress bar after each fold
                def progress_bar_callback(train_loss, val_loss):
                    pbar.update(1)
                    pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss})

                # Pass the callback to fit_params for the classifier
                fit_params = {
                    "classifier__train_dataset": train_dataset,
                    "classifier__test_dataset": test_dataset,
                    "classifier__fold_callback": progress_bar_callback,
                }
            else:
                fit_params = {
                    "classifier__train_dataset": train_dataset,
                    "classifier__test_dataset": test_dataset,
                }

            grid_search.fit(train_dataset, test_dataset, **fit_params)

            self.best_estimators[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            self.best_scores[model_name] = -grid_search.best_score_

            if self.best_scores[model_name] > self.best_model_score:
                self.best_model_name = model_name
                self.best_model_score = self.best_scores[model_name]

        if use_progress_bar:
            pbar.close()

    def help(self):
        """
        Provides information on how to use the CNNTrainer class.
        """
        print("=== CNNTrainer Help ===")
        print(
            "This trainer is designed to support classification tasks using PyTorch CNN models."
        )
        print("\nUsage:")
        print("1. Initialize the CNNTrainer with the device ('cpu' or 'cuda').")
        print("   Example:")
        print("       trainer = CNNTrainer(device='cuda')")
        print(
            "\n2. Create a pipeline with a classifier, for example using PyTorchCNNClassifier."
        )
        print("   Example:")
        print(
            "       pipeline = Pipeline(steps=[('classifier', PyTorchCNNClassifier())])"
        )
        print("\n3. Define the parameter grid for hyperparameter search.")
        print("   Example:")
        print(
            '       param_grid = { "classifier__lr": [0.001, 0.01], "classifier__epochs": [10, 20] }'
        )
        print(
            "\n4. Call the `train` method with the training and test datasets, pipeline, and parameter grid."
        )
        print("   Example:")
        print(
            "       trainer.train(train_dataset, test_dataset, pipelines={'cnn': pipeline}, param_grids={'cnn': param_grid})"
        )
        print("\nParameters:")
        print("- train_dataset : Training data (torchvision.datasets.ImageFolder).")
        print("- test_dataset : Test data (torchvision.datasets.ImageFolder).")
        print(
            "- pipelines : Dictionary of model pipelines (e.g., {'cnn': Pipeline(steps=[('classifier', PyTorchCNNClassifier())])})."
        )
        print(
            "- param_grids : Dictionary of hyperparameter grids for GridSearchCV (e.g., {'cnn': {'classifier__lr': [0.001, 0.01]}})."
        )
        print("- scoring : Metric for evaluating models (default is 'accuracy').")
        print("- cv : Number of cross-validation folds (default is 5).")
        print("- n_jobs : Number of jobs to run in parallel (default is -1).")
        print(
            "- error_score : How to handle errors during fitting (default is 'raise')."
        )
        print(
            "- use_progress_bar : If True, a progress bar and callbacks will be used during training (default is True)."
        )
