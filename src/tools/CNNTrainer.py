from tqdm import tqdm
from sklearn.model_selection import ParameterGrid


class CNNTrainer:
    """
    A class to train PyTorch CNN models with manual hyperparameter tuning.
    Supports only classification tasks.
    """

    def __init__(self, device="cpu"):
        """
        Initialize the trainer with a device.
        """
        self.device = device
        self.best_estimators = {}
        self.best_params = {}
        self.best_scores = {}
        self.best_model_name = None
        self.best_model_score = float("-inf")

    def train(
        self,
        train_dataset,
        test_dataset,
        pipelines,
        param_grids,
        scoring="accuracy",
        verbose=0,
        use_progress_bar=True,
    ):
        """
        Train the PyTorch CNN models using manual hyperparameter tuning.
        """
        print(f"Training on device: {self.device}")

        for model_name, pipeline in pipelines.items():
            param_grid = param_grids[model_name]

            # Generate all combinations of hyperparameters
            param_combinations = list(ParameterGrid(param_grid))
            total_combinations = len(param_combinations)

            if use_progress_bar:
                pbar = tqdm(total=total_combinations, desc="Total Progress")
            else:
                pbar = None

            for params in param_combinations:
                # Set parameters for the pipeline
                pipeline.set_params(**params)

                # Extract the classifier to manually set the device and train
                classifier = pipeline.named_steps["classifier"]
                classifier.device = self.device  # Set the device for the model

                if verbose:
                    print(f"Training with parameters: {params}")

                # Train the model, which internally handles the validation logic (e.g. loss plots)
                accuracy = classifier.fit(train_dataset, test_dataset)

                if verbose:
                    print(f"Validation Accuracy: {accuracy}")

                # Store the best model and its parameters
                if accuracy > self.best_model_score:
                    self.best_model_name = model_name
                    self.best_model_score = accuracy
                    self.best_estimators[model_name] = classifier
                    self.best_params[model_name] = params

                if use_progress_bar:
                    pbar.update(1)

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
        print("\n3. Define the parameter grid for hyperparameter tuning.")
        print("   Example:")
        print(
            '       param_grid = { "classifier__lr": [0.001, 0.01], "classifier__epochs": [10, 20] }'
        )
        print(
            "\n4. Call the `train` method with the training and test datasets, pipeline, and parameter grid."
        )
        print("   Example:")
        print(
            "       trainer.train(train_dataset, test_dataset, pipelines={'pytorch_classifier': pipeline}, param_grids={'pytorch_classifier': param_grid})"
        )
