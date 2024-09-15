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
        self.best_scores = {}  # Dictionary to store the best scores
        self.best_model_name = None
        self.best_model_score = float("-inf")

    def train(
        self,
        train_dataset,
        test_dataset,
        models,
        param_grids,
        scoring="accuracy",
        verbose=0,
        use_progress_bar=True,
    ):
        """
        Train the PyTorch CNN models using manual hyperparameter tuning.
        """
        print(f"Training on device: {self.device}")

        for model_name, model in models.items():
            param_grid = param_grids[model_name]

            # Generate all combinations of hyperparameters
            param_combinations = list(ParameterGrid(param_grid))
            total_combinations = len(param_combinations)

            # Получаем количество эпох из параметров модели
            epochs = model.epochs

            # Инициализация прогресс-бара для всех комбинаций гиперпараметров и эпох
            if use_progress_bar:
                pbar = tqdm(
                    total=epochs * total_combinations, desc=f"Training {model_name}"
                )

            for params in param_combinations:
                # Set parameters for the model
                print(f"\nTraining {model_name} with parameters: {params}")
                model.set_params(**params)  # Устанавливаем параметры модели

                # Устанавливаем устройство для обучения
                model.device = self.device

                def fold_callback(loss, epoch):
                    """
                    Колбек для обновления прогресса по эпохам в tqdm.
                    """
                    if use_progress_bar and pbar is not None:
                        pbar.update(1)

                # Устанавливаем колбек
                model.fold_callback = fold_callback

                # Лог настроек модели
                if verbose:
                    print(f"Training with parameters: {params}")

                # Train the model and return accuracy
                accuracy = model.fit(train_dataset, test_dataset)

                if verbose:
                    print(f"Validation Accuracy for {model_name}: {accuracy}")

                # Сохранение результата для каждой модели
                self.best_scores[model_name] = accuracy

                # Store the best model and its parameters
                if accuracy > self.best_model_score:
                    self.best_model_name = model_name
                    self.best_model_score = accuracy
                    self.best_estimators[model_name] = model
                    self.best_params[model_name] = params

            # Закрытие прогресс-бара после завершения
            if use_progress_bar and pbar is not None:
                pbar.close()

        print(
            f"\nBest Model: {self.best_model_name} with score: {self.best_model_score}"
        )

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
            "\n2. Create model objects and define the parameter grid for hyperparameter tuning."
        )
        print("   Example:")
        print('       param_grid = { "lr": [0.001, 0.01], "epochs": [10, 20] }')
        print(
            "\n3. Call the `train` method with the training and test datasets, models, and parameter grid."
        )
        print("   Example:")
        print(
            "       trainer.train(train_dataset, test_dataset, models={'cnn_model': model}, param_grids={'cnn_model': param_grid})"
        )
