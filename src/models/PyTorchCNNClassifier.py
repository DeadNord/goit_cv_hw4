import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class PyTorchCNNClassifier:
    """
    CNN Classifier with customizable architecture and hyperparameters.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=6,
        conv_layers=[
            (32, 3, 1, 1),
            (64, 3, 1, 1),
            (128, 3, 1, 1),
        ],
        hidden_sizes=[256, 128],
        activation_fn="ReLU",
        pool_fn="MaxPool2d",
        pool_kernel_size=2,
        pool_stride=2,
        pool_padding=0,
        lr=0.001,
        batch_size=32,
        epochs=100,
        device="cpu",
        optimizer_type="adam",
        criterion_type="cross_entropy",
        dropout_rate=0.5,
        random_state=None,
        fold_callback=None,
        epochs_logger=True,
    ):
        """
        Initialize the CNN classifier with the provided architecture and hyperparameters.
        """
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.hidden_sizes = hidden_sizes
        self.activation_fn = self._get_activation_fn(activation_fn)
        self.pool_fn = self._get_pool_fn(pool_fn)
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.dropout_rate = dropout_rate
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []  # Add history for train accuracy
        self.val_accuracy_history = []  # Add history for validation accuracy
        self.random_state = random_state
        self.fold_callback = fold_callback
        self.epochs_logger = epochs_logger

        if random_state is not None:
            self._set_random_state(random_state)

    def _get_activation_fn(self, activation_name):
        """
        Maps a string to an activation function.
        """
        activations = {
            "ReLU": nn.ReLU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
        }
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        return activations[activation_name]

    def _get_pool_fn(self, pool_name):
        """
        Maps a string to a pooling function.
        """
        pools = {"MaxPool2d": nn.MaxPool2d, "AvgPool2d": nn.AvgPool2d}
        if pool_name not in pools:
            raise ValueError(f"Unsupported pooling function: {pool_name}")
        return pools[pool_name]

    def _set_random_state(self, random_state):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(random_state)

    def _initialize_model(self):
        """
        Initialize the CNN model with convolutional, activation, pooling, and fully connected layers.
        """
        layers = []
        input_channels = self.input_channels

        for out_channels, kernel_size, stride, padding in self.conv_layers:
            layers.append(
                nn.Conv2d(
                    input_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(self.activation_fn())
            layers.append(
                self.pool_fn(
                    kernel_size=self.pool_kernel_size,
                    stride=self.pool_stride,
                    padding=self.pool_padding,
                )
            )
            input_channels = out_channels

        layers.append(nn.Flatten())

        flattened_size = self._calculate_flattened_size()

        input_dim = flattened_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation_fn())
            layers.append(nn.Dropout(p=self.dropout_rate))
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, self.num_classes))

        self.model = nn.Sequential(*layers)
        self.model.to(self.device)

        # Optimizer setup
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Loss function setup
        if self.criterion_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.criterion_type}")

    def _calculate_flattened_size(self, input_shape=(3, 224, 224)):
        """
        Calculate the flattened size of the input after passing through the convolutional layers.
        This method takes into account the convolutional layers, pooling layers, and padding.
        """
        channels, height, width = input_shape

        for out_channels, kernel_size, stride, padding in self.conv_layers:
            height = (height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            width = (width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

            pool_kernel_size = self.pool_kernel_size
            pool_stride = self.pool_stride
            pool_padding = self.pool_padding

            height = (
                height + 2 * pool_padding - (pool_kernel_size - 1) - 1
            ) // pool_stride + 1
            width = (
                width + 2 * pool_padding - (pool_kernel_size - 1) - 1
            ) // pool_stride + 1

        flattened_size = out_channels * height * width
        return flattened_size

    def set_params(self, **params):
        """
        Set parameters for the classifier and reinitialize the model.
        """
        for param, value in params.items():
            if param == "activation_fn":
                value = self._get_activation_fn(value)
            elif param == "pool_fn":
                value = self._get_pool_fn(value)
            setattr(self, param, value)

        self._initialize_model()

    def fit(self, train_dataset, test_dataset=None):
        """
        Train the CNN model on the data. Handles both training and validation logic.
        """
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = (
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            if test_dataset
            else None
        )

        self._initialize_model()
        self.model.train()

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            correct_preds = 0
            total_samples = 0

            if self.epochs_logger:
                print(f"\nEpoch {epoch+1}/{self.epochs}")

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == targets).sum().item()
                total_samples += targets.size(0)

            train_loss = running_train_loss / len(train_loader)
            train_accuracy = correct_preds / total_samples

            # Log train accuracy history
            self.train_accuracy_history.append(train_accuracy)

            if self.epochs_logger:
                print(
                    f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}"
                )

            self.train_loss_history.append(train_loss)

            if self.fold_callback:
                self.fold_callback(train_loss, epoch + 1)

            if val_loader is not None:
                self._evaluate(val_loader)

        return train_accuracy

    def _evaluate(self, dataset):
        """
        Evaluate the model on the dataset, dividing it into batches.
        """
        if isinstance(dataset, DataLoader):
            data_loader = dataset
        else:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        running_val_loss = 0.0
        correct_preds = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == targets).sum().item()
                total_samples += targets.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(data_loader)
        val_accuracy = correct_preds / total_samples
        self.val_loss_history.append(val_loss)
        self.val_accuracy_history.append(val_accuracy)

        if self.epochs_logger:
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        self.model.train()

        return all_preds, all_targets, val_loss, val_accuracy

    def predict(self, dataset):
        """
        Predict labels for the dataset without targets.
        """
        if isinstance(dataset, DataLoader):
            data_loader = dataset
        else:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())

        return all_preds
