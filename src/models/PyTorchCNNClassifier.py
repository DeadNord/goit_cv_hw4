import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin


class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    """
    CNN Classifier with customizable architecture and hyperparameters.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=6,
        conv_layers=[(32, 3), (64, 3), (128, 3)],
        hidden_sizes=[256, 128],
        lr=0.001,
        batch_size=32,  # Batch size handled inside the model
        epochs=100,
        device="cpu",
        optimizer_type="adam",
        criterion_type="cross_entropy",
        dropout_rate=0.5,
        epochs_logger=True,
        random_state=None,
        fold_callback=None,  # Added callback
    ):
        """
        Initialize the CNN classifier with the provided architecture and hyperparameters.
        """
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.hidden_sizes = hidden_sizes
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
        self.epochs_logger = epochs_logger
        self.random_state = random_state
        self.fold_callback = fold_callback  # Save callback

        if random_state is not None:
            self._set_random_state(random_state)

    def _set_random_state(self, random_state):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(random_state)

    def _initialize_model(self):
        """
        Initialize the CNN model with convolutional, activation, pooling, and fully connected layers.
        """
        layers = []
        input_channels = self.input_channels

        # Add convolutional layers
        for out_channels, kernel_size in self.conv_layers:
            layers.append(
                nn.Conv2d(
                    input_channels, out_channels, kernel_size=kernel_size, padding=1
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = out_channels

        layers.append(nn.Flatten())

        # Calculate the flattened size for the fully connected layer
        flattened_size = self._calculate_flattened_size()

        # Add fully connected layers
        input_dim = flattened_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            input_dim = hidden_size

        # Add output layer
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

    def _calculate_flattened_size(self):
        """
        Calculate the flattened size of the input after passing through the convolutional layers.
        """
        size = 224  # Assuming the input image size is 224x224
        for _, kernel_size in self.conv_layers:
            size = size // 2  # Each MaxPool layer halves the size

        flattened_size = size * size * self.conv_layers[-1][0]
        return flattened_size

    def fit(self, train_dataset, test_dataset=None):
        """
        Train the CNN model on the data.

        Parameters
        ----------
        train_dataset : Dataset
            The training dataset (ImageFolder).
        test_dataset : Dataset, optional
            The test dataset (ImageFolder).
        """
        # Initialize DataLoader inside the model
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
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()

            # Calculate and save training loss
            train_loss = running_train_loss / len(train_loader)
            self.train_loss_history.append(train_loss)

            if self.epochs_logger & ((epoch + 1) % 10 == 0):
                print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss}")

            # Call fold callback after each epoch
            if self.fold_callback is not None:
                self.fold_callback(train_loss, None)

            # Validation phase (if validation data is provided)
            if val_loader is not None:
                self._evaluate(val_loader)

    def _evaluate(self, val_loader):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        running_val_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == targets).sum().item()
                total_samples += targets.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_samples
        self.val_loss_history.append(val_loss)

        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        self.model.train()

    def predict(self, test_loader):
        """
        Make predictions on new data.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())

        return predictions
