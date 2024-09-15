import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class CNNEvaluator:
    """
    A class to evaluate and display PyTorch CNN model performance results for classification tasks.

    Methods
    -------
    display_results(test_dataset, best_models, best_params, best_scores, best_model_name):
        Displays the evaluation metrics for the given model using the test dataset.
    predict_on_val(val_dataset, best_models, best_model_name):
        Runs predictions on an unlabeled validation dataset.
    plot_loss_history(best_models, best_model_name):
        Plots the training and validation loss history.
    visualize_pipeline(model_name, best_models):
        Visualizes the architecture of the best CNN model.
    """

    def display_results(
        self,
        test_dataset,
        best_models,
        best_params,
        best_scores,
        best_model_name,
    ):
        """
        Displays the evaluation metrics for the best models and their parameters using the test dataset.
        """
        test_loader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )  # Process the entire dataset at once
        results = []

        for model_name, cnn_model in best_models.items():
            device = cnn_model.device
            cnn_model.model.eval()
            correct_preds = 0
            total_samples = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = cnn_model.model(inputs)

                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                    correct_preds += (preds == targets).sum().item()
                    total_samples += targets.size(0)

            accuracy = correct_preds / total_samples
            balanced_acc = balanced_accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average="weighted")
            precision = precision_score(all_targets, all_preds, average="weighted")
            recall = recall_score(all_targets, all_preds, average="weighted")

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Balanced Accuracy": balanced_acc,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall,
                }
            )

        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        param_df = (
            pd.DataFrame(best_params).T.reset_index().rename(columns={"index": "Model"})
        )

        best_model_df = pd.DataFrame(
            {
                "Overall Best Model": [best_model_name],
                "Score (based on cross-validation score)": [
                    best_scores[best_model_name]
                ],
            }
        )

        print("Evaluation Metrics for Test Set:")
        display(results_df)

        print("\nBest Parameters for Each Model (found during hyperparameter tuning):")
        display(param_df)

        print("\nOverall Best Model and Score (based on cross-validation score):")
        display(best_model_df)

        if help_text:
            print("\nMetric Explanations for Classification:")
            print(
                "Accuracy: The ratio of correctly predicted instances to the total instances."
            )
            print("Balanced Accuracy: The average of recall obtained on each class.")
            print("F1 Score: Harmonic mean of precision and recall.")
            print(
                "Precision: Ratio of correctly predicted positive observations to all positive predictions."
            )
            print(
                "Recall: Ratio of correctly predicted positive observations to all actual positives."
            )

    def predict_on_val(self, val_dataset, best_models, best_model_name):
        """
        Runs predictions on an unlabeled validation dataset (without ground truth labels).
        Returns the predicted classes.
        """
        val_loader = DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False
        )  # Process the entire dataset at once
        best_model = best_models[best_model_name]
        device = best_model.device
        best_model.model.eval()

        all_preds = []

        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                outputs = best_model.model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())

        print(f"Predictions for validation dataset: {all_preds}")
        return all_preds

    def plot_loss_history(self, best_models, best_model_name):
        """
        Plots the training and validation loss history of the provided PyTorch model.

        Parameters
        ----------
        best_models : dict
            Dictionary of best models from GridSearchCV.
        best_model_name : str
            Name of the best model to plot the loss history.
        """
        best_model = best_models[best_model_name]

        if hasattr(best_model, "train_loss_history") and hasattr(
            best_model, "val_loss_history"
        ):
            plt.plot(best_model.train_loss_history, label="Training Loss")
            plt.plot(
                best_model.val_loss_history,
                label="Validation Loss",
                color="orange",
            )
            plt.title("Training vs Validation Loss per Epoch")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            print("The provided model does not have a loss history.")

    def visualize_pipeline(self, model_name, best_models):
        """
        Visualizes the structure of a PyTorch model within the best models.

        Parameters
        ----------
        model_name : str
            The name of the model to visualize.
        best_models : dict
            A dictionary containing the best models.
        """
        best_model = best_models.get(model_name)
        if best_model is None:
            raise ValueError(f"Model with name {model_name} not found in best_models.")

        model = best_model.model
        if isinstance(model, torch.nn.Module):
            print(f"Visualizing the architecture of the model: {model_name}")
            # Assuming input image size of 224x224 and 3 channels
            summary(model, input_size=(3, 224, 224))
        else:
            raise ValueError(
                f"Model {model_name} is not a PyTorch nn.Module, but {type(model)}"
            )
