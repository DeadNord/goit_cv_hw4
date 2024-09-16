import matplotlib.pyplot as plt
import numpy as np
import torch


class EDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on image datasets, specifically for CNN-based tasks.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        The dataset (train/val/test) to be analyzed.
    class_names : list
        List of class names in the dataset.
    """

    def __init__(self, dataset, class_names):
        """
        Constructs all the necessary attributes for the EDA object.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset (train/val/test) to be analyzed.
        class_names : list
            List of class names in the dataset.
        """
        self.dataset = dataset
        self.class_names = class_names

    def show_sample_images(self, num_images=6):
        """
        Displays a grid of sample images from the dataset.

        Parameters
        ----------
        num_images : int, optional
            Number of images to display (default is 6).
        """

        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=num_images, shuffle=True
        )
        inputs, classes = next(iter(dataloader))

        self.imshow(inputs, title=[self.class_names[x] for x in classes])

    def imshow(self, inputs, title=None):
        """
        Helper function to display images.

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing image data.
        title : list of str, optional
            The titles (class names) for each image.
        """

        inputs = inputs.numpy()

        fig, axes = plt.subplots(1, len(inputs), figsize=(15, 5))
        for idx, img in enumerate(inputs):
            img = img.transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            axes[idx].imshow(img)
            if title is not None:
                axes[idx].set_title(title[idx])
            axes[idx].axis("off")
        plt.tight_layout()
        plt.show()

    def show_images_from_each_class(self):
        """
        Displays one image from each class in the dataset.
        """
        images_per_class = {}

        for img, label in self.dataset:
            class_name = self.class_names[label]
            if class_name not in images_per_class:
                images_per_class[class_name] = img
            if len(images_per_class) == len(self.class_names):
                break

        plt.figure(figsize=(10, 10))
        for i, (class_name, img) in enumerate(images_per_class.items()):
            ax = plt.subplot(1, len(self.class_names), i + 1)
            img = img.numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(class_name)
            ax.axis("off")
        plt.show()

    def show_batch_images(self, dataloader, batch_size=4):
        """
        Displays a batch of images from the dataset.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader for the dataset.
        batch_size : int, optional
            Number of images to display in the batch (default is 4).
        """

        inputs, classes = next(iter(dataloader))

        self.imshow(inputs, title=[self.class_names[x] for x in classes])

    def plot_class_distribution(self):
        """
        Plots the distribution of images across different classes in the dataset.
        """
        class_counts = [0] * len(self.class_names)

        for _, label in self.dataset:
            class_counts[label] += 1

        plt.figure(figsize=(10, 6))
        plt.bar(self.class_names, class_counts, color="blue")
        plt.title("Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.show()

    def display_image_by_index(self, idx):
        """
        Displays a specific image by its index in the dataset.

        Parameters
        ----------
        idx : int
            Index of the image to display.
        """
        img, label = self.dataset[idx]

        img = img.numpy().transpose(1, 2, 0)

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Class: {self.class_names[label]}")
        plt.axis("off")
        plt.show()

    def show_image_shape(self):
        """
        Displays the shape of the first image in the dataset to verify dimensions.
        """
        img, _ = self.dataset[0]
        print(f"Shape of an image: {img.shape}")
