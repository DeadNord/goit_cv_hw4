import torchvision.transforms as transforms
from tabulate import tabulate
from PIL import Image


class CNNPreprocessor:
    """
    A class to preprocess image datasets for CNN-based tasks with customizable transformations
    for training, validation, test datasets, and unlabeled datasets.
    """

    def __init__(self):
        """
        Initializes the preprocessor without any transformation settings.
        Transformation settings can be added later using `set_transform_settings()`.
        """
        self.train_transform_settings = None
        self.val_transform_settings = None
        self.test_transform_settings = None

    def set_transform_settings(
        self,
        train_transform_settings=None,
        val_transform_settings=None,
        test_transform_settings=None,
    ):
        """
        Sets transformation settings for training, validation, and test datasets.

        Parameters
        ----------
        train_transform_settings : dict, optional
            Dictionary with settings for training transformations (default is None).
        val_transform_settings : dict, optional
            Dictionary with settings for validation transformations (default is None).
        test_transform_settings : dict, optional
            Dictionary with settings for test transformations (default is None).
        """
        self.train_transform_settings = (
            train_transform_settings if train_transform_settings else {}
        )
        self.val_transform_settings = (
            val_transform_settings if val_transform_settings else {}
        )
        self.test_transform_settings = (
            test_transform_settings if test_transform_settings else {}
        )

    def build_transforms(self, transform_settings):
        """
        Builds the transformation pipeline based on the settings.

        Parameters
        ----------
        transform_settings : dict
            Settings for the transformations.

        Returns
        -------
        torchvision.transforms.Compose
            The transformation pipeline for the dataset.
        """
        transforms_list = []

        if transform_settings.get("ToGray", {}).get("enabled", False):
            transforms_list.append(transforms.Grayscale())

        if transform_settings.get("VerticalFlip", {}).get("enabled", False):
            p = transform_settings["VerticalFlip"].get("p", 0.5)
            transforms_list.append(transforms.RandomVerticalFlip(p=p))

        if transform_settings.get("RandomResizedCrop", {}).get("enabled", False):
            size = transform_settings["RandomResizedCrop"].get("size", 224)
            transforms_list.append(transforms.RandomResizedCrop(size=size))

        if transform_settings.get("RandomHorizontalFlip", {}).get("enabled", False):
            p = transform_settings["RandomHorizontalFlip"].get("p", 0.5)
            transforms_list.append(transforms.RandomHorizontalFlip(p=p))

        if transform_settings.get("Resize", {}).get("enabled", False):
            size = transform_settings["Resize"].get("size", 256)
            transforms_list.append(transforms.Resize(size=size))

        if transform_settings.get("CenterCrop", {}).get("enabled", False):
            size = transform_settings["CenterCrop"].get("size", 224)
            transforms_list.append(transforms.CenterCrop(size=size))

        # Цветовые преобразования
        if transform_settings.get("ColorJitter", {}).get("enabled", False):
            brightness = transform_settings["ColorJitter"].get("brightness", 0.5)
            contrast = transform_settings["ColorJitter"].get("contrast", 0.5)
            saturation = transform_settings["ColorJitter"].get("saturation", 0.5)
            hue = transform_settings["ColorJitter"].get("hue", 0.5)
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            )

        # Шум и искажения
        if transform_settings.get("GaussianBlur", {}).get("enabled", False):
            kernel_size = transform_settings["GaussianBlur"].get("kernel_size", (5, 9))
            sigma = transform_settings["GaussianBlur"].get("sigma", (0.1, 5))
            transforms_list.append(
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            )

        if transform_settings.get("ToTensor", {}).get("enabled", False):
            transforms_list.append(transforms.ToTensor())

        if transform_settings.get("Normalize", {}).get("enabled", True):
            mean = transform_settings["Normalize"].get("mean", [0.485, 0.456, 0.406])
            std = transform_settings["Normalize"].get("std", [0.229, 0.224, 0.225])
            transforms_list.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(transforms_list)

    def transform_dataset(self, dataset, dataset_type="train"):
        """
        Applies the appropriate transformations to the dataset based on the dataset type (train or test).

        Parameters
        ----------
        dataset : torchvision.datasets.ImageFolder
            The dataset to transform.
        dataset_type : str
            The type of dataset ('train' or 'test').

        Returns
        -------
        torchvision.datasets.ImageFolder
            Transformed dataset.
        """
        if dataset_type == "train" and not self.train_transform_settings:
            raise ValueError(
                "Training transformation settings have not been set. Use 'set_transform_settings()' to provide settings."
            )
        elif dataset_type == "test" and not self.test_transform_settings:
            raise ValueError(
                "Test transformation settings have not been set. Use 'set_transform_settings()' to provide settings."
            )

        if dataset_type == "train":
            transform = self.build_transforms(self.train_transform_settings)
        elif dataset_type == "test":
            transform = self.build_transforms(self.test_transform_settings)
        else:
            raise ValueError(
                f"Invalid dataset_type: {dataset_type}. Choose from 'train' or 'test'."
            )

        dataset.transform = transform
        return dataset

    def transform_unlabeled_data(self, images):
        """
        Applies the validation transformations to a list of unlabeled images (e.g., validation set).

        Parameters
        ----------
        images : list of PIL.Image or list of paths to images
            The list of images to be transformed.

        Returns
        -------
        list
            List of transformed images as tensors.
        """
        if not self.val_transform_settings:
            raise ValueError(
                "Validation transformation settings have not been set. Use 'set_transform_settings()' to provide settings."
            )

        transform = self.build_transforms(self.val_transform_settings)

        transformed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            transformed_img = transform(img)
            transformed_images.append(transformed_img)

        return transformed_images

    def help(self):
        """
        Prints the available transformations and their settings for train, val, and test datasets.
        Also includes descriptions of available transformations and their parameters.
        """
        transform_docs = {
            "ToGray": {
                "description": "Converts the image to grayscale.",
                "parameters": {
                    "None": "No parameters. Converts to grayscale using luminance."
                },
            },
            "RandomResizedCrop": {
                "description": "Randomly crops a part of the image and resizes it to the given size.",
                "parameters": {"size": "(int): Target size of the crop."},
            },
            "RandomHorizontalFlip": {
                "description": "Randomly flips the image horizontally with a given probability.",
                "parameters": {
                    "p": "(float): Probability of flipping the image. Default: 0.5"
                },
            },
            "VerticalFlip": {
                "description": "Randomly flips the image vertically with a given probability.",
                "parameters": {
                    "p": "(float): Probability of flipping the image. Default: 0.5"
                },
            },
            "Resize": {
                "description": "Resizes the image to the given size.",
                "parameters": {"size": "(int or tuple): Target size of the image."},
            },
            "CenterCrop": {
                "description": "Crops the center part of the image to the given size.",
                "parameters": {"size": "(int or tuple): Target size of the crop."},
            },
            "ColorJitter": {
                "description": "Randomly changes the brightness, contrast, saturation, and hue of the image.",
                "parameters": {
                    "brightness": "(float): How much to jitter brightness.",
                    "contrast": "(float): How much to jitter contrast.",
                    "saturation": "(float): How much to jitter saturation.",
                    "hue": "(float): How much to jitter hue.",
                },
            },
            "GaussianBlur": {
                "description": "Applies Gaussian blur to the image.",
                "parameters": {
                    "kernel_size": "(tuple): Kernel size for the Gaussian blur.",
                    "sigma": "(tuple): Standard deviation for Gaussian kernel.",
                },
            },
            "Normalize": {
                "description": "Normalizes the image with given mean and standard deviation.",
                "parameters": {
                    "mean": "(sequence): Sequence of means for each channel.",
                    "std": "(sequence): Sequence of standard deviations for each channel.",
                },
            },
            "ToTensor": {
                "description": "Converts a PIL Image or NumPy ndarray to a tensor.",
                "parameters": {
                    "None": "None",
                },
            },
        }

        table_data = []
        for transform_name, details in transform_docs.items():
            params = (
                "\n".join(
                    [
                        f"{param}: {desc}"
                        for param, desc in details["parameters"].items()
                    ]
                )
                or "None"
            )
            table_data.append([transform_name, details["description"], params])

        print("Available transformations and their parameters:")
        print(
            tabulate(
                table_data,
                headers=["Transformation", "Description", "Parameters"],
                tablefmt="grid",
            )
        )
