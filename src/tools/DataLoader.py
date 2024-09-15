import os
from torchvision import datasets, transforms
from PIL import Image


class DataLoader:
    """
    A class to load datasets (train, val, or test) from a local directory.

    Attributes
    ----------
    dataset_type : str
        Type of dataset to load, e.g., 'train', 'val', or 'test'.
    path : str
        Path to the local directory containing the datasets.

    Methods
    -------
    load_data():
        Loads the specified dataset (train or test) using ImageFolder and returns it.

    load_unlabeled_data():
        Loads an unlabeled dataset (val) from a directory and returns a list of image tensors.
    """

    def __init__(self, dataset_type, path):
        """
        Constructs all the necessary attributes for the DataLoader object.

        Parameters
        ----------
        dataset_type : str
            Type of dataset to load, e.g., 'train', 'val', or 'test'.
        path : str
            Path to the local directory of the dataset.
        """
        self.dataset_type = dataset_type
        self.path = path

    def load_data(self):
        """
        Loads the dataset (train/test) using ImageFolder and returns it.

        Returns
        -------
        dataset : torchvision.datasets.ImageFolder
            Loaded dataset as an ImageFolder object with images transformed to tensors.
        """
        # Применение трансформации для преобразования изображений в тензоры
        transform = transforms.Compose(
            [transforms.ToTensor()]  # Преобразует PIL изображение в тензор
        )

        # Используем указанный путь напрямую
        dataset_path = self.path

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Directory for {self.dataset_type} dataset not found at {dataset_path}"
            )

        # Load dataset using ImageFolder with the ToTensor transform
        dataset = datasets.ImageFolder(dataset_path, transform=transform)

        return dataset

    def load_unlabeled_data(self):
        """
        Loads an unlabeled dataset (e.g., validation set) from a directory of images.

        Returns
        -------
        list
            List of image tensors.
        """
        # Применение трансформации для преобразования изображений в тензоры
        transform = transforms.Compose([transforms.ToTensor()])

        images = []

        # Проверка, что директория существует
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Directory for unlabeled dataset not found at {self.path}"
            )

        # Загрузка всех изображений из директории
        for file_name in os.listdir(self.path):
            if file_name.endswith(
                (".png", ".jpg", ".jpeg")
            ):  # Поддерживаемые форматы изображений
                img_path = os.path.join(self.path, file_name)
                img = Image.open(img_path).convert(
                    "RGB"
                )  # Открытие изображения и преобразование в RGB
                img_tensor = transform(img)  # Преобразование в тензор
                images.append(img_tensor)

        return images
