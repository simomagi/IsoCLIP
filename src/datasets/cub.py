import os
from pathlib import Path
from typing import Callable, Dict, Union

import PIL
import PIL.Image
import pandas as pd
from torch.utils.data import Dataset


class CUB(Dataset):
    """
    A PyTorch Dataset class for the CUB-200-2011 dataset.

    Attributes:
        dataset_dir (Path): The directory path where the dataset is stored.
        split (str): The split to load ('train', 'test', or 'all').
        images_folder (Path): The folder containing image files.
        data (pd.DataFrame): The DataFrame containing image metadata (filepath, class labels, split information).
        preprocess (callable): The preprocessing function to apply to the images.
    """

    dataset_dir = Path("CUB_200_2011")

    def __init__(self, dataroot: Path, split: str, preprocess: Callable):
        """
        Initializes the CUB dataset by loading the metadata and applying preprocessing.

        Args:
            dataroot (Path): The root path to the dataset.
            split (str): The split to use, can be 'train', 'test', or 'all'.
            preprocess (Callable): The function to preprocess the images.

        Raises:
            ValueError: If the split is not 'train', 'test', or 'all'.
        """
        super().__init__()
        self.preprocess = preprocess

        self.split = split
        if split not in ['train', 'test', 'all']:
            raise ValueError(f"Invalid split: {split}")

        self.dataset_dir = dataroot / self.dataset_dir
        self.images_folder = self.dataset_dir / "images"

        self.data = self.load()

    def load(self) -> pd.DataFrame:
        """
        Loads the dataset annotations and merges them based on the split.

        Returns:
            pd.DataFrame: A DataFrame containing the image metadata (filepaths, class labels, split information).
        """
        # Load the images, class labels, and train/test split information
        images = pd.read_csv(self.dataset_dir / "images.txt", sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(self.dataset_dir / "image_class_labels.txt", sep=' ',
                                         names=['img_id', 'target'])
        train_test_split = pd.read_csv(self.dataset_dir / "train_test_split.txt", sep=' ',
                                       names=['img_id', 'is_training_img'])

        # Merge the data
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        # Filter the data based on the split
        if self.split == "train":
            data = data[data.is_training_img == 1]
        elif self.split == "test":
            data = data[data.is_training_img == 0]
        elif self.split == "all":
            pass

        return data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[PIL.Image.Image, int, str]]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image, label, and image name.
        """
        sample = self.data.iloc[idx]
        path = os.path.join(self.images_folder, sample.filepath)
        image_name = sample.filepath.split('/')[1].split('.')[0]  # Extract image name from filepath
        processed_image = self.preprocess(PIL.Image.open(path).convert("RGB"))  # Open and preprocess the image
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0-indexed

        return {
            'image': processed_image,
            'label': target,
            'image_name': image_name
        }

    def get_labels(self, *args, **kwargs) -> list:
        """
        Retrieves the labels (class IDs) for all images in the dataset.

        Returns:
            list: A list of labels for each image in the dataset.
        """
        return self.data.target.to_list()
