import csv
import os
from pathlib import Path
from typing import Callable, List, Dict, Union

from PIL import Image
from torch.utils.data import Dataset

# Constants for dataset splits and super classes
_SPLITS = {"train": "Ebay_train", "test": "Ebay_test"}

_SUPER_CLASSES = [
    "bicycle",
    "cabinet",
    "chair",
    "coffee_maker",
    "fan",
    "kettle",
    "lamp",
    "mug",
    "sofa",
    "stapler",
    "table",
    "toaster",
]


class StanfordOnlineProducts(Dataset):
    """
    A PyTorch Dataset class for the Stanford Online Products dataset.

    Attributes:
        dataset_dir (Path): The directory where the dataset is located.
        preprocess (callable): The preprocessing function to apply to the images.
        split (str): The split to load ('train', 'test', or 'all').
        data (List[Dict]): A list of dictionaries containing information about each sample (image path, labels).
        classes (List[str]): A list of super class names for classification.
        num_classes (int): The number of super classes.
    """

    dataset_dir = Path("Stanford_Online_Products")

    def __init__(self, dataroot: Path, split: str, preprocess: Callable):
        """
        Initializes the StanfordOnlineProducts dataset by loading the data for the specified split.

        Args:
            dataroot (Path): The root directory where the Stanford Online Products dataset is stored.
            split (str): The dataset split ('train', 'test', or 'all').
            preprocess (Callable): A preprocessing function for image transformations.

        Raises:
            ValueError: If the split is not one of 'train', 'test', or 'all'.
        """
        super().__init__()
        self.preprocess = preprocess

        self.split = split
        if split not in ['train', 'test', 'all']:
            raise ValueError(f"Invalid split: {split}")

        self.dataset_dir = dataroot / self.dataset_dir

        self.data: List[Dict] = []
        self.classes = _SUPER_CLASSES
        self.num_classes = len(self.classes)

        # Load data from the appropriate file based on the split
        file_paths = [os.path.join(self.dataset_dir, f"{_SPLITS[split]}.txt")] if split in ['train', 'test'] else [
            os.path.join(self.dataset_dir, f"{_SPLITS['train']}.txt"),
            os.path.join(self.dataset_dir, f"{_SPLITS['test']}.txt")]

        for file_path in file_paths:
            with open(file_path, "r") as file_:
                dataset = csv.DictReader(file_, delimiter=" ")
                for row in dataset:
                    self.data.append({
                        "class_id": int(row["class_id"]) - 1,
                        "super_class_id/num": int(row["super_class_id"]) - 1,
                        "super_class_id": self.classes[int(row["super_class_id"]) - 1],
                        "image_path": os.path.join(self.dataset_dir, row["path"]),
                    })

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[Image.Image, int, str]]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the 'image' (processed image), 'label' (class ID),
                  'image_name' (image file name), and 'super_class_id' (super class name).
        """
        sample = self.data[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_name = sample["image_path"].split('/')[-1].split('.')[0]  # Extract the image name (without extension)
        preprocessed_image = self.preprocess(image)

        return {
            "image": preprocessed_image,
            "label": sample["class_id"],  # Class label for the image
            "image_name": image_name,  # The image file name (without extension)
            "super_class_id": sample["super_class_id"],  # The super class ID
        }

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels (class IDs) for all samples in the dataset.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of class IDs for each sample in the dataset.
        """
        return [d['class_id'] for d in self.data]
