import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Union

import PIL.Image
from torch.utils.data import Dataset


class INaturalist2021(Dataset):
    """
    A PyTorch Dataset class for the iNaturalist2021 dataset (train_mini split).

    This dataset contains 500,000 images organized into 10,000 fine-grained species categories.
    Each category has exactly 50 images, providing a balanced dataset for retrieval tasks.

    Attributes:
        dataset_dir (Path): The directory path where the dataset is stored.
        split (str): The split to load (only 'train' is supported for train_mini).
        preprocess (callable): The preprocessing function to apply to the images.
        annotations (dict): The loaded JSON annotations containing image metadata.
        images (List[dict]): List of image metadata dictionaries.
        image_to_category (Dict[int, int]): Mapping from image_id to category_id.
    """

    dataset_dir = Path("iNaturalist2021_train")

    def __init__(self, dataroot: Path, split: str, preprocess: Callable):
        """
        Initializes the iNaturalist2021 dataset by loading annotations and metadata.

        Args:
            dataroot (Path): The root path to the dataset.
            split (str): The split to use, must be 'train' for train_mini.
            preprocess (Callable): The function to preprocess the images.

        Raises:
            ValueError: If the split is not 'train'.
            FileNotFoundError: If the annotations file is not found.
        """
        super().__init__()
        self.preprocess = preprocess
        self.split = split

        # Only 'train' split is supported (corresponds to train_mini)
        if split != 'train':
            raise ValueError(f"Invalid split: {split}. Only 'train' split is supported for iNaturalist2021.")

        self.dataset_dir = dataroot / self.dataset_dir
        self.images_folder = self.dataset_dir / "train_mini"

        # Load annotations
        annotations_path = self.dataset_dir / "annotations" / "train_mini.json"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found at {annotations_path}")

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Build image list and image_id to category_id mapping
        self.images = self.annotations['images']
        
        # Create efficient mapping from image_id to category_id
        self.image_to_category = {}
        for ann in self.annotations['annotations']:
            self.image_to_category[ann['image_id']] = ann['category_id']

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples (500,000 for train_mini).
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[PIL.Image.Image, int, str]]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image': The preprocessed image tensor
                - 'label': The category_id (0-9999)
                - 'image_name': A unique identifier for the image (e.g., 'inat_12345')

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        img_metadata = self.images[idx]
        image_id = img_metadata['id']
        file_name = img_metadata['file_name']  # Format: 'category_folder/uuid.jpg'

        # Get category_id for this image
        category_id = self.image_to_category[image_id]

        # Construct full image path
        image_path = self.dataset_dir / file_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")

        # Load and preprocess image
        image = PIL.Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image)

        # Create a unique image name identifier
        image_name = f"inat_{image_id}"

        return {
            'image': processed_image,
            'label': category_id,
            'image_name': image_name
        }

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels (category IDs) for all images in the dataset.

        Returns:
            List[int]: A list of category IDs (0-9999) for each image in the dataset.
        """
        return [self.image_to_category[img['id']] for img in self.images]
