import os
from pathlib import Path
from typing import Callable, Dict, List, Union

import PIL.Image
from torch.utils.data import Dataset


class Places365(Dataset):
    """
    A PyTorch Dataset class for the Places365 dataset (validation split).

    This dataset contains 36,500 images organized into 365 scene categories.
    Each category has exactly 100 images, providing a balanced dataset for retrieval tasks.

    Labels are loaded from places365_val.txt which maps each image filename to its
    category label (0-364).

    Attributes:
        dataset_dir (Path): The directory path where the dataset is stored.
        split (str): The split to load (only 'val' is supported).
        preprocess (callable): The preprocessing function to apply to the images.
        image_files (List[str]): Sorted list of image filenames.
        labels_dict (Dict[str, int]): Mapping from filename to label.
    """

    dataset_dir = Path("Places365_val")

    def __init__(self, dataroot: Path, split: str, preprocess: Callable):
        """
        Initializes the Places365 dataset by loading the validation split images.

        Args:
            dataroot (Path): The root path to the dataset.
            split (str): The split to use, must be 'val' for validation.
            preprocess (Callable): The function to preprocess the images.

        Raises:
            ValueError: If the split is not 'val'.
            FileNotFoundError: If the val_large directory or labels file is not found.
            AssertionError: If the number of images is not exactly 36,500.
        """
        super().__init__()
        self.preprocess = preprocess
        self.split = split

        # Only 'val' split is supported
        if split != 'val':
            raise ValueError(f"Invalid split: {split}. Only 'val' split is supported for Places365.")

        self.dataset_dir = dataroot / self.dataset_dir
        self.images_folder = self.dataset_dir / "val_large"

        # Verify the images folder exists
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found at {self.images_folder}")

        # Load labels from places365_val.txt
        labels_file = self.dataset_dir / "places365_val.txt"
        if not labels_file.exists():
            raise FileNotFoundError(
                f"Labels file not found at {labels_file}. "
                f"Please download it from http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
            )

        # Parse labels file: each line is "filename label"
        self.labels_dict = {}
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label = parts
                    self.labels_dict[filename] = int(label)

        # Load all image files and sort them alphabetically
        self.image_files = sorted([
            f for f in os.listdir(self.images_folder)
            if f.endswith('.jpg') and f.startswith('Places365_val_')
        ])

        # Verify we have exactly 36,500 images
        if len(self.image_files) != 36500:
            raise AssertionError(
                f"Expected 36,500 images but found {len(self.image_files)}. "
                f"Please verify the dataset integrity."
            )

        # Verify all images have labels
        missing_labels = [f for f in self.image_files if f not in self.labels_dict]
        if missing_labels:
            raise AssertionError(
                f"Found {len(missing_labels)} images without labels. "
                f"First few: {missing_labels[:10]}"
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples (36,500 for validation split).
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[PIL.Image.Image, int, str]]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve (0-36499).

        Returns:
            dict: A dictionary containing:
                - 'image': The preprocessed image tensor
                - 'label': The category_id (0-364) from places365_val.txt
                - 'image_name': The image filename without extension

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        # Get the image filename
        filename = self.image_files[idx]
        
        # Get label from labels dictionary
        label = self.labels_dict[filename]
        
        image_path = self.images_folder / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found at {image_path}")

        # Load and preprocess image
        image = PIL.Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image)

        # Extract image name without extension
        image_name = filename.rsplit('.', 1)[0]

        return {
            'image': processed_image,
            'label': label,
            'image_name': image_name
        }

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels (category IDs) for all images in the dataset.

        Returns:
            List[int]: A list of category IDs (0-364) for each image in the dataset.
                      Labels are loaded from places365_val.txt
        """
        return [self.labels_dict[f] for f in self.image_files]
