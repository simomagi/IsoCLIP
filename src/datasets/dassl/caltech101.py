import os
from pathlib import Path
from typing import List, Dict

import PIL
import PIL.Image
from torch.utils.data import Dataset

from .dtd import DescribableTextures as DTD
from .oxford_pets import OxfordPets

# Constants for ignored classes and new class names
IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


class Caltech101(Dataset):
    """
    A PyTorch Dataset for the Caltech 101 dataset, which contains images of 101 object categories.
    This class handles dataset loading, splitting, and preprocessing, including support for few-shot learning.

    Attributes:
        dataset_dir (Path): Directory where the dataset is stored.
        images_folder (Path): Directory where the images are located.
        labels_path (str, optional): Path to the labels file (not used in this implementation).
        split_path (Path): Path to the split file (train, validation, test).
        data (List[Datum]): A list of `Datum` objects representing the dataset samples.
        labels (List[int]): A list of labels corresponding to each sample.
        classnames (List[str]): A list of class names.
    """

    dataset_dir = Path("caltech-101")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        """
        Initializes the Caltech101 dataset by loading the data for the specified split and handling data preprocessing.

        Args:
            dataroot (Path): The root directory where the Caltech 101 dataset is stored.
            split (str): The dataset split ('train', 'val', or 'test').
            preprocess (callable): A function to preprocess the images.

        Raises:
            ValueError: If the split is not valid.
        """
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "101_ObjectCategories"
        self.labels_path = None
        self.split_path = self.dataset_dir / "split_zhou_Caltech101.json"

        # Load or generate dataset splits
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.images_folder)
        else:
            # p_tst = 1 - p_trn - p_val
            train, val, test = DTD.read_and_split_data(self.images_folder, ignored=IGNORED, new_cnames=NEW_CNAMES, p_trn=0.5, p_val=0.2)
            OxfordPets.save_split(train, val, test, self.split_path, self.images_folder)

        # Subsampling the classes if required
        subsample = "all"
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # Assign the data based on the split
        if self.split == "train":
            self.data = train
        elif self.split == "val":
            self.data = val
        elif self.split == "test":
            self.data = test
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Prepare the labels and classnames
        self.labels = [item.label for item in self.data]
        label2classname = {item.label: item.classname for item in self.data}
        self.classnames = [label2classname[label].replace("_", " ") for label in sorted(label2classname)]

    def __getitem__(self, index: int) -> Dict[str, object]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the 'image' (processed image), 'image_name' (image filename),
                  'label' (image label), and 'super_class_id' (optional).
        """
        image_path = str(self.data[index].impath)
        label = self.data[index].label
        image = self.preprocess(PIL.Image.open(image_path))
        image_name = f"{Path(image_path).parent.name}__{Path(image_path).name}"

        return {
            'image': image,
            'image_name': image_name,
            'label': label
        }

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels for the dataset.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of labels for the dataset samples.
        """
        return self.labels

    def get_classnames(self, *args, **kwargs) -> List[str]:
        """
        Retrieves the class names for the dataset.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of class names.
        """
        return self.classnames
