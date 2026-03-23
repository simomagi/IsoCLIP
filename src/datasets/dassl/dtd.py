import os
import random
from pathlib import Path
from typing import List, Dict

import PIL
import PIL.Image
from dassl.data.datasets import Datum
from dassl.utils import listdir_nohidden
from torch.utils.data import Dataset

from .oxford_pets import OxfordPets


class DescribableTextures(Dataset):
    """
    A PyTorch Dataset class for the Describable Textures dataset. This dataset handles loading and splitting data
    for texture classification tasks. It also includes support for subsampling and few-shot learning.

    Attributes:
        dataset_dir (Path): Directory where the dataset is stored.
        images_folder (Path): Directory where the images are located.
        labels_path (str, optional): Path to the labels file (not used in this implementation).
        split_path (Path): Path to the split file (train, validation, test).
        data (List[Datum]): A list of `Datum` objects representing the dataset samples.
        labels (List[int]): A list of labels corresponding to each sample.
        classnames (List[str]): A list of class names.
    """

    dataset_dir = Path("dtd")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        """
        Initializes the DescribableTextures dataset by loading the data for the specified split and handling data preprocessing.

        Args:
            dataroot (Path): The root directory where the DescribableTextures dataset is stored.
            split (str): The dataset split ('train', 'val', or 'test').
            preprocess (callable): A function to preprocess the images.

        Raises:
            ValueError: If the split is not valid.
        """
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "images"
        self.labels_path = None
        self.split_path = self.dataset_dir / "split_zhou_DescribableTextures.json"

        # Load or generate splits if needed
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.images_folder)
        else:
            train, val, test = self.read_and_split_data(self.images_folder, p_trn=0.5, p_val=0.2)
            OxfordPets.save_split(train, val, test, self.split_path, self.images_folder)

        # Subsampling classes if required
        subsample = "all"
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # Assign data based on the split
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

    @staticmethod
    def read_and_split_data(image_dir: str, p_trn: float = 0.5, p_val: float = 0.2, ignored: List[str] = None,
                            new_cnames: Dict[str, str] = None) -> List[Datum]:
        """
        Reads and splits the data into training, validation, and test sets based on the provided proportions.

        Args:
            image_dir (str): Directory where the images are stored.
            p_trn (float): Proportion of data to be used for training.
            p_val (float): Proportion of data to be used for validation.
            ignored (List[str]): List of categories to ignore.
            new_cnames (Dict[str, str]): Dictionary for remapping class names.

        Returns:
            train (List[Datum]): The training data.
            val (List[Datum]): The validation data.
            test (List[Datum]): The test data.
        """
        if ignored is None:
            ignored = []

        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims: List[str], y: int, c: str) -> List[Datum]:
            items = []
            for im in ims:
                # is already 0-based
                item = Datum(impath=im, label=y, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            # Remap class names if necessary
            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test
