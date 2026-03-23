import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import PIL
import PIL.Image
from dassl.data.datasets import Datum
from dassl.utils import read_json
from scipy.io import loadmat
from torch.utils.data import Dataset

from .oxford_pets import OxfordPets


class OxfordFlowers(Dataset):
    """
    A PyTorch Dataset for the Oxford Flowers dataset, which contains images of 102 flower categories.
    This class handles loading the dataset, splitting it into train, validation, and test sets, and preprocessing the images.

    Attributes:
        dataset_dir (Path): The directory where the Oxford Flowers dataset is stored.
        images_folder (Path): The directory containing the images.
        label_file (Path): Path to the label file.
        lab2cname_file (Path): Path to the label to class name mapping file.
        split_path (Path): Path to the split file (train, validation, test).
        split_fewshot_dir (Path): Directory for storing preprocessed few-shot data.
        data (List[Datum]): A list of `Datum` objects representing the dataset samples.
        labels (List[int]): A list of labels corresponding to each sample.
        classnames (List[str]): A list of class names for each flower category.
    """

    dataset_dir = Path("oxford_flowers")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        """
        Initializes the Oxford Flowers dataset by loading the data for the specified split and handling data preprocessing.

        Args:
            dataroot (Path): The root directory where the Oxford Flowers dataset is stored.
            split (str): The dataset split ('train', 'val', or 'test').
            preprocess (callable): A function to preprocess the images.

        Raises:
            ValueError: If the split is not valid.
        """
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir

        self.images_folder = self.dataset_dir / "jpg"
        self.label_file = self.dataset_dir / "imagelabels.mat"
        self.lab2cname_file = self.dataset_dir / "cat_to_name.json"
        self.split_path = self.dataset_dir / "split_zhou_OxfordFlowers.json"
        self.split_fewshot_dir = self.dataset_dir / "split_fewshot"

        # Load or generate dataset splits
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.images_folder)
        else:
            train, val, test = self.read_data()
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
            list: A list of class names for the flower categories.
        """
        return self.classnames

    def read_data(self) -> List[Datum]:
        """
        Reads and splits the dataset into train, validation, and test sets.

        Returns:
            train (List[Datum]): The training data.
            val (List[Datum]): The validation data.
            test (List[Datum]): The test data.
        """
        tracker = defaultdict(list)
        label_file = loadmat(str(self.label_file))["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.images_folder, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims: List[str], y: int, c: str) -> List[Datum]:
            items = []
            for im in ims:
                # Convert to 0-based label
                item = Datum(impath=im, label=y - 1, classname=c)
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train: n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val:], label, cname))

        return train, val, test
