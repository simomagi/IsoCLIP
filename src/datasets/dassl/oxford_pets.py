import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import PIL
import PIL.Image
import math
from dassl.data.datasets import Datum
from dassl.utils import read_json, write_json
from torch.utils.data import Dataset


class OxfordPets(Dataset):
    """
    A PyTorch Dataset for the Oxford Pets dataset, which contains images of various pet breeds.
    This class handles loading the dataset, splitting it into train, validation, and test sets, and preprocessing the images.

    Attributes:
        dataset_dir (Path): The directory where the Oxford Pets dataset is stored.
        images_folder (Path): The directory containing the images.
        labels_path (Path): Path to the annotations directory.
        split_path (Path): Path to the split file (train, validation, test).
        split_fewshot_dir (Path): Directory for storing preprocessed few-shot data.
        data (List[Datum]): A list of `Datum` objects representing the dataset samples.
        labels (List[int]): A list of labels corresponding to each sample.
        classnames (List[str]): A list of class names for each pet breed.
    """

    dataset_dir = Path("oxford_pets")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        """
        Initializes the Oxford Pets dataset by loading the data for the specified split and handling data preprocessing.

        Args:
            dataroot (Path): The root directory where the Oxford Pets dataset is stored.
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
        self.labels_path = self.dataset_dir / "annotations"
        self.split_path = self.dataset_dir / "split_zhou_OxfordPets.json"

        # Load or generate dataset splits
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.images_folder)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval, p_val=0.2)
            self.save_split(train, val, test, self.split_path, self.images_folder)

        # Subsampling the classes if required
        subsample = "all"
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

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
            list: A list of class names for the pet breeds.
        """
        return self.classnames

    def read_data(self, split_file: str) -> List[Datum]:
        """
        Reads the data from a split file and converts it into a list of `Datum` objects.

        Args:
            split_file (str): The name of the split file to read.

        Returns:
            List[Datum]: A list of `Datum` objects representing the dataset samples.
        """
        filepath = os.path.join(self.labels_path, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.images_folder, imname)
                # Convert to 0-based labels
                label = int(label) - 1
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval: List[Datum], p_val: float = 0.2) -> (List[Datum], List[Datum]):
        """
        Splits the training data into train and validation sets.

        Args:
            trainval (List[Datum]): The list of all training data.
            p_val (float): The proportion of data to use for validation.

        Returns:
            train (List[Datum]): The training data.
            val (List[Datum]): The validation data.
        """
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train: List[Datum], val: List[Datum], test: List[Datum], filepath: Path, path_prefix: str):
        """
        Saves the dataset splits (train, validation, test) to a JSON file.

        Args:
            train (List[Datum]): The training data.
            val (List[Datum]): The validation data.
            test (List[Datum]): The test data.
            filepath (Path): The path where the split file should be saved.
            path_prefix (str): Prefix for the image paths.
        """
        def _extract(items: List[Datum]) -> List[tuple]:
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath: Path, path_prefix: str) -> (List[Datum], List[Datum], List[Datum]):
        """
        Reads a dataset split from a JSON file.

        Args:
            filepath (Path): Path to the split file.
            path_prefix (str): Prefix for the image paths.

        Returns:
            train (List[Datum]): The training data.
            val (List[Datum]): The validation data.
            test (List[Datum]): The test data.
        """
        def _convert(items: List[tuple]) -> List[Datum]:
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample: str = "all") -> List[List[Datum]]:
        """
        Subsamples classes into two groups: base classes and new classes.

        Args:
            args: A list of datasets, e.g. train, val, and test.
            subsample (str): What classes to subsample ("all", "base", "new").

        Returns:
            List: A list of datasets (train, val, test) after subsampling.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output
