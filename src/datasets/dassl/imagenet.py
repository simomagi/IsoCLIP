import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict

import PIL
import PIL.Image
from dassl.data.datasets import Datum
from dassl.utils import listdir_nohidden
from torch.utils.data import Dataset

from .oxford_pets import OxfordPets


class ImageNet(Dataset):
    """
    A PyTorch Dataset for the ImageNet dataset, which contains images of a large variety of objects.
    This class handles loading the dataset, splitting it into train and test sets, and preprocessing the images.

    Attributes:
        dataset_dir (Path): The directory where the ImageNet dataset is stored.
        images_folder (Path): The directory containing the images.
        labels_path (str, optional): Path to the labels file (not used in this implementation).
        split_path (str, optional): Path to the split file (not used in this implementation).
        preprocessed (str): Path to the preprocessed file that contains the train and test splits.
        data (List[Datum]): A list of `Datum` objects representing the dataset samples.
        labels (List[int]): A list of labels corresponding to each sample.
        classnames (List[str]): A list of class names for each object category.
    """

    dataset_dir = Path("imagenet")

    def __init__(self, dataroot: Path, split: str, preprocess: callable):
        """
        Initializes the ImageNet dataset by loading the data for the specified split and handling data preprocessing.

        Args:
            dataroot (Path): The root directory where the ImageNet dataset is stored.
            split (str): The dataset split ('train' or 'test').
            preprocess (callable): A function to preprocess the images.

        Raises:
            ValueError: If the split is not valid or if the validation set is requested.
        """
        super().__init__()
        self.preprocess = preprocess
        self.split = split
        self.dataset_dir = dataroot / self.dataset_dir
        
        self.images_folder = self.dataset_dir / "images"
        self.labels_path = None
        self.split_path = None
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")

        # Load preprocessed data if it exists
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
               
        else:
            # If not preprocessed, read classnames and split the data into train and test
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            test = self.read_data(classnames, "val")

            # Save preprocessed data
            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Subsampling the classes if required
        subsample = "all"
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        # Assign the data based on the split
        if self.split == "train":
            self.data = train
        elif self.split == "val":
            raise ValueError("No validation set in ImageNet")
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
        image_path = image_path.replace("/andromeda/datasets/CoOp/imagenet/images", "/mnt/shared/imagenet/imagenet")
        # print(f"Loading image from: {image_path}")
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
            list: A list of class names for the object categories.
        """
        return self.classnames

    @staticmethod
    def read_classnames(text_file: str) -> Dict[str, str]:
        """
        Reads the class names from a text file and returns a dictionary mapping folder names to class names.

        Args:
            text_file (str): The path to the file containing class names.

        Returns:
            dict: A dictionary mapping folder names to class names.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames: Dict[str, str], split_dir: str) -> List[Datum]:
        """
        Reads the data from a split directory and converts it into a list of `Datum` objects.

        Args:
            classnames (Dict[str, str]): A dictionary mapping folder names to class names.
            split_dir (str): The name of the split directory ('train', 'val').

        Returns:
            List[Datum]: A list of `Datum` objects representing the dataset samples.
        """
        split_dir = os.path.join(self.images_folder, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
