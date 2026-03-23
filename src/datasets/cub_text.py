import os
from pathlib import Path
from typing import Callable, Dict, Union

import PIL
import PIL.Image
import pandas as pd
from torch.utils.data import Dataset
import re 


class CUBTextDataset(Dataset):

    def __init__(self, ):

        classes_file = os.path.join("/mnt/hdd/datasets/CUB_200_2011", "classes.txt") # must return a list of strings per label
        # --- Load class names ---
        with open(classes_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        
            
        classnames = [re.split(r'\d+\s+\d+\.', line.strip())[1] for line in lines]
        assert len(classnames) == 200, f"Expected 200 classes, got {len(classnames)}"

        text_prompts = [f"a photo of {c}" for c in classnames]
            
        self.classnames = classnames

    def __getitem__(self, index: int) -> dict:
        
        return {'text': self.classnames[index], 'text_name': "class_" + str(index)}

    def __len__(self) -> int:
        """
        Returns the number of class names in the dataset.

        Returns:
            int: The total number of class names.
        """
        return len(self.classnames)

    def get_labels(self, *args, **kwargs) -> list:
        """
        Retrieves the labels (class IDs) for all class names in the dataset.

        Returns:
            list: A list of labels for each class name in the dataset.
        """
        return list(range(len(self.classnames)))