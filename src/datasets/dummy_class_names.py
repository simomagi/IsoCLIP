from typing import List
from torch.utils.data import Dataset


class DummyClassNamesDataset(Dataset):
    """
    A simple dataset for handling dummy class names. This dataset returns class names both as 'text' and 'text_name'.
    It's typically used for tasks like OVI (Open Vocabulary Inversion), where the class name will be treated as text
    and used in conjunction with a fixed prompt like "a photo of" to generate corresponding features.

    Attributes:
        classnames (List[str]): A list of class names.
    """

    def __init__(self, classnames: List[str]):
        """
        Initializes the DummyClassNamesDataset with the provided class names.

        Args:
            classnames (List[str]): A list of class names to be used as text.
        """
        self.classnames = classnames

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves the 'text' and 'text_name' for the class at the specified index.

        Args:
            index (int): The index of the class name to retrieve.

        Returns:
            dict: A dictionary containing the 'text' (class name) and 'text_name' (class name).
        """
        return {'text': self.classnames[index], 'text_name': self.classnames[index]}

    def __len__(self) -> int:
        """
        Returns the number of class names in the dataset.

        Returns:
            int: The total number of class names.
        """
        return len(self.classnames)
