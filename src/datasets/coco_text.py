import json
from pathlib import Path
from typing import Union, List, Dict

from torch.utils.data import Dataset


class CocoTextDataset(Dataset):
    """
    A PyTorch Dataset for the COCO Text dataset. This class handles loading and splitting the dataset for text-based
    retrieval tasks (e.g., gallery and query splits for training, validation, and testing).

    Attributes:
        dataset_path (Path): Path to the dataset directory.
        split (str): The split to load (e.g., 'train', 'val', 'test').
        annotations (List[Dict]): List of image annotations for the selected split.
        captions (List[str]): List of captions corresponding to the images.
        captions_id (List[int]): List of caption IDs.
        labels (List[int]): List of image IDs corresponding to captions.
    """

    def __init__(self, dataroot: Union[str, Path], split: str):
        """
        Initializes the CocoTextDataset object by loading the annotations and captions from the dataset.

        Args:
            dataroot (Union[str, Path]): The root directory where the COCO dataset is stored.
            split (str): The split to use (e.g., 'train', 'train_gallery', 'train_query', 'val', 'val_gallery',
                         'val_query', 'test', 'test_query', 'test_gallery').

        Raises:
            ValueError: If the provided split is not recognized.
        """
        dataroot = Path(dataroot)

        # Ensure the split is valid
        assert split in ['train', 'train_gallery', 'train_query', 'val', 'val_gallery',
                         'val_query', 'test', 'test_query', 'test_gallery'], f'Unknown split: {split}'

        # Determine the filename split (train, val, or test)
        if 'train' in split:
            filename_split = 'train'
        elif 'val' in split:
            filename_split = 'val'
        elif 'test' in split:
            filename_split = 'test'
        else:
            raise ValueError(f'Unknown split: {split}')

        # Set dataset path
        dataset_path = dataroot / 'COCO2014'
        self.dataset_path = dataset_path
        self.split = split

        # Load annotations from the dataset
        with open(dataset_path / 'dataset_coco_light.json', 'r') as f:
            annotations = json.load(f)
        annotations = annotations['images']

        # Filter annotations based on the split
        if filename_split == 'train':
            annotations = [annotation for annotation in annotations if annotation['split'] in ['train', 'restval']]
        else:
            annotations = [annotation for annotation in annotations if annotation['split'] == filename_split]

        self.annotations = annotations

        # Initialize lists to store captions, captions IDs, and image labels
        self.captions: List[str] = []
        self.captions_id: List[int] = []
        self.labels: List[int] = []

        # Process annotations and create dataset entries
        for annotation in annotations:
            if 'gallery' in split:
                annotation['sentences'] = annotation['sentences'][1:]  # Use all but the first sentence for gallery
            elif 'query' in split:
                annotation['sentences'] = annotation['sentences'][:1]  # Use only the first sentence for query

            for sentence in annotation['sentences']:
                self.captions.append(sentence['raw'])  # Store the caption text
                self.captions_id.append(sentence['sentid'])  # Store the caption ID
                self.labels.append(annotation['imgid'])  # Store the image ID

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        """
        Retrieve a data sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the caption and its ID.
        """
        data = {}
        caption_id = self.captions_id[index]
        caption = self.captions[index]

        data['text'] = caption
        data['text_name'] = str(caption_id)  # Convert the caption ID to string

        return data

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The total number of captions (samples) in the dataset.
        """
        return len(self.captions)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Get the labels for the dataset, i.e., the image IDs associated with captions.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of image IDs corresponding to the captions.

        Raises:
            ValueError: If labels are requested for a split that is not 'gallery' or 'query'.
        """
        if self.split in ['train', 'val', 'test']:
            raise ValueError('Labels are not available when "gallery" or "query" are not in the split')
        return self.labels
