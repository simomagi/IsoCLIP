import json
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, List

from torch.utils.data import Dataset


class NoCapsTextDataset(Dataset):
    """
    A PyTorch Dataset for the NoCaps dataset. This dataset is used for image-text retrieval tasks where captions are
    associated with images, and a split is specified for query and gallery splits.

    Attributes:
        dataset_path (Path): The path to the NoCaps dataset directory.
        split (str): The dataset split ('val', 'val_gallery', 'val_query').
        captions (List[str]): A list of captions corresponding to the images.
        captions_id (List[int]): A list of caption IDs.
        labels (List[int]): A list of image indices corresponding to each caption.
    """

    def __init__(self, dataroot: Union[str, Path], split: str):
        """
        Initializes the NoCapsTextDataset by loading and processing captions based on the split.

        Args:
            dataroot (Union[str, Path]): The root directory where the NoCaps dataset is stored.
            split (str): The split to use ('val', 'val_gallery', 'val_query').

        Raises:
            ValueError: If the split is not 'val', 'val_gallery', or 'val_query'.
        """
        dataroot = Path(dataroot)

        assert split in ['val', 'val_gallery', 'val_query'], f'Unknown split: {split}'
        self.split = split

        # Set dataset path
        dataset_path = dataroot / 'nocaps'
        self.dataset_path = dataset_path

        # Load captions from the dataset JSON file
        imageid_to_captions = defaultdict(list)
        with open(dataset_path / 'nocaps_val_4500_captions.json', 'r') as f:
            captions = json.load(f)['annotations']
        for caption in captions:
            imageid_to_captions[caption['image_id']].append(caption)

        # Handle the gallery and query splits
        if 'gallery' in split:  # Discard the first caption of each image for gallery
            for image_id, captions in imageid_to_captions.items():
                imageid_to_captions[image_id] = captions[1:]
        elif 'query' in split:  # Use only the first caption of each image as the query
            for image_id, captions in imageid_to_captions.items():
                imageid_to_captions[image_id] = captions[:1]

        # Prepare lists for captions, caption IDs, and labels
        image_ids = list(imageid_to_captions.keys())
        self.captions: List[str] = []
        self.captions_id: List[int] = []
        self.labels: List[int] = []
        for image_id, captions in imageid_to_captions.items():
            for caption in captions:
                self.captions.append(caption['caption'])
                self.captions_id.append(caption['id'])
                self.labels.append(image_ids.index(image_id))

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        """
        Retrieves a caption and its corresponding ID for the sample at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'text' (caption) and 'text_name' (caption ID).
        """
        data = {}
        caption_id = self.captions_id[index]
        caption = self.captions[index]

        data['text'] = caption
        data['text_name'] = str(caption_id)

        return data

    def __len__(self) -> int:
        """
        Returns the total number of captions in the dataset.

        Returns:
            int: The total number of captions.
        """
        return len(self.captions)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels for the dataset. Labels are not available for the 'val' split.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of image indices corresponding to each caption.

        Raises:
            ValueError: If labels are requested for the 'val' split.
        """
        if self.split in ['val']:
            raise ValueError('Labels are not available for val split')
        return self.labels
