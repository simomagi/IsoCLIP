from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict

import pandas as pd
from torch.utils.data import Dataset


class Flickr30KTextDataset(Dataset):
    """
    A PyTorch Dataset class for the Flickr30K dataset with captions. This dataset is used for image-text retrieval tasks
    where captions for images are used as text data.

    Attributes:
        dataset_path (Path): Path to the Flickr30K dataset directory.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        image_to_captions (defaultdict): A dictionary mapping image names to their associated captions.
        image_names (List[str]): A list of image names.
        captions (List[str]): A list of captions corresponding to the images.
        caption_names (List[str]): A list of unique caption names (image_name_caption_id).
        labels (List[int]): A list of image indices corresponding to each caption.
    """

    def __init__(self, dataroot: Union[str, Path], split: str):
        """
        Initializes the Flickr30KTextDataset by loading captions and processing them based on the split.

        Args:
            dataroot (Union[str, Path]): The root directory where the Flickr30K dataset is stored.
            split (str): The split to use ('train', 'test', 'val', 'all', etc.).

        Raises:
            ValueError: If the split is not valid.
        """
        dataroot = Path(dataroot)

        dataset_path = dataroot / 'Flickr30K'
        self.dataset_path = dataset_path
        self.split = split

        # Validate the split
        assert split in ['train', 'train_query', 'train_gallery', 'val', 'val_query', 'val_gallery', 'test',
                         'test_query', 'test_gallery', 'all', 'all_query', 'all_gallery']
        if 'train' in split:
            filename_split = 'train'
        elif 'val' in split:
            filename_split = 'val'
        elif 'test' in split:
            filename_split = 'test'
        elif 'all' in split:
            filename_split = 'all'
        else:
            raise ValueError(f'Unknown split: {split}')

        # Load image names for the split (e.g., train, val, test)
        split_image_names = None
        if filename_split != 'all':
            with open(dataset_path / f'karpathy_{filename_split}.txt', 'r') as f:
                split_image_names = set(f.read().splitlines())  # These are without the .jpg extension

        # Initialize dictionary to store image-to-caption mappings
        self.image_to_captions = defaultdict(list)

        # Load the annotation CSV file
        annotation_csv = pd.read_csv(dataset_path / 'results.csv', delimiter='|')

        # Process each row in the annotation file
        for index, row in annotation_csv.iterrows():
            if split_image_names and row['image_name'].replace('.jpg', '') not in split_image_names:
                continue
            augmented_caption = {'id': row[' comment_number'], 'caption': row[' comment']}
            self.image_to_captions[row['image_name']].append(augmented_caption)

        # Handle 'gallery' and 'query' splits
        if 'gallery' in split:  # Discard the first caption of each image for the gallery
            for image_name, captions in self.image_to_captions.items():
                self.image_to_captions[image_name] = captions[1:]
        elif 'query' in split:  # Use only the first caption of each image as the query
            for image_name, captions in self.image_to_captions.items():
                self.image_to_captions[image_name] = captions[:1]

        # Create lists for captions, caption names, and labels
        self.image_names = list(self.image_to_captions.keys())
        self.captions: List[str] = []
        self.caption_names: List[str] = []
        self.labels: List[int] = []

        for image_name, captions in self.image_to_captions.items():
            for caption in captions:
                self.captions.append(caption['caption'])
                self.caption_names.append(f"{image_name.strip()}_{caption['id']}")
                self.labels.append(self.image_names.index(image_name))

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        """
        Retrieves a caption and its corresponding caption name for the specified index.

        Args:
            index (int): The index of the caption to retrieve.

        Returns:
            dict: A dictionary containing the 'text' (caption) and 'text_name' (caption name).
        """
        data = {}
        caption = self.captions[index]
        data['text'] = caption.strip()
        data['text_name'] = self.caption_names[index]
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
        Retrieves the image labels (indices) for the dataset.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of image indices corresponding to each caption.

        Raises:
            ValueError: If labels are requested for the 'train', 'val', or 'test' splits.
        """
        if self.split in ['train', 'val', 'test', 'all']:
            raise ValueError('Labels are not available for the train, val, or test splits')
        return self.labels
