# Code for loading the Flickr30K dataset
from collections import defaultdict
from pathlib import Path
from typing import Union

import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset


class Flickr30KDataset(Dataset):
    def __init__(self, dataroot: Union[str, Path], split: str, preprocess: callable, return_image=False):
        dataroot = Path(dataroot)

        dataset_path = dataroot / 'Flickr30K'
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess
        self.return_image = return_image

        assert split in ['train', 'val', 'test', 'all'], 'Split must be either train, val, test or all'
        split_image_names = None
        if split != 'all':
            with open(dataset_path / f'karpathy_{split}.txt', 'r') as f:
                split_image_names = set(f.read().splitlines())  # These are without the .jpg extension

        self.image_to_captions = defaultdict(list)
        annotation_csv = pd.read_csv(dataset_path / 'results.csv', delimiter='|')
        # iterate over the rows of the csv file
        for index, row in annotation_csv.iterrows():
            if split_image_names and row['image_name'].replace('.jpg', '') not in split_image_names:
                continue
            self.image_to_captions[row['image_name']].append(row[' comment'])

        self.image_names = list(self.image_to_captions.keys())
        self.image_labels = torch.arange(len(self.image_names))
        self.text_labels = []
        for image_name, captions in self.image_to_captions.items():
            self.text_labels.extend([self.image_names.index(image_name)] * len(captions))

    def __getitem__(self, index):
        data = {}
        image_name = self.image_names[index]
        data['image_name'] = image_name

        if self.return_image:
            image_path = self.dataset_path / 'images' / image_name
            image = PIL.Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            data['image'] = image

        captions = self.image_to_captions[image_name]
        data['text'] = captions
        data['text_name'] = [image_name + f'_{i}' for i in range(len(captions))]

        return data

    def __len__(self):
        return len(self.image_names)

    def get_labels(self, features_type: str, *args, **kwargs):
        if features_type == 'image':
            return self.image_labels
        elif features_type == 'text':
            return self.text_labels
        else:
            raise ValueError(f'Unknown features_type: {features_type}')


# Code for loading the Flickr30K dataset
if __name__ == '__main__':
    from tqdm import tqdm
    import clip

    _, preprocess = clip.load('ViT-B/32')
    dataset = Flickr30KDataset('/andromeda/datasets', 'all', preprocess, return_image=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for data in tqdm(loader):
        pass

# text-to-image original test
# flickr30k_recall_at_1 = 71.50
# flickr30k_recall_at_5 = 91.76
# flickr30k_recall_at_10 = 95.46

# text-to-image karpathy test
# flickr30k_recall_at_1 = 66.94
# flickr30k_recall_at_5 = 88.96
# flickr30k_recall_at_10 = 93.38


# image-to-text original test
# flickr30k_recall_at_1 = 88.10
# flickr30k_recall_at_5 = 98.20
# flickr30k_recall_at_10 = 99.60


# image-to-text karpathy test
# flickr30k_recall_at_1 = 87.70
# flickr30k_recall_at_5 = 98.50
# flickr30k_recall_at_10 = 99.40