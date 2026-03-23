# Code for loading the COCO 2017 dataset
import json
from pathlib import Path
from typing import Union

import PIL
import torch
from torch.utils.data import Dataset


class ModifiedCocoDataset(Dataset):
    def __init__(self, dataroot: Union[str, Path], split: str, preprocess: callable, return_image=False):
        dataroot = Path(dataroot)

        assert split in ['train', 'val', 'test'], f'Unknown split: {split}'

        dataset_path = dataroot / 'COCO2014'
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess
        self.return_image = return_image

        with open(dataset_path / 'dataset_coco_light.json', 'r') as f:
            annotations = json.load(f)
        annotations = annotations['images']

        # Filter out images that are not in the split, if split == 'train' keep both 'train' and 'restval'
        if split == 'train':
            annotations = [annotation for annotation in annotations if annotation['split'] in ['train', 'restval']]
        else:
            annotations = [annotation for annotation in annotations if annotation['split'] == split]

        self.annotations = annotations

        # get labels
        self.image_labels = list(range(len(annotations)))
        self.text_labels = []
        for idx, annotation in enumerate(annotations):
            self.text_labels.extend([idx] * len(annotation['sentences']))

    def __getitem__(self, index):
        data = {}
        annotation = self.annotations[index]
        image_name = annotation['filename']
        data['image_name'] = image_name
        if self.return_image:
            image_path = self.dataset_path / annotation['filepath'] / annotation['filename']
            image = PIL.Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            data['image'] = image

        captions = [caption['raw'] for caption in annotation['sentences']]
        captions += [''] * (7 - len(captions))  # Pad to 7 captions
        data['text'] = captions
        data['text_name'] = [image_name + f'_{i}' for i in range(7)]
        return data

    def __len__(self):
        return len(self.annotations)

    def get_labels(self, features_type: str, *args, **kwargs):
        if features_type == 'image' or features_type == 'image_noproj':
            return self.image_labels
        elif features_type == 'text' or features_type == 'text_noproj':
            return self.text_labels
        else:
            raise ValueError(f'Unknown features_type: {features_type}')


if __name__ == '__main__':
    from tqdm import tqdm
    import clip

    _, preprocess = clip.load('ViT-B/32')
    dataset = ModifiedCocoDataset('/andromeda/datasets', 'train', preprocess, return_image=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for data in tqdm(loader):
        pass