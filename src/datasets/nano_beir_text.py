import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Dict, List

from .utils import PROJECT_ROOT

# Mapping lowercased dataset names to their full names
lower_to_name = {
    'nanoclimatefever': 'NanoClimateFEVER',
    'nanodbpedia': 'NanoDBPedia',
    'nanofever': 'NanoFEVER',
    'nanonfcorpus': 'NanoNFCorpus',
    'nanonq': 'NanoNQ',
    'nanoscidocs': 'NanoSCIDOCS',
    'nanoscifact': 'NanoSciFact'
}


class NanoBEIRDataset(Dataset):
    """
    A PyTorch Dataset for the NanoBEIR dataset. This dataset handles querying and gallery splits for different subsets of the NanoBEIR dataset.

    Attributes:
        split (str): The split type ('query' or 'gallery').
        name (str): The name of the dataset (e.g., 'nanoclimatefever', 'nanodbpedia', etc.).
        gallery (pd.DataFrame): The DataFrame containing the gallery data.
        qrels (pd.DataFrame): The DataFrame containing the relevance data for queries.
        queries (pd.DataFrame): The DataFrame containing the queries.
        summarized_texts (dict, optional): A dictionary containing the summarized texts for the gallery split.
        dataframe (pd.DataFrame): The DataFrame for the current split (either queries or gallery).
    """

    def __init__(self, split: str, name: str):
        """
        Initializes the NanoBEIRDataset by loading the data for the specified split.

        Args:
            split (str): The split to load ('query' or 'gallery').
            name (str): The dataset name, such as 'nanoclimatefever', 'nanodbpedia', etc.

        Raises:
            ValueError: If the split or name is invalid.
        """
        assert name in ['nanoclimatefever', 'nanodbpedia', 'nanofever', 'nanonfcorpus', 'nanonq',
                        'nanoscidocs', 'nanoscifact'], f'Unknown dataset: {name}'
        upper_name = lower_to_name[name]
        assert split in ['query', 'gallery'], f'Unknown split: {split}'
        self.split = split
        self.name = name

        # Load data for gallery, queries, and qrels
        self.gallery = pd.read_parquet(f"hf://datasets/zeta-alpha-ai/{upper_name}/corpus/train-00000-of-00001.parquet")
        self.qrels = pd.read_parquet(f"hf://datasets/zeta-alpha-ai/{upper_name}/qrels/train-00000-of-00001.parquet")
        self.queries = pd.read_parquet(f"hf://datasets/zeta-alpha-ai/{upper_name}/queries/train-00000-of-00001.parquet")

        # If split is 'gallery', load the summarized texts (if available)
        if split == 'gallery':
            try:
                with open(PROJECT_ROOT / 'data' / 'summarized_texts' / lower_to_name[name] / split /
                          "meta-llama_Llama-3.2-1B-Instruct" / "summarized_text.pkl", 'rb') as f:
                    self.summarized_texts = pickle.load(f)
                print("Summarized texts loaded successfully.")
            except FileNotFoundError:
                print(f"Summarized texts not found for {name} {split}")
                self.summarized_texts = None

        # Set the DataFrame based on the split ('query' or 'gallery')
        if split == 'query':
            self.dataframe = self.queries
        else:
            self.dataframe = self.gallery

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'text', 'text_name' (ID), and other relevant data.
        """
        data = {}
        if self.split == 'gallery':
            caption = self.dataframe['text'][index]
            data['long_text'] = caption.strip()
            if self.summarized_texts:
                data['text'] = self.summarized_texts[self.dataframe['_id'][index]]
            data['text_name'] = self.dataframe['_id'][index]
        else:
            data['text'] = self.dataframe['text'][index]
            data['text_name'] = self.dataframe['_id'][index]
        return data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataframe (either query or gallery).
        """
        return len(self.dataframe)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves fake labels for the dataset (used for compatibility with other datasets).

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of zeros corresponding to the number of samples in the dataset.
        """
        return [0] * len(self.dataframe)

    def get_ground_truth(self) -> torch.Tensor:
        """
        Retrieves the ground truth relevance information for the queries.

        Returns:
            torch.Tensor: A tensor representing the ground truth relevance for each query and gallery item.
        """
        ground_truth_tensor = torch.zeros(len(self.queries), len(self.gallery))
        for i in range(len(self.queries)):
            gallery_ids = self.qrels[self.qrels['query-id'] == self.queries['_id'][i]]['corpus-id']
            positive_idx = self.gallery[self.gallery['_id'].isin(gallery_ids)].index
            ground_truth_tensor[i, positive_idx] = 1

        return ground_truth_tensor.int()
