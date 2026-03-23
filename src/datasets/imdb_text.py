import pickle
from pathlib import Path
from typing import Union, Dict, List

import pandas as pd
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT


class IMDBTextDataset(Dataset):
    """
    A PyTorch Dataset class for the IMDB Text dataset. This dataset contains movie reviews and their associated sentiment labels
    (positive or negative). The dataset can be used for tasks like sentiment analysis or text classification.

    Attributes:
        split (str): The dataset split ('all' or 'query').
        summarized_texts (dict, optional): A dictionary containing summarized texts for each review, if available.
        dataset_path (Path): Path to the IMDB dataset.
        dataframe (pd.DataFrame): The DataFrame containing the IMDB dataset.
        sentiments (List[str]): A list of unique sentiments (positive and negative).
        labels (List[int]): A list of labels corresponding to the sentiments (0 for negative, 1 for positive).
        template_sentences (List[str]): Template sentences for the query split (positive and negative review examples).
    """

    def __init__(self, dataroot: Union[str, Path], split: str):
        """
        Initializes the IMDBTextDataset.

        Args:
            dataroot (Union[str, Path]): The root directory where the IMDB dataset is stored.
            split (str): The split to use ('all' or 'query').

        Raises:
            ValueError: If the split is not 'all' or 'query'.
        """
        dataroot = Path(dataroot)
        assert split in ['all', 'query'], f'Unknown split: {split}'
        self.split = split

        if split in ['all']:
            # Attempt to load the summarized texts for the "all" split
            try:
                with open(PROJECT_ROOT / 'data' / 'summarized_texts' / 'imdb_text' / split /
                          "meta-llama_Llama-3.2-1B-Instruct" / "summarized_text.pkl", 'rb') as f:
                    self.summarized_texts = pickle.load(f)
                print("Summarized texts loaded successfully.")
            except FileNotFoundError:
                print(f"Summarized texts not found for 'imdb_text' {split}")
                self.summarized_texts = None

            # Load the IMDB dataset
            dataset_path = dataroot / 'IMDB_Reviews'
            self.dataset_path = dataset_path
            self.split = split

            # Read the IMDB dataset CSV into a DataFrame
            self.dataframe = pd.read_csv(dataset_path / 'IMDB_Dataset.csv')
            self.sentiments = list(self.dataframe['sentiment'].unique())
            self.labels = [self.sentiments.index(sentiment) for sentiment in self.dataframe['sentiment']]
        else:  # query
            self.split = split
            self.labels = list(range(2))  # Sentiment labels (0: negative, 1: positive)
            self.template_sentences = [
                'a positive review of a movie.', 'a negative review of a movie.']

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        """
        Retrieves a data sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the review text, text name (ID), and label.
        """
        data = {}
        if self.split in ['all']:
            caption = self.dataframe['review'][index]  # Full review text
            label = self.sentiments.index(self.dataframe['sentiment'][index])  # Sentiment label
            data['long_text'] = caption.strip()

            if self.summarized_texts:
                # If summarized texts are available, use them
                data['text'] = self.summarized_texts[str(index)]
            data['text_name'] = str(index)  # Use the index as the text name
            data['label'] = label
        else:
            # For the 'query' split, return template sentences
            data['text'] = self.template_sentences[index]
            data['text_name'] = self.template_sentences[index]
            data['label'] = self.labels[index]

        return data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.dataframe) if self.split in ['all'] else len(self.template_sentences)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels for the dataset.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of labels corresponding to the sentiment for each review.

        Raises:
            ValueError: If labels are requested for the 'train', 'val', or 'test' splits.
        """
        if self.split in ['train', 'val', 'test', 'all']:
            raise ValueError('Labels are not available for the train and val splits')
        return self.labels
