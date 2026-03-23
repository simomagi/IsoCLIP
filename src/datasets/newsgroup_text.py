import pickle
from pathlib import Path
from typing import Union, Dict, List

from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT

# Mapping of class names to descriptive templates for each category in the dataset
class_to_template = {
    'alt.atheism': 'Atheism, philosophy, and the absence of belief in deities.',
    'comp.graphics': 'Computer graphics, rendering, and visual technologies.',
    'comp.os.ms-windows.misc': 'Microsoft Windows: software, settings, and troubleshooting.',
    'comp.sys.ibm.pc.hardware': 'IBM PC hardware: components, peripherals, and systems.',
    'comp.sys.mac.hardware': 'Mac hardware: devices, accessories, and configurations.',
    'comp.windows.x': 'X Windows: graphical interfaces and Unix system configuration.',
    'misc.forsale': 'Buying and selling products like electronics and furniture.',
    'rec.autos': 'Automobiles: car models, maintenance, and industry updates.',
    'rec.motorcycles': 'Motorcycles: bike models, culture, and maintenance tips.',
    'rec.sport.baseball': 'Baseball: games, teams, players, and statistics.',
    'rec.sport.hockey': 'Hockey: NHL, teams, games, and techniques.',
    'sci.crypt': 'Cryptography: encryption, security, and data protection.',
    'sci.electronics': 'Electronics: circuits, components, and device technology.',
    'sci.med': 'Medicine: healthcare, treatments, and medical research.',
    'sci.space': 'Space: exploration, astronomy, and scientific discoveries.',
    'soc.religion.christian': 'Christianity: beliefs, theology, and practices.',
    'talk.politics.guns': 'Gun politics: control, rights, and related issues.',
    'talk.politics.mideast': 'Middle East politics: conflicts, diplomacy, and events.',
    'talk.politics.misc': 'Politics: issues, governance, and international relations.',
    'talk.religion.misc': 'Religion and spirituality: diverse beliefs and philosophies.',
}


class NewsGroupDataset(Dataset):
    """
    A PyTorch Dataset class for the 20 Newsgroups dataset. This dataset is commonly used for text classification tasks.

    Attributes:
        split (str): The dataset split ('train', 'test', 'all', 'query').
        dataset (sklearn.utils.Bunch): The dataset loaded from `fetch_20newsgroups`.
        summarized_texts (dict, optional): A dictionary containing summarized texts for each review, if available.
        labels (List[int]): A list of sentiment labels corresponding to the reviews.
        class_to_template (dict): A dictionary mapping class names to descriptive template sentences.
    """

    def __init__(self, split: str):
        """
        Initializes the NewsGroupDataset by loading the data for the specified split and optionally loading the summarized texts.

        Args:
            split (str): The split to use ('train', 'test', 'all', 'query').

        Raises:
            ValueError: If the split is not 'train', 'test', 'all', or 'query'.
        """
        assert split in ['train', 'test', 'all', 'query'], f'Unknown split: {split}'
        self.split = split

        # Load the full dataset for 'train', 'test', or 'all'
        if split in ['train', 'test', 'all']:
            self.dataset = fetch_20newsgroups(subset=split)

            # Attempt to load the summarized texts if available
            try:
                with open(PROJECT_ROOT / 'data' / 'summarized_texts' / 'newsgroup_text' / split /
                          "meta-llama_Llama-3.2-1B-Instruct" / "summarized_text.pkl", 'rb') as f:
                    self.summarized_texts = pickle.load(f)
                print("Summarized texts loaded successfully.")
            except FileNotFoundError:
                print(f"Summarized texts not found for 'newsgroup_text' {split}")
                self.summarized_texts = None

            # Map sentiments to numerical labels (0: negative, 1: positive)
            self.labels = self.dataset.target
        else:  # 'query' split
            dummy_dataset = fetch_20newsgroups(subset='train')
            self.target_names = dummy_dataset.target_names
            self.class_to_template = class_to_template
            self.labels = [self.target_names.index(class_name) for class_name in self.class_to_template]

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the 'text', 'text_name' (ID), and label of the sample.
        """
        data = {}

        if self.split in ['train', 'test', 'all']:
            caption = self.dataset.data[index]  # Full review text
            label = self.dataset.target[index]  # Sentiment label (0 or 1)
            data['long_text'] = caption.strip()

            # Use summarized text if available
            if self.summarized_texts:
                data['text'] = self.summarized_texts[str(index)]

            data['text_name'] = str(index)  # Use the index as the text name
            data['label'] = label
        else:  # 'query' split, return template sentences
            data['text'] = self.class_to_template[self.target_names[index]]
            data['text_name'] = self.target_names[index]
            data['label'] = self.labels[index]

        return data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples (either reviews or template sentences).
        """
        return len(self.dataset.data) if self.split in ['train', 'test', 'all'] else len(self.class_to_template)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels for the dataset.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of labels corresponding to the sentiment for each review.

        Raises:
            ValueError: If labels are requested for invalid splits ('train', 'val', 'test').
        """
        if self.split in ['train', 'val', 'test', 'all']:
            raise ValueError('Labels are not available for the train, val, or test splits')
        return self.labels
