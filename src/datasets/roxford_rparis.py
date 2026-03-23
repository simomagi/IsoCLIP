import os
import pickle
import shutil
from pathlib import Path
from typing import Callable, List, Dict, Union

import PIL.Image
import numpy as np
from PIL import ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import PROJECT_ROOT

ImageFile.LOAD_TRUNCATED_IMAGES = True


def configdataset(dataset: str, dir_main: Path) -> Dict:
    """
    Configure the dataset by loading image lists and ground truth data.

    Args:
        dataset (str): The dataset name ('roxford5k' or 'rparis6k').
        dir_main (Path): The main directory where the dataset is stored.

    Returns:
        dict: A dictionary containing dataset configuration, including image paths, labels, and ground truth.

    Raises:
        ValueError: If the dataset is unknown.
    """
    DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m']

    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset == 'roxford5k' or dataset == 'rparis6k':
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

    elif dataset == 'revisitop1m':
        # loading imlist from a .txt file
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg


def config_imname(cfg: Dict, i: int) -> Path:
    """
    Generate the file path for an image in the dataset.

    Args:
        cfg (dict): The dataset configuration.
        i (int): The index of the image.

    Returns:
        Path: The file path to the image.
    """
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg: Dict, i: int) -> Path:
    """
    Generate the file path for a query image in the dataset.

    Args:
        cfg (dict): The dataset configuration.
        i (int): The index of the query image.

    Returns:
        Path: The file path to the query image.
    """
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


def read_imlist(imlist_fn: str) -> List[str]:
    """
    Read the image list from a file.

    Args:
        imlist_fn (str): The file name containing the image list.

    Returns:
        List[str]: A list of image names.
    """
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=None):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X number_of_queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    if kappas is None:
        kappas = []
    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)  # average precision for each query
    pr = np.zeros(len(kappas))  # precision at kappas
    prs = np.zeros((nq, len(kappas)))  # precision at kappas for each query
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


class ROxfordRParisDataset(Dataset):
    """
    A PyTorch Dataset for the ROxford5k and Rparis6k datasets. This dataset handles image retrieval tasks, where images
    are paired with queries for retrieval evaluations.

    Attributes:
        preprocess (callable): A function to preprocess the images.
        dataset (str): The name of the dataset ('roxford5k' or 'rparis6k').
        split (str): The dataset split ('gallery' or 'query').
        cfg (dict): Configuration containing image lists and ground truth data.
        image_paths (List[Path]): A list of image paths for the current split.
        image_names (List[str]): A list of image names for the current split.
        labels (List[int]): A list of labels for the images (used for retrieval).
    """

    def __init__(self, dataroot: Path, dataset: str, split: str, preprocess: Callable):
        """
        Initializes the ROxfordRParisDataset by loading the data for the specified split.

        Args:
            dataroot (Path): The root directory where the dataset is stored.
            dataset (str): The dataset name ('roxford5k' or 'rparis6k').
            split (str): The dataset split ('gallery' or 'query').
            preprocess (callable): The function to preprocess images.

        Raises:
            ValueError: If the dataset or split is unknown.
        """
        super().__init__()

        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError("Dataset should be `roxford5k` of `rparis6k`")

        if split not in ['gallery', 'query']:
            raise ValueError("Split should be 'gallery' or 'query'")

        self.preprocess = preprocess
        self.dataset = dataset
        self.split = split

        # Configure dataset and load image data
        self.cfg = configdataset(dataset, dataroot)
        if split == "gallery":
            self.cfg_distractors = configdataset('revisitop1m', dataroot)

        # Load image paths and names for the 'query' or 'gallery' split
        if split == 'query':
            self.image_paths = [Path(self.cfg['dir_images']) / (x + ".jpg") for x in self.cfg['qimlist']]
            self.image_names = self.cfg['qimlist']
        else:
            self.image_paths = [Path(self.cfg['dir_images']) / (x + ".jpg") for x in self.cfg['imlist']]
            self.image_names = self.cfg['imlist']
            self.find_existing_distractors()

        # Copy the ground truth file if not already present
        if not (PROJECT_ROOT / 'data' / 'roxford_rparis_gnds' / f'gnd_{dataset}.pkl').exists():
            (PROJECT_ROOT / 'data' / 'roxford_rparis_gnds').mkdir(exist_ok=True, parents=True)
            shutil.copy(self.cfg['gnd_fname'], PROJECT_ROOT / 'data' / 'roxford_rparis_gnds' / f'gnd_{dataset}.pkl')

        self.cfg_distractors = None
        self.labels = [-1] * len(self.image_paths)

    def find_existing_distractors(self):
        """
        Finds and appends distractor images from the 'revisitop1m' dataset to the image paths for gallery split.
        """
        images_path = self.cfg_distractors['dir_images']
        for subdir in tqdm(os.listdir(images_path), desc="Finding existing distractors"):
            for image in os.listdir(Path(images_path) / subdir):
                self.image_paths.append(Path(images_path) / subdir / image)
                self.image_names.append(image)

    def __getitem__(self, index: int) -> Dict[str, Union[PIL.Image.Image, str, int]]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the 'image' (processed image), 'image_name' (name of the image), and 'label' (image label).
        """
        image = PIL.Image.open(self.image_paths[index])
        processed_image = self.preprocess(image)
        image_name = self.image_names[index]
        label = self.labels[index]

        return {'image': processed_image,
                'image_name': image_name,
                'label': label}

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.image_paths)

    def get_labels(self, *args, **kwargs) -> List[int]:
        """
        Retrieves the labels for the dataset. Returns a list of -1 for each image in the gallery split.

        Args:
            *args, **kwargs: Additional arguments (not used here).

        Returns:
            list: A list of labels for the images in the dataset.
        """
        return self.labels

    @staticmethod
    def get_ground_truth(dataset_name: str) -> Dict:
        """
        Retrieves the ground truth relevance data for the dataset.

        Args:
            dataset_name (str): The name of the dataset ('roxford5k' or 'rparis6k').

        Returns:
            dict: The ground truth data for the dataset.
        """
        with open(PROJECT_ROOT / 'data' / 'roxford_rparis_gnds' / f'gnd_{dataset_name}.pkl', 'rb') as f:
            gnd = pickle.load(f)
            gnd = gnd['gnd']
        return gnd
