import os
import numpy as np

from tqdm import tqdm
from typing import List, Dict, Union, Tuple

from torch.utils.data import Dataset

class ImageCaptioningDataset(Dataset):
    """
    A dataset for image captioning tasks.

    This class provides a convenient way to load and format image captioning data.
    It takes an annotations loader and a data loader (image or feature) as input,
    and provides methods for retrieving the image data and captions.

    Parameters
    ----------
    annotations_loader : AnnotationsLoader
        The annotations loader.
    data_loader : FeatureLoader or ImageLoader
        The data loader (image or feature).

    Methods
    -------
    get_corpus() -> List[str]:
        Get a flat list containing the captions.
    display_statistics():
        Display the statistics on the dataset.

    Example
    -------
    >>> from flashcaps.datasets import CocoAnnotations, FeatureLoader
    >>> annotations_loader = CocoAnnotations("path/to/annotations.json")
    >>> data_loader = FeatureLoader("path/to/features")
    >>> dataset = ImageCaptioningDataset(annotations_loader, data_loader)
    >>> img, captions, idx = dataset[0]
    >>> print(captions)
    ['A man riding a horse.', 'A person on a horse.']
    """
    def __init__(self, annotations_loader, data_loader):
        self._annotations = annotations_loader.load_annotations()
        self._data_loader = data_loader
        self._data_loader.check_paths(self._annotations)

    def get_corpus(self) -> List[str]:
        """
        Get a flat list containing the captions.

        Returns
        -------
        List[str]
            A flat list containing the captions.
        """
        return [caption for item in tqdm(self._annotations) for caption in item['captions']]
    
    def display_statistics(self):
        """
        Display the maximum number of captions per image and the number of images.
        """
        num_images = len(self._annotations)
        num_captions = [len(item['captions']) for item in self._annotations]

        print(f"Number of images: {num_images}")
        print(f"Median number of captions per image: {np.median(num_captions)}")
        print(f"Maximum number of captions per image: {max(num_captions)}")
        print(f"Minimum number of captions per image: {min(num_captions)}")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[str], int]:
        """
        Get the image data and captions for the given index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Tuple[np.ndarray, List[str], int]
            The image data as a numpy array and the captions as a list of strings and the image id.
        """
        image_path = self._annotations[index]['image_path']
        captions = self._annotations[index]['captions']
        image_id = self._annotations[index]['image_id'] # We use the image_id in case of evaluation
        data = self._data_loader.load(image_path)
        return data, captions, image_id

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self._annotations)

class FeatureLoader:
    def __init__(self, root_folder: str):
        self.root_folder = root_folder

    def _convert_path_to_npy(self, image_name: str) -> str:
        """
        Convert an image file name to a feature file path.

        Parameters
        ----------
        image_name : str
            The image file name.
        
        Returns
        -------
        str
            The feature file path.
        """
        base_name, _ = os.path.splitext(image_name)
        feature_name = f"{base_name}.npy"
        feature_path = os.path.join(self.root_folder, feature_name)
        return feature_path

    def load(self, features_path: str) -> np.ndarray:
        """
        Load the features from the given features path.

        This method converts the given features path to the corresponding .npy file path,
        and then loads the features from the .npy file.

        Parameters
        ----------
        features_path : str
            The path to the features file.

        Returns
        -------
        np.ndarray
            The features loaded from the .npy file.
        """       
        features_path = self._convert_path_to_npy(features_path)
        return np.load(features_path)
    

    def check_paths(self, captions: List[Dict[str, Union[str, List[str]]]]) -> bool:
        """
        Check if all the features for the images are present.

        Parameters
        ----------
        captions : List[Dict[str, Union[str, List[str]]]]
            A list of dictionaries containing the image paths and captions.

        Returns
        -------
        bool
            True if all the features are present, False otherwise.
        """
        for item in captions:
            image_path = item['image_path']
            feature_path = self._convert_path_to_npy(image_path)
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
                
        return True
