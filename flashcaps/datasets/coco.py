"""
COCO Captions Dataset
======================

The COCO (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset. The COCO captions dataset specifically focuses on image captioning, where the goal is to generate descriptive captions for images. It contains over 120,000 images, each annotated with five different captions. These captions provide a rich description of the image contents, including the objects present, their attributes, and their relationships.

The COCO captions dataset is widely used for training and evaluating image captioning models. It provides a diverse set of images and captions that cover a wide range of object categories and scenes. This makes it a valuable resource for developing and testing image captioning algorithms.

Reference
---------
Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár,
"Microsoft COCO: Common Objects in Context," arXiv:1405.0312, 2014.
`Link to paper <https://arxiv.org/abs/1405.0312>`_

Website
-------
`COCO website <http://cocodataset.org/>`_
"""

from typing import List, Dict, Union

class CocoAnnotations:
    """
    A class for loading and formatting annotations from the COCO dataset.

    Attributes
    ----------
    annotations : List[Dict[str, List[str]], int]
        A list of dictionaries containing the image paths, captions and image ids.

    Methods
    -------
    load_annotations():
        Returns the formatted annotations.
    """

    def __init__(self, annFile: str):
        """
        Initialize the CocoAnnotations class.

        Parameters
        ----------
        annFile : str
            Path to the annotation file.
        """
        self.annotations = self._format_annotations(annFile)

    def _format_annotations(self, annFile: str) -> List[Dict[str, Union[str, List[str], int]]]:
        """
        Format the annotations into a list of dictionaries.

        This method takes an annotation file and formats it into a list of dictionaries of this form:
        [{'image_path': image_path, 'captions': [caption, caption, ...], 'image_id': 250006}, ...]

        Parameters
        ----------
        annFile : str
            Path to the annotation file.

        Returns
        -------
        List[Dict[str, List[str], int]]
            A list of dictionaries containing the image paths, captions and image_ids. 
        """
        from pycocotools.coco import COCO
        coco = COCO(annFile)
        formatted_annotations = []

        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            
            formatted_annotations.append({
                'image_path': img_info['file_name'],
                'captions': captions,
                'image_id': img_id
            })

        return formatted_annotations
    
    def load_annotations(self) -> List[Dict[str, List[str]]]:
        """
        Get the formatted annotations.

        Returns
        -------
        List[Dict[str, List[str], int]]
            A list of dictionaries containing the image paths and captions.
        """
        return self.annotations
