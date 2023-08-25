To load a dataset we have to provide an annotations `.json` file for the corresponding AnnotationsLoader. 
We can either load and use the images with `ImageLoader` or directly use the precomputed features with `FeatureLoader`
``` python
from flashcaps.datasets import CocoAnnotations, FeatureLoader, ImageCaptioningDataset

annotations_loader = CocoLoader('path/to/annotations')
annotations_loader = FeatureLoader('path/to/features')
dataset = ImageCaptioningDataset(annotations_loader, features_loader)
```

Once the dataset created, it is used to instantiate the tokenizer. The simple WordLevel tokenizer is provided as part of text utils. 
``` python

from flashcaps.utils.text import create_word_level_tokenizer

tokenizer = create_word_level_tokenizer(dataset.get_corpus())
```

In order to use the dataset for training, we need to batch the input features. For that a dataloader is needed. The dataloader will select one of the groundtruth captions and use a tokenizer to return an encoded version of it. The captions of all the batch are padded into a closest multiple of 4 of the max_length. 

```python

from flashcaps.utils.data import get_collate_fn
from torch.utils.data import DataLoader

collate_fn = get_collate_fn(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
```

