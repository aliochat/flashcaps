import torch
from random import choice

def get_collate_fn(tokenizer, sampling_strategy='random'):
    """
    Factory function that creates and returns a customized collate function for batching.

    Parameters
    ----------
    tokenizer : Tokenizer
        The tokenizer to be used for encoding the captions.
    sampling_strategy : str, optional
        The strategy for sampling captions, by default 'random'.

    Returns
    -------
    function
        The customized collate function.
    """
    def collate_fn(batch):
        """
        Collate function that batches the data and encodes the captions.

        Parameters
        ----------
        batch : list
            The batch of data to be collated.

        Returns
        -------
        tuple
            The batched features, token IDs, and indices.
        """
        feats, captions, idx = zip(*batch)
        
        if sampling_strategy == 'random':
            captions = [choice(c) for c in captions]
        else:
            raise NotImplementedError('Only random sampling strategy is supported for now!')
        
        tokenized_captions = tokenizer.encode_batch(captions)
        token_ids = [item.ids for item in tokenized_captions]

        return torch.Tensor(feats), torch.Tensor(token_ids), idx
    
    return collate_fn
