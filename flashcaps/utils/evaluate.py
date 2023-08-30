import torch
from tqdm.notebook import tqdm

def generate_predictions(val_loader, model, tokenizer, bs_scorer):
    """
    Generate predictions for the validation set.
    
    Parameters
    ----------
    val_loader : DataLoader
        DataLoader object containing the validation data.
    model : nn.Module
        The pre-trained model to use for encoding and decoding.
    tokenizer : Tokenizer
        Tokenizer to decode model outputs into human-readable text.
    bs_scorer : object
        Scorer object with search functionality to score sequences.

    Returns
    -------
    id_pred_list : List[Dict]
        List of dictionaries, each containing an image ID and its corresponding caption.
    """
    id_pred_list = []

    for img_features, _, idx in tqdm(val_loader, desc="Generating predictions"):

        with torch.no_grad():
            encoded_features = model.encoder(img_features.to('cuda'))
            _, *init_states = model.decoder._init_step(encoded_features.unsqueeze(0))

            outputs = bs_scorer.search(
                step_function=model.decoder.step_func,
                init_states=(init_states),
                batch_size=img_features.shape[0]
            )

            decoded_predictions = tokenizer.decode_batch(outputs[0].tolist())
            id_pred_dict = [{'image_id': id, 'caption': caption} for id, caption in zip(idx, decoded_predictions)]
            id_pred_list.extend(id_pred_dict)

    return id_pred_list

