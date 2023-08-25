import numpy as np
from yacs.config import CfgNode

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.init as init

from ...utils.text import get_special_token


class NicDecoder(nn.Module):
    """
    NicDecoder Decoder module for the Neural Image Captioning model.
    """

    def __init__(self, config, tokenizer):
        """
        Initializes the NicDecoder with the given configuration and tokenizer.

        Parameters
        ----------
        config : object
            Configuration object containing model hyperparameters.
        tokenizer : object
            Tokenizer object for converting text to tokens and vice versa.
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.special_tokens_ids = {
            k: tokenizer.token_to_id(v) for k, v in get_special_token("all").items()
        }

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Embedding(
                config.vocab_size,
                config.embedding_size,
                padding_idx=self.special_tokens_ids["padding_token"],
            ),
            nn.Dropout(config.embedding_dropout),
        )

        # Language modeling head
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.vocab_size),
            nn.Dropout(config.lm_dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            config.embedding_size,
            config.hidden_size,
            config.num_layers,
            dropout=config.lstm_dropout,
        )

        # Initialize weights
        self._init_weights()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _init_weights(self):
        """
        Initializes the weights of the model using Xavier uniform initialization.
        """
        init.xavier_uniform_(self.embedding[0].weight)
        init.xavier_uniform_(self.lm_head[0].weight)
        init.zeros_(self.lm_head[0].bias)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.zeros_(param)

    def _init_step(self, input):
        """
        Initializes the hidden and cell states of the LSTM using the input tensor.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple
            The hidden and cell states of the LSTM.
        """
        assert input.dim() == 3, f"Input tensor has dimensions {input.size()} but must have three dimensions. (L, N, D) Where L is the sequence length, N is the batch size, and D is the features size."
        
        hidden = torch.zeros(
            self.config.num_layers,
            input.size(1),  # batch_size
            self.config.hidden_size,
            device=self.device,
        )

        cell = torch.zeros(
            self.config.num_layers,
            input.size(1),  # batch_size
            self.config.hidden_size,
            device=self.device,
        )

        
        _, (hidden, cell) = self.lstm(input, (hidden, cell))

        return _, hidden, cell

    def step_func(self, step_input, step_hidden, step_cell):
        """
        Defines the step function for beam search.

        Parameters
        ----------
        step_input : torch.Tensor
            The input tensor for the current step. 

        step_hidden : torch.Tensor
            The hidden state tensor for the current step.

        step_cell : torch.Tensor
            The cell state tensor for the current step.

        Returns
        -------
        tuple
            The prediction, hidden, and cell tensors for the current step.
        """
        step_input = step_input.unsqueeze(0)
        step_embedded = self.embedding(step_input)

        step_output, (step_hidden, step_cell) = self.lstm(
            step_embedded, (step_hidden, step_cell)
        )
        step_prediction = self.lm_head(step_output.squeeze(0))
        return step_prediction, step_hidden, step_cell

    def _forward_train(
        self, image_features, ground_truth_captions, teacher_forcing_ratio=0.8
    ):
        """
        Performs the forward pass of the decoder during training.

        Parameters
        ----------
        image_features : torch.Tensor
            The image features tensor.
        ground_truth_captions : torch.Tensor
            The ground truth captions tensor.
        teacher_forcing_ratio : float, optional, default=0.5
            The probability of using teacher forcing.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        # Sequence length x batch size x vocab size
        if len(image_features.size()) == 2:
            image_features = image_features.unsqueeze(0)

        batch_size = image_features.size(1)
        seq_len = ground_truth_captions.size(1)
        vocab_size = self.config.vocab_size
        outputs = torch.zeros(seq_len, batch_size, vocab_size, device=self.device)

        # Initialize the hidden and cell states using the image features
        _, hidden, cell = self._init_step(image_features)

        # Use teacher forcing with a certain probability
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        # Start token
        input_tokens = torch.tensor(
            [self.special_tokens_ids["bos_token"]] * batch_size,
            dtype=torch.long,
            device=self.device,
        )

        for t in range(1, seq_len):
            step_prediction, hidden, cell = self._step_func(input_tokens, hidden, cell)
            outputs[t] = step_prediction

            if use_teacher_forcing:
                input_tokens = ground_truth_captions[:, t].to(torch.long)

            else:
                input_tokens = step_prediction.argmax(dim=1).to(torch.long)

        return outputs

    def _forward_inference(self, input):
        """
        Performs the forward pass of the decoder during inference.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        # Sequence length x batch size x vocab size
        if len(input.size()) == 2:
            input = input.unsqueeze(0)

        _, hidden, cell = self._init_step(input)

        # Initialize the BeamSearch module
        beam_search = BeamSearch(
            start_token_id=self.special_tokens_ids["bos_token"],
            end_token_id=self.special_tokens_ids["eos_token"],
            beam_size=self.config.beam_size,
            max_length=self.config.max_length,
            device=self.device,
        )

        # Start the beam search process
        sequences = beam_search.search(
            step_func=self._step_func,
            initial_states=(hidden, cell),
        )

        return sequences

class NicModel(pl.LightningModule):
    def __init__(self, tokenizer, use_encoder=False, config_file=None):
        super().__init__()

        config_file = config_file or "./nic_config.yaml"
        config = CfgNode(new_allowed=True)
        config.merge_from_file("flashcaps/models/nic/nic_config.yaml")

        self.inference_params = config.inference
        self.training_params = config.training

        if use_encoder:
            raise NotImplementedError

        else:
            self.encoder = nn.Sequential(
                nn.Linear(config.encoder.feature_size, config.decoder.embedding_size),
                nn.Dropout(config.encoder.proj_dropout),
            )

        self.decoder = NicDecoder(config.decoder, tokenizer)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.decoder.special_tokens_ids["padding_token"]
        )

        self.train_loss = []

    def forward(self, image, ground_truth_captions=None):
        image_features = self.encoder(image)

        if self.training:
            return self.decoder._forward_train(
                image_features,
                ground_truth_captions,
                self.training_params.teacher_forcing_ratio,
            )

        else:
            return self.decoder._forward_inference(image_features)

    def training_step(self, batch, batch_idx):
        image_features, ground_truth_captions, *_ = batch
        outputs = self.forward(image_features, ground_truth_captions)

        outputs = outputs.transpose(0, 2).transpose(0, 1)

        loss = self.loss_fn(outputs, ground_truth_captions.to(torch.long))
        self.log("train_loss", loss)

        self.train_loss.append(loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=self.training_params.learning_rate
        )
        return optimizer
    
    def on_train_epoch_end(self ):
        print('avg_train_loss', np.mean(self.train_loss))
        self.train_loss = []