import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activ_function(activ_name):
    """
    Returns the specified activation function.

    Parameters
    ----------
    activ_name : str
        Name of the activation function.

    Returns
    -------
    nn.Module
        Activation function.

    Activation Functions
    ---------------------
    - 'relu': Rectified Linear Unit (ReLU). It replaces negative values with zero.
    - 'leaky_relu': Leaky ReLU. Similar to ReLU, but allows a small gradient for negative values.
    - 'sigmoid': Sigmoid function. It squashes the input values into the range [0, 1].
    - 'tanh': Hyperbolic Tangent (tanh). It squashes the input values into the range [-1, 1].
    - 'softmax': Softmax function. It converts the input values into a probability distribution.
    - 'gelu': Gaussian Error Linear Unit (GELU). It smooths the input values around zero.
    - 'swish': Swish function. It is a self-gated activation function.

    Raises
    ------
    ValueError
        If the specified activation function is not in the list.

    """
    activation_functions = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=1),
        'gelu': nn.GELU(),
        'swish': nn.SiLU()  # Swish is also known as SiLU (Sigmoid Linear Unit)
    }

    if activ_name not in activation_functions:
        raise ValueError(f"Activation function '{activ_name}' is not in the list. Available activation functions: {', '.join(activation_functions.keys())}.")

    return activation_functions[activ_name]

class BeamSearch:
    def __init__(self, start_token_id, end_token_id, beam_size, max_length, device):
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.beam_size = beam_size
        self.max_length = max_length
        self.repetition_penalty = 1.0
        self.device = device

    def _search_single(self, step_func, initial_state):
        beams = [(torch.tensor([self.start_token_id], dtype=torch.long, device=self.device), 0, initial_state)]
        completed_beams = []

        for step in range(self.max_length):
            new_beams = []
            for beam in beams:
                token, score, state = beam
                
                logits, new_hidden, new_cell = step_func(token[-1].unsqueeze(0), *state)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = probs.topk(self.beam_size, dim=-1)

                for i in range(self.beam_size):
                    new_token = top_indices[0, i].unsqueeze(0)
                    repetition_penalty = -self.repetition_penalty * (token == new_token.item()).sum().item() # Apply penalty for word repetition
                    new_score = score + torch.log(top_probs[0, i]) + repetition_penalty
                    new_state = (new_hidden, new_cell)

                    if new_token.item() == self.end_token_id:
                        completed_beams.append((torch.cat([token, new_token]), new_score.item(), new_state))
                    else:
                        new_beams.append((torch.cat([token, new_token]), new_score.item(), new_state))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # If no sentence with an EOS token is found, add an EOS token to the beams manually
        if not completed_beams:
            for beam in beams:
                token, score, state = beam
                completed_beams.append((torch.cat([token, torch.tensor([self.end_token_id], dtype=torch.long, device=self.device)]), score, state))

        completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)
        top_beam = completed_beams[0][0] if completed_beams else None
        return top_beam


    def search(self, step_func, initial_states):
        batch_size = initial_states[0].size(1)
        top_beams = []

        for i in range(batch_size):
            single_initial_state = (initial_states[0][:, i].unsqueeze(1).contiguous(), 
                                    initial_states[1][:, i].unsqueeze(1).contiguous())
            top_beam = self._search_single(step_func, single_initial_state)
            top_beams.append(top_beam)

        return top_beams
