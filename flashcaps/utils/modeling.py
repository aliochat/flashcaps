import torch.nn as nn

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