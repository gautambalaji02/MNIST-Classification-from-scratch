'''
Python file for utility functions
'''

import numpy as np
from dataclasses import dataclass, field

# Weight initialization using Xavier method
def weightInitializer(n_in: int, n_out: int) ->np.ndarray:
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out)) 

# Adam optimizer function
def adam(dl_dw: np.ndarray, dl_db:np.ndarray, eta: float, delta: float, gamma: float, training_epoch: int, update_params: list = [0.0, 0.0, 0.0, 0.0]) -> list:
    
    # Giving the parameters for momentum and learning rate update
    [mw_k, mb_k, vw_k, vb_k] = update_params
    mw_k = delta * mw_k + (1 - delta) * dl_dw
    mb_k = delta * mb_k + (1 - delta) * dl_db

    vw_k = gamma * vw_k + (1 - gamma) * (dl_dw**2)
    vb_k = gamma * vb_k + (1 - gamma) * (dl_db**2)

    # the update params are updated
    update_params = [mw_k, mb_k, vw_k, vb_k]

    # To allow for learning
    mw_k = mw_k / (1 - delta**training_epoch)
    mb_k = mb_k / (1 - delta**training_epoch)
    vw_k = vw_k / (1 - gamma**training_epoch)
    vb_k = vb_k / (1 - gamma**training_epoch)

    # Update value for the weights and biasses
    del_w, del_b = eta * mw_k / (np.sqrt(vw_k) + 1e-9), eta * mb_k / (np.sqrt(vb_k) + 1e-9)

    return [del_w, del_b, update_params]


# Hyperparameters
@dataclass
class Hyperparameters:
   # Architecture and activation function choices
    architecture_sizes    : list[int] = field(default_factory=lambda: [784, 500, 250, 100, 10])
    activation_functions  : list[str] = field(default_factory=lambda: ['ReLU', 'ReLU', 'ReLU', 'Softmax'])
    
    # Training parameters
    batch_size: int = 250
    epochs: int = 15

    # Regularization parameters
    dropout_rate: float = 0.2
    lambda_L2: float = 1e-2

    # Adam optimizer parameters
    learning_rate: float = 3e-4
    adam_delta: float = 0.9
    adam_gamma: float = 0.999

    # Data split
    val_split: float = 0.1
    test_split: float = 0.2

# Creating the hyperparameters object
HP = Hyperparameters()