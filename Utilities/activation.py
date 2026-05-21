'''
Python file containing activation functions and their derivatives for use in the neural network implementation.
'''

import numpy as np

# Get the sigmoid activation function
class Sigmoid():

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return Sigmoid.forward(x) * (1 - Sigmoid.forward(x))


# Get the ReLU activation
class ReLU():

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, 0)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)
            

# Get the tanh activation
class Tanh():

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2

# Softmax function
class Softmax():

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        x = x.T                                                    
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))       
        return (exp_x / np.sum(exp_x, axis=1, keepdims=True)).T      

    @staticmethod
    def backward(s: np.ndarray, dL_ds: np.ndarray) -> np.ndarray:
        s     = s.T       
        dL_ds = dL_ds.T   

        batch_size = s.shape[0]   
        dL_dz = np.zeros_like(s)

        for i in range(batch_size):
            sv       = s[i]                              
            J        = np.diagflat(sv) - np.outer(sv, sv) 
            dL_dz[i] = J @ dL_ds[i]                       

        return dL_dz.T  