'''
Python file for dense layer of the neural network
'''

import numpy as np
from Utilities.utilities import *
from Utilities.activation import *

# The dense layer class
class DenseLayer():

    def __init__(self, input_dims: int, output_dims: int, activation: str = 'ReLU', enable_L2: bool = False, dropout_rate: float = 0.0) -> None:
        
        # Variables to define the layer structure
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.activation = activation
        self.enable_L2 = enable_L2
        self.lambda_L2 : float = HP.lambda_L2
        self.dropout_rate = dropout_rate
        self.mask = None

        # Components for the layer for forward prop
        self.x : np.ndarray
        self.w : np.ndarray = weightInitializer(self.input_dims, self.output_dims)
        self.b : np.ndarray = np.zeros(shape = (self.output_dims, 1))
        self.z : np.ndarray
        self.y : np.ndarray

        # Gradients for the layer for backprop
        self.dL_dw : np.ndarray
        self.dL_db : np.ndarray

        # Gradients for the batch norm params
        self.gamma : np.ndarray = np.ones((self.output_dims, 1))
        self.beta  : np.ndarray = np.zeros((self.output_dims, 1))

        # Parameters for Adam optimizer
        self.adam_params = [0.0, 0.0, 0.0, 0.0]
        self.adam_params_bn = [0.0, 0.0, 0.0, 0.0]

    # Forward pass through the layer
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:

        # Forward pass
        self.x = x
        self.z = self.w.T @ self.x + self.b

        # Batch normalization
        self.mu     = self.z.mean(axis=1, keepdims=True)      
        self.var    = self.z.var(axis=1, keepdims=True)     
        self.z_norm = (self.z - self.mu) / np.sqrt(self.var + 1e-8)  
        self.z_bn   = self.gamma * self.z_norm + self.beta    

        # Activation functions for non linearity
        if self.activation == 'ReLU':
            self.y = ReLU.forward(self.z_bn)
        elif self.activation == 'Sigmoid':
            self.y = Sigmoid.forward(self.z_bn)
        elif self.activation == 'Tanh':
            self.y = Tanh.forward(self.z_bn)
        elif self.activation == 'Softmax':
            self.y = Softmax.forward(self.z_bn)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        # Including dropout if the following conditions are met
        if self.dropout_rate > 0.0 and self.activation != 'Softmax' and training:
            self.mask = (np.random.rand(*self.y.shape) > self.dropout_rate).astype(float)
            self.y = (self.mask * self.y) / (1 - self.dropout_rate)
        # Returning the output of the layer
        return self.y
    
    # Backward pass through the layer
    def backward(self, dL_dy: np.ndarray) -> np.ndarray:

        if self.dropout_rate > 0.0 and self.mask is not None:
            dL_dy = (self.mask * dL_dy) / (1 - self.dropout_rate)

        if self.activation == 'ReLU':
            dL_dzbn = dL_dy * ReLU.backward(self.z_bn)
        elif self.activation == 'Sigmoid':
            dL_dzbn = dL_dy * Sigmoid.backward(self.z_bn)
        elif self.activation == 'Tanh':
            dL_dzbn = dL_dy * Tanh.backward(self.z_bn)
        elif self.activation == 'Softmax':
            dL_dzbn = Softmax.backward(self.y, dL_dy)   
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
        # batch normalization backward pass
        batch_size = self.x.shape[1]

        # Gradients for gamma and beta
        self.dL_dgamma = np.sum(dL_dzbn * self.z_norm, axis=1, keepdims=True)
        self.dL_dbeta  = np.sum(dL_dzbn, axis=1, keepdims=True)

        # Gradient w.r.t z_norm
        dL_dznorm = dL_dzbn * self.gamma

        # Gradient w.r.t z (through the normalization)
        std_inv = 1 / np.sqrt(self.var + 1e-8)
        dL_dz   = (1 / batch_size) * std_inv * (
                    batch_size * dL_dznorm
                    - np.sum(dL_dznorm, axis=1, keepdims=True)
                    - self.z_norm * np.sum(dL_dznorm * self.z_norm, axis=1, keepdims=True)
                )

        self.dL_dw  = self.x @ dL_dz.T

        if self.enable_L2:
            self.dL_dw += self.lambda_L2 * self.w
        self.dL_db = np.sum(dL_dz, axis=1, keepdims=True)
        dL_dx = self.w @ dL_dz

        return dL_dx
    
    # Update the weights and biases of the layer using the computed gradients
    def updateParameters(self, learning_rate: float, training_epoch: int) -> None:

        # Adam optimizer updates for weights and biases
        changes = adam(self.dL_dw, self.dL_db, learning_rate, 
                       delta=HP.adam_delta, gamma=HP.adam_gamma, 
                       training_epoch=training_epoch, update_params=self.adam_params)
        self.w -= changes[0]
        self.b -= changes[1]
        self.adam_params = changes[2]

        # Add gamma and beta updates
        changes_gb = adam(self.dL_dgamma, self.dL_dbeta, learning_rate, 
                        delta=HP.adam_delta, gamma=HP.adam_gamma, 
                        training_epoch=training_epoch, update_params=self.adam_params_bn)
        self.gamma -= changes_gb[0]
        self.beta  -= changes_gb[1]
        self.adam_params_bn = changes_gb[2]
