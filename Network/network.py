'''
Python file where I'll write the neural network
'''

import numpy as np
from Data.MNIST import MNISTLoader
from Network.denselayer import DenseLayer
from Utilities.loss import *
from Utilities.utilities import HP


# The neural network class
class NeuralNetwork():

    def __init__(self) -> None:
        
        # Components and parameters
        self.data = MNISTLoader()
        self.architecture_sizes : list[int] = HP.architecture_sizes
        self.activation_functions : list[str] = HP.activation_functions
        self.params : list[np.ndarray] = []

        # For dropout
        self.dropout_prob: float = HP.dropout_rate

        # To enable L2 regularization 
        self.enable_L2 : bool = False

        self.network : list[DenseLayer] = []


        # Initializing the network
        self.createNetwork()

    def enableL2Regularization(self) -> None:
        self.enable_L2 = True

    def createNetwork(self) -> None:
        for i in range(len(self.architecture_sizes) - 1):
            self.network.append(DenseLayer(input_dims = self.architecture_sizes[i], 
                                           output_dims = self.architecture_sizes[i + 1], 
                                           activation = self.activation_functions[i],
                                           enable_L2 = self.enable_L2,
                                           dropout_rate= self.dropout_prob if i < len(self.architecture_sizes) - 2 else 0.0))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        
        output = x
        # Forward pass through the network
        for layer in self.network:
            output = layer.forward(output, training=training)
        
        return output

    def backward(self, dL_dY, training_epoch: int, learning_rate: float) -> None:
        
        # Backward pass through the network
        dL_dy = dL_dY
        for layer in reversed(self.network):
            dL_dy = layer.backward(dL_dy)

        # Update the parameters of the network
        for layer in self.network:
            layer.updateParameters(learning_rate, training_epoch)

    def paramsList(self) -> None:
        for layer in self.network:
            self.params.append(layer.w)
            self.params.append(layer.b)