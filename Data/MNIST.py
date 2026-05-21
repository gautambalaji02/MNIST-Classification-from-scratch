'''
Python file to load the MNIST dataset and preprocess it for training and testing. 
The dataset is loaded from a local directory, and the images are normalized to have pixel values between 0 and 1. 
The labels are also one-hot encoded for use in classification tasks.
'''

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from Utilities.utilities import HP

# Class to load the MNIST data
class MNISTLoader:

    # Initializing the class
    def __init__(self) -> None:

        self.Xtrain : np.ndarray
        self.Xtest  : np.ndarray
        self.Ytrain : np.ndarray
        self.Ytest  : np.ndarray
        self.Xval   : np.ndarray
        self.Yval   : np.ndarray
        self.MNIST = fetch_openml('mnist_784', version=1)

        self.loadData()

    def loadData(self) -> None:
        
        # Loading data from the library
        mnist_data = self.MNIST.data.to_numpy()
        mnist_labels = self.MNIST.target.to_numpy()

        # Passing labels through a one hot encoder
        one_hot_labels = self.oneHotEncoder(mnist_labels)

        # Normalizing the data
        mnist_data_norm = self.normalizeData(mnist_data)

        self.splitData(mnist_data_norm, one_hot_labels, HP.test_split, HP.val_split)
        

    def oneHotEncoder(self, labels: np.ndarray) -> np.ndarray:
        n_classes = len(np.unique(labels))
        one_hot = np.zeros((labels.shape[0], n_classes))
        for i, label in enumerate(labels):
            one_hot[i, int(label)] = 1
        return one_hot

    def normalizeData(self, data: np.ndarray) -> np.ndarray:
        return data / 255.0

    def splitData(self, data, labels, test_split: float, val_split: float) -> None:
        
        xtrain, self.Xtest, ytrain, self.Ytest = train_test_split(data, labels, test_size=test_split, random_state=42)
        self.Xtrain, self.Xval, self.Ytrain, self.Yval = train_test_split(xtrain, ytrain, test_size=val_split/(1-test_split), random_state=42)


    