'''
The main file where the training and testing of the network will be done.
'''

import numpy as np
import matplotlib.pyplot as plt

from Network.network import NeuralNetwork
from Utilities.loss import CrossEntropyLoss
from Utilities.utilities import HP

# The MNIST classifcation class
class MNISTClassification:

    def __init__(self) -> None:
        
        # Initialize the neural network
        self.network = NeuralNetwork()
        
        # Number of epochs for training
        self.epochs : int = HP.epochs

        self.batch_size : int = HP.batch_size

        # Counter for increasing steps for adam optimizer
        self.counter : int = 1

        # Data stored for displaying
        self.train_loss_graph : list = []
        self.val_loss_graph : list = []

    def train(self) -> None:
    
        sample_size = self.network.data.Xtrain.shape[0]

        # Get the validation set
        X_val = self.network.data.Xval.T
        y_val = self.network.data.Yval

        for epoch in range(self.epochs):

            # Shuffle the train set
            indices = np.random.permutation(sample_size)
            X_train = self.network.data.Xtrain[indices]
            y_train = self.network.data.Ytrain[indices]
            print(f"\n --------------- Epoch {epoch + 1}/{self.epochs} --------------- ")

            for batch_idx in range(0, sample_size, self.batch_size):
                
                # NumPy handles the last batch automatically
                x = X_train[batch_idx : batch_idx + self.batch_size]
                y = y_train[batch_idx : batch_idx + self.batch_size]

                # Forward pass
                y_pred = self.network.forward(x.T, training=True).T

                # Loss
                loss = CrossEntropyLoss.forward(y_true=y, y_pred=y_pred)
                print(f"Samples: {min(batch_idx + self.batch_size, sample_size)}/{sample_size}, Loss: {loss}")

                # Backward pass
                dL_dy = CrossEntropyLoss.backward(y_true=y, y_pred=y_pred).T
                self.network.backward(dL_dY=dL_dy, training_epoch=self.counter, learning_rate=HP.learning_rate)

                self.counter += 1
            
            # Validation data testing
            y_val_pred = self.network.forward(X_val, training=False).T

            val_loss = CrossEntropyLoss.forward(y_true = y_val, y_pred = y_val_pred)

            print(f"\nValidation Loss : {val_loss}")

            # Storing the data
            self.train_loss_graph.append(loss)
            self.val_loss_graph.append(val_loss)

    # TO test the accuracy and loss of the network
    def test(self, x_test = None, y_test = None) -> None:
        
        # The data points for testing
        if (x_test == None) | (y_test == None):
            x_test = self.network.data.Xtest.T
            y_test = self.network.data.Ytest

        # Forward pass
        y_pred = self.network.forward(x_test, training=False).T

        # Loss obtained
        loss = CrossEntropyLoss.forward(y_true = y_test, y_pred = y_pred)

        # Predicted class is the argmax of the softmax output
        predicted = np.argmax(y_pred, axis=1)   # (n_test,)
        actual    = np.argmax(self.network.data.Ytest, axis=1) # (n_test,) — if one-hot
        
        accuracy = np.mean(predicted == actual) * 100
        print(f"Test loss: {loss}, Test Accuracy: {accuracy:.2f}%")

    # To display the data obtained for analysis
    def display(self) -> None:
        
        plt.figure(figsize = (10, 12))
        plt.plot(range(1, self.epochs + 1), self.train_loss_graph)
        plt.plot(range(1, self.epochs + 1), self.val_loss_graph)
        plt.title("Loss Graph")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Train Loss", "Validation Loss"])
        plt.savefig("Results/train_val_loss_plot.png")
        plt.show()

# The main function call
if __name__ == "__main__":
    mnist_classifier = MNISTClassification()
    mnist_classifier.network.enableL2Regularization()
    mnist_classifier.train()

    mnist_classifier.test()
    mnist_classifier.display()