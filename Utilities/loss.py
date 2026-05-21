'''
Python file to get the losses, depending on what is achieved
'''
import numpy as np

# The Mean square error function
class MSELoss:

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        n = 2 * y_true.shape[0]
        return (1 / n) * np.sum((y_pred - y_true) ** 2)
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = 2 * y_true.shape[0]
        return (2 / n) * (y_pred - y_true)

class CrossEntropyLoss:

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        
        loss = -np.sum(y_true * np.log(y_pred + 1e-9), axis=1)
        return np.mean(loss) 
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (-y_true / (y_pred + 1e-9)) / n