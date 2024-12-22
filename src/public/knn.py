from typing import Literal, Union, Optional
import numpy as np
from numpy.typing import NDArray
from collections import Counter
from joblib import dump, load

class KNN:
    """
    K-Nearest Neighbors Classifier implementation from scratch
    
    Parameters:
    -----------
    n_neighbors: int, default=5
        Number of neighbors to use
    metric: str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'minkowski')
    p: int, default=2
        Power parameter for Minkowski metric

    Attributes:
    -----------
    X_train: np.ndarray
        Training data
    y_train: np.ndarray
        Target values

    Methods:
    --------
    fit(X: np.ndarray, y: np.ndarray) -> 'KNN'
        Fit the model using X as training data and y as target values
    predict(X: np.ndarray) -> np.ndarray
        Predict the class labels for the provided data

    """
    
    def __init__(
        self, 
        n_neighbors: int = 5,
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.X_train: Optional[NDArray] = None
        self.y_train: Optional[NDArray] = None

    def _calculate_distance(self, x1: NDArray, x2: NDArray) -> float:
        """Calculate distance between two points using specified metric"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1/self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def fit(self, X: Union[NDArray, np.ndarray], y: Union[NDArray, np.ndarray]) -> 'KNN':
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X: Union[NDArray, np.ndarray]) -> NDArray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        X = np.asarray(X)
        predictions = []
        
        for x in X:
            distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)

    def save(self, filename: str) -> None:
        dump(self, filename)

    @classmethod
    def load(cls, filename: str) -> 'KNN':
        return load(filename)