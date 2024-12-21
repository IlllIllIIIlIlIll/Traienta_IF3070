import numpy as np
import pickle
from typing import Dict, Tuple, List

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    Parameters:
    -----------
    var_smoothing: float, default=1e-9
        Portion of the largest variance of all features that is added to variances for calculation stability.

    Attributes:
    -----------
    classes_: array, shape (n_classes,)
        Unique class labels.
    class_priors_ : array, shape (n_classes,)
        Probability of each class.
    features_ : int
        Number of features in the dataset.
    means_ : array, shape (n_classes, n_features)
        Mean of each feature for each class.
    vars_ : array, shape (n_classes, n_features)
        Variance of each feature for each class.

    Methods:
    --------
    fit(X: np.ndarray, y: np.ndarray)
        Fit the Gaussian Naive Bayes model according to the given training data.
    predict(X: np.ndarray) -> np.ndarray
        Perform classification on an array of test vectors X.
    
    """
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing: float = var_smoothing
        self.classes_: np.ndarray = None
        self.class_priors_: np.ndarray = None
        self.features_: int = None
        self.means_: np.ndarray = None
        self.vars_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Get unique class labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Get number of features
        self.features_ = X.shape[1]

        # Initialize arrays
        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, self.features_))
        self.vars_ = np.zeros((n_classes, self.features_))

        # Calculate maximum variance for each feature
        var_smoothing_factor = self.var_smoothing * np.max(np.var(X, axis=0))

        # For every class
        for idx, c in enumerate(self.classes_):
            mask = (y == c)
            X_c = X[mask]

            # Compute the prior probability
            self.class_priors_[idx] = np.mean(mask)
            
            # Compute mean and variance (with smoothing factor) for each feature 
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + var_smoothing_factor

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check if the model has been trained
        if self.classes_ is None:
            raise ValueError("Model has not been trained yet. Please call the fit method first.")
        
        # Check if the number of features in X is the same as in the training data
        if X.shape[1] != self.features_:
            raise ValueError("Number of features in X does not match the number of features in the training data.")
        
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x: np.ndarray) -> int:
        """Predict the class label for a single sample"""
        # Initialize the list of posterior likelihood
        log_likelihoods = np.zeros(len(self.classes_))

        # Compute the posterior probability for each class
        for idx in range(len(self.classes_)):
            # Add log prior probability
            log_likelihoods[idx] = np.log(self.class_priors_[idx] + 1e-9)  # epsilon
        
            # Add log likelihood
            log_likelihoods[idx] += np.sum(self._log_pdf(idx, x))

        # Return the class with the highest posterior probability
        return self.classes_[np.argmax(log_likelihoods)]
    
    def _log_pdf(self, idx: int, x: np.ndarray) -> np.ndarray:
        """Compute the log probability density function for a Gaussian distribution"""
        # Add a small value to avoid division by zero
        epsilon = 1e-9

        var = self.vars_[idx] + epsilon
        mean = self.means_[idx]
        
        return -0.5 * (np.log(var) + np.log(2 * np.pi) + (x - mean) ** 2 / var)
    
    def save(self, path: str):
        model_attributes = {
            'classes_': self.classes_,
            'class_priors_': self.class_priors_,
            'features_': self.features_,
            'means_': self.means_,
            'vars_': self.vars_
        }

        with open(path, 'wb') as f:
            pickle.dump(model_attributes, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            model_attributes = pickle.load(f)
        
        self.classes_ = model_attributes['classes_']
        self.class_priors_ = model_attributes['class_priors_']
        self.features_ = model_attributes['features_']
        self.means_ = model_attributes['means_']
        self.vars_ = model_attributes['vars_']
