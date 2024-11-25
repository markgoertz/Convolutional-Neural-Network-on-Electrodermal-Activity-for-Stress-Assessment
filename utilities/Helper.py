from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, Metric
import tensorflow as tf
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

class OpenerHelper:
    @staticmethod
    def load_data_from_pickle(directory):
        """
        Load all data from pickle files in the specified directory.
        
        Returns:
            dict: Loaded data from pickle files.
        """
        data = {}
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                key = filename.split('.')[0]  # Use the filename without extension as key
                data[key] = pd.read_pickle(os.path.join(directory, filename))
        return data
    
    @staticmethod
    def reshape_features(X_resampled, X_dict):
        """
        Reshape the resampled features back to their original shape.

        Args:
            X_resampled (array): Resampled feature array.
            X_dict (dict): Original dictionary of features.

        Returns:
            dict: Reshaped feature dictionary.
        """
        start_idx = 0
        X_resampled_dict = {}
        for feature, X in X_dict.items():
            feature_length = X.shape[1] * X.shape[2] if X.ndim == 3 else X.shape[1]
            X_resampled_dict[feature] = X_resampled[:, start_idx:start_idx + feature_length].reshape(-1, X.shape[1], X.shape[2] if X.ndim == 3 else 1)
            start_idx += feature_length
        
        return X_resampled_dict
