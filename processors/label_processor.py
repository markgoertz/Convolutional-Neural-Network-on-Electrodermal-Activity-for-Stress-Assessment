import numpy as np
from scipy.stats import mode

class LabelProcessor:
    """
    LabelProcessor class for downsampling and encoding labels.
    Attributes:
        original_rate (int): The original sampling rate of the labels.
        target_rate (int): The target sampling rate for downsampling.
    Methods:
        __init__(original_rate, target_rate):
            Initializes the LabelProcessor with the original and target rates.
        downsample(labels):
            Downsample labels using mode filtering.
            Args:
                labels (list or np.array): The input labels to be downsampled.
            Returns:
                np.array: The downsampled labels.
        encode(labels):
            Encode labels: 1 if label is 2, else 0.
            Args:
                labels (list or np.array): The input labels to be encoded.
            Returns:
                np.array: The encoded labels.
        majority_vote(label_window):
            Perform a majority vote on a window of labels.
            Args:
                label_window (list or np.array): The input window of labels.
            Returns:
                int: The label with the highest count in the window.
    """
    def __init__(self, original_rate, target_rate):
        self.original_rate = original_rate
        self.target_rate = target_rate

    def downsample(self, labels):
        """Downsample labels using mode filtering."""
        decimation_factor = self.original_rate // self.target_rate
        downsampled_labels = []
        for i in range(0, len(labels), decimation_factor):
            segment = labels[i:i + decimation_factor]
            mode_value, _ = mode(segment)  # Extract mode value
            downsampled_labels.append(mode_value)
        
        return np.array(downsampled_labels)

    @staticmethod
    def encode(labels):
        """Encode labels: 1 if label is 2, else 0."""
        return np.where(labels == 2, 1, 0)

    @staticmethod
    def majority_vote(label_window):
        label_counts = np.bincount(label_window)
        return np.argmax(label_counts)