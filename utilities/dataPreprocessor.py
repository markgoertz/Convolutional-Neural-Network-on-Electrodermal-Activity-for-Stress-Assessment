import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def extract_zip(self, zip_path, extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    def load_data(self, file_path):
        print(f"Loading data from {file_path}...")
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def downsample(self, signal, target_rate, current_rate):
        factor = int(current_rate / target_rate)
        return decimate(signal, factor)

    def standardize(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def process_signal(self, signal, signal_type):
        target_rate = self.config.target_rate.get(signal_type)
        if target_rate:
            downsampled = self.downsample(signal, target_rate, current_rate=len(signal))
            standardized = self.standardize(downsampled)
            return standardized
        else:
            raise ValueError(f"Target rate for {signal_type} is not defined.")
