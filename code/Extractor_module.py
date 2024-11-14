import numpy as np
import pandas as pd
import os

class SignalExtractor:
    @staticmethod
    def extract_selected_features(signals, features):
        extracted_signals = {}

        for feature in features:
            if feature in signals:
                if len(signals[feature].shape) > 1:
                    extracted_signals[feature] = signals[feature][:, 0]
                else:
                    extracted_signals[feature] = signals[feature]
            else:
                print(f"Feature {feature} not found in signals.")
                
        print("Selected features extracted:", list(extracted_signals.keys()))
        return extracted_signals


    @staticmethod
    def load_data_from_pickle(directory):
        """
        Load all data from pickle files in the specified directory.

        Returns:
            tuple: Loaded training and test data as dictionaries.
        """
        
        data = {}
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                key = filename.split('.')[0]  # Use the filename without extension as key
                data[key] = pd.read_pickle(os.path.join(directory, filename))

        return data
    
    @staticmethod
    def filter_columns(data, metrics):
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, dict) and key.startswith('x_'):
                filtered_data[key] = {k: v for k, v in value.items() if k in metrics}
            else:
                filtered_data[key] = value
        return filtered_data
    
    
    @staticmethod
    def restructure_x_train_and_y_val(x_train, y_val):
        # Initialize an empty dictionary for the restructured x_train
        x_train_restructured = {}

        # Loop through the first subject to get the signal keys automatically
        first_subject_signals = list(x_train[list(x_train.keys())[0]].keys())
        
        # Initialize empty lists for each signal type based on the first subject's data
        for signal in first_subject_signals:
            x_train_restructured[signal] = []

        # Loop through each subject's data and aggregate the signals
        for subject, signals in x_train.items():
            for signal, data in signals.items():
                x_train_restructured[signal].append(data)

        # Convert lists of windows into single numpy arrays for each signal type
        for signal in x_train_restructured:
            # Concatenate data for each signal type across subjects (e.g., all windows for BVP across subjects)
            x_train_restructured[signal] = np.concatenate(x_train_restructured[signal], axis=0)

        # Now restructure y_val similarly, ensuring labels align with the concatenated windows
        y_val_restructured = []
        for subject in y_val:
            y_val_restructured.extend(y_val[subject].ravel())  # Flatten the labels for each subject

        # Convert y_val_restructured to a numpy array and ensure it's a column vector
        y_val_restructured = np.array(y_val_restructured).reshape(-1, 1)

        return x_train_restructured, y_val_restructured
