from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import ast

class SignalScaler:
    @staticmethod
    def fit_transform(*signals):
        """Scale multiple signals using MinMaxScaler."""
        scaler = StandardScaler()
        return [scaler.fit_transform(signal.reshape(-1, 1)).flatten() for signal in signals]

    @staticmethod
    def transform_signals(*signals):
        """Scale multiple signals using StandardScaler."""
        scaler = StandardScaler()
        return [scaler.transform(signal.reshape(-1, 1)).flatten() for signal in signals]

    @staticmethod
    def _parse_signal(signal):
        """
        Parse a signal string into a list, if necessary.
        
        Parameters:
            signal (str or list): Signal data, either a list or a string representing a list.
        
        Returns:
            list: Parsed signal data.
        """
        if isinstance(signal, str):
            try:
                signal = ast.literal_eval(signal)
                if not isinstance(signal, list):
                    raise ValueError("Parsed signal is not a list")
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Error parsing signal string: {e}")
        return signal
    
class Labelsprocessor:
    @staticmethod
    def extract_labels(df, t_df1, t_df2, label_column='labels'):
        try:
            # Extract labels for the main dataset and test subjects
            labels_array = df['labels'].values
            test_labels_array_subject_1 = t_df1['labels'].values
            test_labels_array_subject_2 = t_df2['labels'].values

            print(
                f"Labels list Count Subject 1: {len(test_labels_array_subject_1)}\n"
                f"Labels list Count Subject 2: {len(test_labels_array_subject_2)}"
            )
            
            
            return labels_array, test_labels_array_subject_1, test_labels_array_subject_2
        except Exception as e:
            raise ValueError(f"Failed to extract labels: {e}")
        



