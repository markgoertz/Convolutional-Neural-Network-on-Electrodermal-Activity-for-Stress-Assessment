import numpy as np
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler


class SignalProcessor:
    @staticmethod
    def process_acc_signal(data):
        return np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)

    @staticmethod
    def extract_all_signals(signals):
        extracted_signals = {}
        
        for key in signals.keys():
            if key == 'ACC':
                data = signals[key]
                magnitudes = SignalProcessor.process_acc_signal(data)
                extracted_signals[key] = magnitudes
            else:
                extracted_signals[key] = signals[key][:, 0]  # Assuming multi-dimensional data
        return extracted_signals
    
    @staticmethod
    def downsample(signal, original_rate, target_rate):
        factor = original_rate // target_rate
        return decimate(signal, factor)

    @staticmethod
    def normalize(signal):
        scaler = StandardScaler()
        signal = signal.reshape(-1, 1)
        normalized_signal = scaler.fit_transform(signal)
        return normalized_signal.flatten()
