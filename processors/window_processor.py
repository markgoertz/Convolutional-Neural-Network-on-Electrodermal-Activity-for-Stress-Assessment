import numpy as np

class WindowProcessor:
    @staticmethod
    def sliding_window(arr, window_size, step_size):
        """Generate non-overlapping windows for a 1D numpy array."""
        return [np.array(arr[i:i + window_size]) for i in range(0, len(arr) - window_size + 1, step_size)]

    @staticmethod
    def apply_to_signals(signals, sizes):
        windowed_signals = {}
        
        for signal_name, signal_data in signals.items():
            # Use the provided window size for each signal
            window_size = sizes.get(signal_name, len(signal_data))
            windowed_signals[signal_name] = [window.reshape(-1, 1) for window in WindowProcessor.sliding_window(signal_data, window_size, window_size)]
        return windowed_signals