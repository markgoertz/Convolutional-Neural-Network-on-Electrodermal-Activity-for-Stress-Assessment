import pandas as pd
import numpy as np

def combine_windows_into_dataframe(windowed_signals, labels, subject_id, session_name, window_duration_seconds, start_time, metrics):
    """
    Combine windowed signals into a DataFrame with time, subject information, and labels.
    """
    num_windows = len(windowed_signals[metrics[0]])  # Get number of windows from the first metric

    # Check if all metrics have the same number of windows
    for metric in metrics:
        if len(windowed_signals[metric]) != num_windows:
            return None

    # Initialize DataFrame with common columns
    df_windows = pd.DataFrame(index=np.arange(num_windows), columns=["ID", "Session", "StartTime", "Label"] + metrics)

    # Fill common columns
    df_windows["ID"] = subject_id
    df_windows["Session"] = session_name
    df_windows["StartTime"] = start_time + pd.to_timedelta(np.arange(num_windows) * window_duration_seconds, unit='s')
    df_windows["labels"] = labels

    # Populate metric columns with the flattened windows
    for metric in metrics:
        df_windows[metric] = [window.flatten().tolist() for window in windowed_signals[metric]]

    return df_windows, session_name
