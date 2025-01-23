import pickle
import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unpickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            return data
    except FileNotFoundError:
        logging.error(f"The file {file_path} does not exist.")
    except pickle.UnpicklingError as e:
        logging.error(f"The file {file_path} could not be unpickled. {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

# Save dictionary to pickle after converting DataFrame to dict
def save_windows_as_pickle(windowed_signals, metrics, labels, subject_id, session_name, base_path, start_time, window_duration_seconds):
    """Save windowed signals for each metric into a pickle file for a subject."""
    print(f"Saving windowed signals to pickle for subject: {subject_id}, session: {session_name}")

    # Ensure output directory exists
    output_dir = os.path.join(base_path, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    num_windows = len(windowed_signals[metrics[0]])  # Get the number of windows from the first metric

    # Prepare data dictionary to hold the windows for each metric
    data_to_save = {
        "ID": subject_id,
        "Session": session_name,
        "Time": start_time + pd.to_timedelta(np.arange(num_windows) * window_duration_seconds, unit='s'),
        "labels": np.array(labels)
    }

    # Add each metric's windows to the dictionary
    for metric in metrics:
        # Store each metric’s windowed data as-is, preserving shape
        data_to_save[metric] = np.array([window.flatten().tolist() for window in windowed_signals[metric]])
        print(f"Saved metric '{metric}' with shape {data_to_save[metric].shape}")

    # Define the file path and save the data as a pickle file
    pickle_file_path = os.path.join(output_dir, f"{subject_id}_{session_name}_windowed_data.pkl")
    with open(pickle_file_path, "wb") as file:
        pickle.dump(data_to_save, file)

    print(f"Data saved to {pickle_file_path}")
    return pickle_file_path

def save_dataframe_to_csv(df, subject_id, session_name):
    """Save the DataFrame to a CSV file in the specified folder."""
    # Create a directory path for saving the CSV
    save_dir = os.path.join(os.getcwd(), "data", "WESAD", subject_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the filename
    filename = f"{subject_id}{session_name}_data.csv"
    file_path = os.path.join(save_dir, filename)

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)  # index=False to avoid saving the index as a column
