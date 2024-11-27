import os
import zipfile
import pandas as pd
import numpy as np
from scipy.signal import decimate
from sklearn.preprocessing import MinMaxScaler

# Configuration for both datasets
DATASET_CONFIG = {
    "WESAD": {
        "target_rate": 32,
        "window_duration_seconds": 8,  # Adjust according to your needs
        "base_path": r"C:/Master of Applied IT/data/WESAD.zip",  # Update this path
    },
    "AffectiveRoad": {
        "target_rate": 32,
        "window_duration_seconds": 8,  # Duration of each window in seconds
        "base_path": r"C:/Master of Applied IT/data/AffectiveROAD_Data/Database/E4",
    }
}

def unzip_files(folder_path):
    """Unzip all files in the given folder if they are not already unzipped."""
    for item in os.listdir(folder_path):
        if item.endswith(".zip"):
            file_path = os.path.join(folder_path, item)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_path = file_path.replace(".zip", "")
                if not os.path.exists(extract_path):
                    zip_ref.extractall(extract_path)
            print(f"Unzipped: {file_path}")


def load_csv_file(file_path):
    """Load CSV file and extract time, sample rate, and data values."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    print(f'processing: {file_path}')
    # Extract sample rate from the second line (first value)
    sample_rate = float(lines[1].strip().split(',')[0])  # Get the first value of the second line

    # Prepare to collect data
    data = []

    # # Process the data lines
    # for line in lines[2:]:  # Skip the first two lines
    #     # Split by comma and convert to float
    #     values = list(map(float, line.strip().split(',')))

    #     # If it's ACC data (3 values), compute the magnitude
    #     if len(values) == 3:
    #         magnitude = np.sqrt(values[0]**2 + values[1]**2 + values[2]**2)
    #         data.append(magnitude)
    #     else:
    #         # Normal data (1 value), add it directly
    #         data.append(values[0])

    # Convert data to a 1D numpy array
    return int(sample_rate), np.array(data)

def preprocess_data(data_dict, sample_rates, config):
    """Preprocess the EDA, TEMP, BVP, and ACC data using dataset-specific configuration."""
    target_rates = config.get("target_rates", {})  # Retrieve target rates from config
    window_duration_seconds = config["window_duration_seconds"]

    # Prepare lists to hold processed features
    all_windows = []

    # Normalize EDA and TEMP directly
    for feature_name in ['EDA', 'TEMP']:
        data = data_dict[feature_name]
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        all_windows.append(normalized_data)

    # Process BVP
    bvp_data = data_dict['BVP']
    # Downsample BVP based on its sample rate
    bvp_target_rate = target_rates.get('BVP', 32)  # Assuming default target rate is 32Hz
    bvp_sample_rate = sample_rates['BVP']
    if bvp_sample_rate > bvp_target_rate:
        downsample_factor = int(bvp_sample_rate / bvp_target_rate)
        bvp_data = decimate(bvp_data, downsample_factor)
    bvp_data = MinMaxScaler().fit_transform(bvp_data.reshape(-1, 1)).flatten()  # Normalize
    all_windows.append(bvp_data)

    # Process ACC
    acc_data = data_dict['ACC']
    # Calculate the magnitude
    magnitude = np.sqrt(np.sum(acc_data ** 2, axis=1))  # Shape will be [num_samples]
    magnitude = MinMaxScaler().fit_transform(magnitude.reshape(-1, 1)).flatten()  # Normalize
    
    # Downsample ACC magnitude based on its sample rate
    acc_target_rate = target_rates.get('ACC', 32)  # Assuming default target rate is 32Hz
    acc_sample_rate = sample_rates['ACC']
    if acc_sample_rate > acc_target_rate:
        downsample_factor = int(acc_sample_rate / acc_target_rate)
        magnitude = decimate(magnitude, downsample_factor)
    
    all_windows.append(magnitude)

    # Windowing
    # Calculate window size in terms of samples for the lowest target rate
    min_target_rate = min(target_rates.values())  # Get minimum target rate
    window_size = window_duration_seconds * min_target_rate
    combined_windows = []
    
    # Make sure all windows have the same length
    for i in range(0, len(all_windows[0]) - window_size + 1, window_size):
        combined_window = np.concatenate([feature[i:i + window_size] for feature in all_windows])
        combined_windows.append(combined_window)

    return combined_windows


def process_wesad_data(base_path):
    """Process WESAD dataset for training and testing."""
    config = DATASET_CONFIG["WESAD"]
    all_windows = []

    # Traverse each subject/session folder in WESAD
    for folder_name in os.listdir(base_path):
        subject_path = os.path.join(base_path, folder_name)
        if os.path.isdir(subject_path):
            # Process each CSV file in WESAD
            for file_name in os.listdir(subject_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(subject_path, file_name)
                    sample_rate, data = load_csv_file(file_path)
                    processed_windows = preprocess_data(data, sample_rate, config)
                    all_windows.extend(processed_windows)  # Collecting all windows for WESAD

    return all_windows






def main():
    wesad_base_path = DATASET_CONFIG["WESAD"]["base_path"]
    affective_road_base_path = DATASET_CONFIG["AffectiveRoad"]["base_path"]

    print("Processing WESAD dataset for training/testing...")
    wesad_windows = process_wesad_data(wesad_base_path)
    print(f"Total windows from WESAD: {len(wesad_windows)}")

    print("Processing AffectiveRoad dataset for validation...")
    affective_road_windows = process_affective_road_data(affective_road_base_path)
    print(a)
    print(f"Total windows from AffectiveRoad: {len(affective_road_windows)}")

    print("Processing complete.")
    return affective_road_windows
    # Here, you can save or further process `wesad_windows` and `affective_road_windows` as needed
    # Example: Saving to pickle files
    # with open("wesad_windows.pkl", "wb") as f:
    #     pickle.dump(wesad_windows, f)
    # with open("affective_road_windows.pkl", "wb") as f:
    #     pickle.dump(affective_road_windows, f)

if __name__ == "__main__":
    main()
