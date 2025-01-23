import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

from processors.wesad_processor import WESADProcessor

def main():
    CONFIG = {
        "target_rate": {'ACC': 32, 'BVP': 32, 'EDA': 4, 'TEMP': 4},
        "window_duration_seconds": 8,
        "metrics": ["ACC", "EDA", "TEMP", "BVP"],
        "preprocessed": "unknown",
        "timestamp": pd.Timestamp("2020-01-01 00:00:00")
    }

    #Make sure that the project contains a data folder + WESAD folder to process the WESAD dataset.
    base_path = os.path.join(os.getcwd(), 'data', 'WESAD')
    processor = WESADProcessor(CONFIG)
    processor.process_dataset(base_path)
    print("Processing complete!")

if __name__ == "__main__":
    main()