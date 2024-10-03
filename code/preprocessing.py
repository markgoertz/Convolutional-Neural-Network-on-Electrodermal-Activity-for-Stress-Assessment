import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from collections import Counter
import cvxEDA
import cvxEDA.src
import cvxEDA.src.cvxEDA

MAIN_PATH = "D:/Master of Applied IT/"
DATA_PATH = MAIN_PATH + "/data/merged_data.csv"

def load_data():
    dataset = pd.read_csv(DATA_PATH)
    return dataset


def calculate_eda_levels(y):
    fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
    Fs = fs_dict['EDA']
    yn = (y - y.mean()) / y.std()
    r, p, t, l, d, e, obj = cvxEDA.src.cvxEDA.cvxEDA(yn, 1. / Fs)
    return r, t, yn

def preprocess_data(dataset):
    # Calculate EDA levels
    unique_ids = dataset['ID'].unique()
    new_dataframe_eda = pd.DataFrame(columns=["cvx_phasic", "cvx_tonic"])
    
    for unique_id in unique_ids:
        subset_data = dataset[dataset['ID'] == unique_id]
        phasic, tonic, yn = calculate_eda_levels(subset_data['w_eda'].values)
        temp_df = pd.DataFrame({"cvx_phasic": phasic, "cvx_tonic": tonic})
        new_dataframe_eda = pd.concat([new_dataframe_eda, temp_df], ignore_index=True)

    dataset = pd.concat([dataset, new_dataframe_eda], axis=1)
    return dataset

def create_sequences_df(merged_df, max_length=32):
    sequences = []
    # Your logic for creating sequences here...
    return pd.DataFrame(sequences)

def scale_labels(sequences_df):
    sequences_df['downsampled_label'] = sequences_df['downsampled_label'].apply(lambda x: 1 if x == 2.0 else 0)
    le = preprocessing.LabelEncoder()
    le.fit(sequences_df['downsampled_label'])
    sequences_df['downsampled_label'] = le.transform(sequences_df['downsampled_label'])
    return sequences_df

def main():
    dataset = load_data()
    processed_data = preprocess_data(dataset)
    sequences_df = create_sequences_df(processed_data)
    sequences_df = scale_labels(sequences_df)
    return sequences_df

if __name__ == "__main__":
    main()