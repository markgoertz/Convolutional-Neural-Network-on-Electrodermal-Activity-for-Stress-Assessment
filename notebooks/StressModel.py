#!/usr/bin/env python
# coding: utf-8

# In[21]:


import keras.callbacks
from sklearn.model_selection import KFold
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers, models, regularizers, optimizers, callbacks
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, Metric
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Dropout


from dvclive import Live  # Ensure DVCLive is imported
from dvclive.keras import DVCLiveCallback  # Import the callback
import yaml
import pickle


# In[22]:


MAIN_PATH = os.path.dirname(os.getcwd())
DATA_PATH = MAIN_PATH + "/data/results"
MODEL_PATH = MAIN_PATH + "/models"
LOG_PATH = MAIN_PATH + "/logs"

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
NUM_FOLDS = 5

BEST_VAL_SCORE = 0
BEST_MODEL = None
HISTORY = []  # Initialize history_list


# In[23]:


def load_config(config_path=MAIN_PATH + "/params.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()  # This loads the configuration once and allows direct access


# In[24]:


def plot_history_metrics(history_dict: dict):
    total_plots = len(history_dict)
    cols = total_plots // 2
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history_dict.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.show()


# In[25]:


def load_df():
    df = pd.read_csv(MAIN_PATH + "/data/result_df.csv")
    return df


# In[26]:


def Clean_missing_values(numpy_data):
    numpy_data['x_train'], numpy_data['y_train'] = Remove_missing_values(numpy_data['x_train'], numpy_data['y_train'])
    numpy_data['x_val'], numpy_data['y_val'] = Remove_missing_values(numpy_data['x_val'], numpy_data['y_val'])
    numpy_data['x_test_1'], numpy_data['y_test_1'] = Remove_missing_values(numpy_data['x_test_1'], numpy_data['y_test_1'])
    numpy_data['x_test_2'], numpy_data['y_test_2'] = Remove_missing_values(numpy_data['x_test_2'], numpy_data['y_test_2'])
    
    return numpy_data

def Remove_missing_values(x_data, y_data):
    # Check if y_data contains missing values (NaNs) and remove corresponding x_data rows
    valid_indices = ~np.isnan(y_data)  # Find valid (non-NaN) indices in y_data
    x_clean = x_data[valid_indices]
    y_clean = y_data[valid_indices]
    return x_clean, y_clean


# In[27]:


def load_all_pickles_and_convert_to_numpy_with_columns(directory):
    try:
        # Dictionary to store the loaded data as NumPy arrays
        numpy_data = {}
        
        # List all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):  # Only process .pkl files
                file_path = os.path.join(directory, filename)
                
                # Load the data from the .pkl file
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                
                # Store the data in the dictionary, using the file name (without .pkl) as the key
                dataset_name = filename.replace(".pkl", "")
                
                # Print to verify if DataFrame still has columns (for debugging)
                if isinstance(data, pd.DataFrame):
                    # Convert the DataFrame into a dictionary of NumPy arrays, one for each column
                    numpy_data[dataset_name] = {col: np.array(data[col].tolist()) for col in data.columns}
                else:
                    # If the dataset is not a DataFrame (like labels), convert directly to NumPy array
                    numpy_data[dataset_name] = np.array(data)
                
                print(f"Loaded {filename} successfully.")
        
        return numpy_data
    
    except Exception as e:
        raise RuntimeError(f"Failed to load and convert datasets to NumPy arrays: {e}")


# In[28]:


def calculate_class_weights(df, label_column):
    vals_dict = {}
    for i in df[label_column]:
        if i in vals_dict.keys():
            vals_dict[i] += 1
        else:
            vals_dict[i] = 1
    total = sum(vals_dict.values())
    weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}

    print(f"Weight dict for model: {weight_dict}")
    return weight_dict


# In[29]:


class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


# In[30]:


def create_cnn_branch(input_shape):
    """Create a single CNN branch for a given input shape."""
    input_layer = Input(shape=input_shape)
    
    x = layers.Conv1D(filters=32, kernel_size=config["model"]["kernel_size"], 
                      activation=config["model"]["activation"], padding="same", 
                      kernel_regularizer=regularizers.l2(0.001))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation=config["model"]["activation"], 
                      padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation=config["model"]["activation"], 
                      padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    return input_layer, x


# In[31]:


def create_cnn_model():
    # Input shape for each branch: (timesteps, features), where features is typically 1 for each metric
    input_shape = (config["model"]["input_shape"], 1)  # Each input will have one feature

    # Create a list to hold the inputs and the outputs of each branch
    branches = []
    
    # Loop through each metric in the config file and create a branch for it
    for metric in config['model']['metrics']:
        input_layer, branch_output = create_cnn_branch(input_shape)
        branches.append((input_layer, branch_output))
    
    # Extract input layers and output branches for concatenation
    input_layers = [branch[0] for branch in branches]
    branch_outputs = [branch[1] for branch in branches]
    
    # Concatenate outputs from all branches
    concatenated = layers.concatenate(branch_outputs)

    # Fully connected layers after concatenation
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(concatenated)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    # Output layer for binary classification
    output_layer = layers.Dense(1, activation="sigmoid")(x)

    # Create and return the model
    model = models.Model(inputs=input_layers, outputs=output_layer)
    
    return model


# In[32]:


# Function to dynamically create the multimodal model
def create_multimodal_model():
    input_shape = (config['model']['input_shape'], 1)  # Shape of each input (timesteps, 1)
    branches = []  # To hold the input layers and branches

    # Loop through the metrics in the config file
    for metric in config['model']['metrics']:
        print(f"Creating branch for metric: {metric}")
        input_layer, branch = create_cnn_branch(input_shape, metric)
        branches.append((input_layer, branch))  # Store input and output branches

    # Separate inputs and outputs for concatenation
    input_layers = [branch[0] for branch in branches]
    output_branches = [branch[1] for branch in branches]

    # Concatenate the outputs from different branches
    concatenated = concatenate(output_branches)

    # Fully connected layers after concatenation
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)

    # Output layer (adjust for binary or multi-class classification)
    output = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = keras.Model(inputs=input_layers, outputs=output)

    return model


# In[33]:


# Compile model with optimizer and loss function
def Compile_model():
    model = create_multimodal_model()  # Dynamically create model based on metrics
    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=config["model"]["learning_rate"])
    loss = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')
        ],
    )
    return model


# In[34]:


def SplitDatasetForFolds(train_index, validation_index, fold_nr, numpy_data):
    print(f"Training fold {fold_nr}...")

    # Split the data into train sets for this fold.
    x_train_fold = numpy_data['x_train'][train_index]
    y_train_fold = numpy_data['y_train'][train_index]
    print(f"x train fold shape: {x_train_fold.shape}")
    print(f"y train fold shape: {y_train_fold.shape}")
    
    # Ensure to use only the training set indices
    x_validation_fold = numpy_data['x_val'][:len(validation_index)] 
    y_validation_fold = numpy_data['y_val'][:len(validation_index)]
    print(f"x fold val shape: {x_validation_fold.shape}")
    print(f"y fold val shape: {y_validation_fold.shape}")

    x_train_fold = x_train_fold.reshape(-1, 32, 2)
    x_validation_fold = x_validation_fold.reshape(-1, 32, 2)

    # Create tf.data.Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_fold, y_train_fold))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation_fold, y_validation_fold))
    test_dataset_subject1 = tf.data.Dataset.from_tensor_slices((numpy_data['x_test_1'], numpy_data['y_test_1']))    
    test_dataset_subject2 = tf.data.Dataset.from_tensor_slices((numpy_data['x_test_2'], numpy_data['y_test_2']))
    
    # Shuffling and batching the datasets
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_dataset = validation_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset_subject1 = test_dataset_subject1.batch(BATCH_SIZE)
    test_dataset_subject2 = test_dataset_subject2.batch(BATCH_SIZE)

    return train_dataset, validation_dataset, test_dataset_subject1, test_dataset_subject2


# In[35]:


def save_history_to_json(history, fold_number, best_model):
    # Create a dictionary for the current fold's metrics
    metrics = {
        "fold_number": fold_number,
        "val_accuracy": history.history['val_accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "best_model": best_model
    }

    # Load existing metrics if the file exists
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = []

    # Append the new metrics
    existing_metrics.append(metrics)

    # Write back the updated metrics to the file
    with open('metrics.json', 'w') as f:
        json.dump(existing_metrics, f, indent=4)


# In[36]:


def Train_fold(train_index, val_index, fold_number, numpy_data, weight_dict):
    exp_mess = f"fold-{fold_number}".lower()
    print(f"Experiment name: {exp_mess}")
    
    with Live(exp_message=f"Training fold {exp_mess}") as live:
        # Split data into training and validation sets for this fold
        train_dataset, validation_dataset, test_dataset_subject1, test_dataset_subject2 = SplitDatasetForFolds(train_index, val_index, fold_number, numpy_data)

       # Create and compile the model
        model = Compile_model()   
        # Set up callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(MODEL_PATH, f"best_model_fold_{fold_number}.keras"),
                save_best_only=True,
                monitor="val_accuracy"
            ),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
            DVCLiveCallback()
        ]

        # Log parameters for this fold
        live.log_param("fold_number", fold_number)
        live.log_param("epochs", config['model']['epochs'])
        live.log_param("batch_size", config['model']['batch_size'])
        live.log_param("learning_rate", config['model']['learning_rate'])
        live.log_param("folds", config['model']['folds'])
        live.log_param("kernel_size", config['model']['kernel_size'])
        live.log_param("activation", config['model']['activation'])
        live.log_param("input_shape", config['model']['input_shape'])
        live.log_param("input_features", config['model']['input_features'])
        live.log_param("shuffle_buffer_size", config['model']['shuffle_buffer_size'])
        
        print("Starting training...")
        
        # Train the model
        model.fit(
            train_dataset,
            epochs=config['model']['epochs'],
            validation_data=(validation_dataset),
            callbacks=callbacks,
            class_weight=weight_dict
        )

        # Save the model to the models directory
        os.makedirs(MODEL_PATH, exist_ok=True)
        model.save(os.path.join(MODEL_PATH, f'model_{fold_number}.h5'))
        print(f'Model saved to {MODEL_PATH}/model_{fold_number}.h5')

        print(f"Training fold {fold_number} completed\n")


# In[71]:


def Cross_validation_training(numpy_data, weight_dict):
    scores = []
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure the model path exists

    # Initialize KFold with the number of splits
    print(f'Shape EDA x_train{numpy_data['x_train']['EDA'].shape}')

    kfold = KFold(n_splits=config['model']['folds'], shuffle=True, random_state=42)
    print(f'Shape EDA x_train{numpy_data['x_train']['EDA'].shape}')
    try:
        for metric in config['model']['metrics']:
            print(f"Processing metric: {metric}")
            for fold_number, (train_index, val_index) in enumerate(kfold.split(numpy_data['x_train'][metric]), start=1):
                print(f"Training fold {fold_number} for metric {metric}")
                score = Train_fold(train_index, val_index, fold_number, numpy_data, weight_dict)
                scores.append(score)
    
    except Exception as e:
        print(f"An error occurred during cross-validation training: {e}")
    
    return scores  


# In[72]:


def main():
    df = load_df();
    datasets = load_all_pickles_and_convert_to_numpy_with_columns(DATA_PATH)

    # Calculate weights
    weight_dict = calculate_class_weights(df, 'downsampled_label')

    x_train = datasets['x_train']
    y_train = datasets['y_train']
    x_val = datasets['x_val']
    y_val = datasets['y_val']
    x_test_1 = datasets['x_test_1']
    y_test_1 = datasets['y_test_1']
    x_test_2 = datasets['x_test_2']
    y_test_2 = datasets['y_test_2']
    
    # Filter out columns that are not in config['model']['metrics']
    def filter_metrics(data, metrics):
        return {key: value for key, value in data.items() if key in metrics}

    # Apply the filter to datasets
    for key in datasets.keys():
        datasets[key] = filter_metrics(datasets[key], config['model']['metrics'])

    for key, value in datasets.items():
        for sub_key, sub_value in value.items():
            print(f"{key} - {sub_key}: {sub_value.shape}")

    # Train model
    Cross_validation_training(datasets, weight_dict)


# In[73]:


main()


# In[ ]:




