# %%
import keras.callbacks
from sklearn.model_selection import KFold
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.losses
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers, models, regularizers, optimizers, callbacks
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, Metric
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.models import Model  # Import Model here


import dvc.api
from dvclive import Live
from dvclive.keras import DVCLiveCallback  # Import the callback
import yaml
import pickle

# %%

DATA_PATH = "data/results"
MODEL_PATH = "models"
LOG_PATH = "logs"

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
NUM_FOLDS = 5

BEST_VAL_SCORE = 0
BEST_MODEL = None
HISTORY = []  # Initialize history_list

# %%
def load_config():
    return dvc.api.params_show("params.yaml")

config = load_config()

# %%
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

# %%
def load_df():
    df = pd.read_csv("data/result_df.csv")
    return df

# %%
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


# %%
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

# %%
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

# %%

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


# %%
def create_model_head(input_layer):
    # First convolutional layer
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=config["model"]["kernel_size"], 
                      activation=config["model"]["activation"], padding="same", 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    # Second convolutional layer
    x = tf.keras.layers.Conv1D(64, kernel_size=config["model"]["kernel_size"], activation=config["model"]["activation"])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)   
    
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=config["model"]["kernel_size"], 
                      activation=config["model"]["activation"], padding="same", 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    # Flatten the output
    x = tf.keras.layers.Flatten()(x)

    return x  

# %%
def build_model(input_layers, model_heads):
    # Merge models using their outputs directly
    combined = tf.keras.layers.concatenate(model_heads)

    # Add additional layers after merging
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Adjust based on your task

    # Final model
    model = keras.Model(inputs=input_layers, outputs=outputs)

    return model

# %%
def compile_model(input_layers, model_heads):
    model = build_model(input_layers, model_heads)
    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=config["model"]["learning_rate"])
    loss = keras.losses.binary_crossentropy  # Ensure this is a callable, not a result

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')
        ]
    )
    model.summary()
    return model

# %%
def train_model(model, x_train, y_train, x_val, y_val, class_weight):
    """Trains the model on the training data."""

    with Live() as live:
        for epoch in range(config['model']['epochs']):
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=1,  # Train for one epoch at a time
                class_weight=class_weight,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(MODEL_PATH, 'best_model.keras'),  # Add filepath argument
                        save_best_only=True,
                        monitor="val_binary_accuracy"
                    ),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
                    DVCLiveCallback(live=live)  # Add DVCLiveCallback to the list
                ]
            )
            live.log_params(config)
            live.log_artifact(
                os.path.join(MODEL_PATH, 'best_model.h5'),
                type="model",
                desc="This is a convolutional neural network model that is developed to detect stress.",
                labels=["no-stress", "stress"],
            )

        model.save(os.path.join(MODEL_PATH, 'best_model.h5'))
        live.end()

# %%
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


# %%
def Preparing_model(x_train, y_train, x_test, y_test, weight_dict):
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure the model path exists

    try:
        # Create the model heads
        model_heads = []
        input_layers = []
        
        for metric in config['model']['metrics']:
            input_shape = (config['model']['input_shapes'][metric], config['model']['input_features'])
            input_layer = tf.keras.layers.Input(shape=input_shape, name=f'input_{metric.lower()}')
            input_layers.append(input_layer)
            print(f"Input shape for {metric}: {input_shape}")
            
            # Create a model head for each input
            model_head = create_model_head(input_layer)
            model_heads.append(model_head)

        print(f"Model heads created: {model_heads}")

        model = compile_model(input_layers, model_heads)

        train_model(
            model,
            [x_train[metric] for metric in config['model']['metrics']],
            y_train,
            [x_test[metric] for metric in config['model']['metrics']],
            y_test,
            weight_dict
        )
    except Exception as e:
        print(f"An error occurred during preparing: {type(e).__name__}: {e}")


# %%
def filter_columns(data, metrics):
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, dict) and key.startswith('x_'):
            filtered_data[key] = {k: v for k, v in value.items() if k in metrics}
        else:
            filtered_data[key] = value
    return filtered_data

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    df = load_df()
    datasets = load_data_from_pickle(DATA_PATH)
 
    # Extract the datasets
    x_train = datasets['x_train']
    y_train = datasets['y_train']

    x_val = datasets['x_val']
    y_val = datasets['y_val']

    # Calculate weights
    weight_dict = calculate_class_weights(df, 'downsampled_label')

    # Filter columns based on config['model']['metrics']
    datasets = filter_columns(datasets, config['model']['metrics'])

    for key, value in datasets.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"Shape of {key}_{sub_key}: {sub_value.shape}")
        else:
            print(f"Shape of {key}: {value.shape}")

    # Train model
    x = Preparing_model(x_train, y_train, x_val, y_val, weight_dict)
    print(f"Model training completed")

    return y_val



# %%
x = main()
x


