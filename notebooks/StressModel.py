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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers, models, regularizers, optimizers, callbacks
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, Metric

from dvclive import Live  # Ensure DVCLive is imported
from dvclive.keras import DVCLiveCallback  # Import the callback
import yaml

MAIN_PATH = os.path.dirname(os.getcwd()) + "/Master of Applied IT"
DATA_PATH = MAIN_PATH + "/data/numpy"
MODEL_PATH = MAIN_PATH + "/models"
LOG_PATH = MAIN_PATH + "/logs"

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
NUM_FOLDS = 5

BEST_VAL_SCORE = 0
BEST_MODEL = None
HISTORY = []  # Initialize history_list

def load_config(config_path=MAIN_PATH + "/params.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()  # This loads the configuration once and allows direct access


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

def load_df():
    df = pd.read_csv(MAIN_PATH + "/data/result_df.csv")
    return df

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

def gather_numpy_files(data_path):
    numpy_data = {}
    
    for file_name in os.listdir(data_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(data_path, file_name)
            numpy_data[file_name[:-4]] = np.load(file_path)  # Store in dict
    
    # Clean data by removing rows where y_* contains missing values
    numpy_data = Clean_missing_values(numpy_data)

    return numpy_data

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

def create_model():
    input_layer = keras.Input(shape=(config["model"]["input_shape"], config["model"]["input_features"]))
    
    x = layers.Conv1D(filters=32, kernel_size=config["model"]["kernel_size"], activation=config["model"]["activation"], padding="same", kernel_regularizer=regularizers.l2(0.001))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation=config["model"]["activation"], padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation=config["model"]["activation"], padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

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

def Compile_model():
    model = create_model()
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

def SplitDatasetForFolds(train_index, validation_index, fold_nr, numpy_data):
    print(f"Training fold {fold_nr}...")

    # Split the data into train sets for this fold.
    x_train_fold = numpy_data['x_train'][train_index]
    y_train_fold = numpy_data['y_train'][train_index]

    print(f"x_val shape: {numpy_data['x_val'].shape}")
    print(f"y_val shape: {numpy_data['y_val'].shape}")
    
    # Ensure to use only the training set indices
    x_validation_fold = numpy_data['x_val'][:len(validation_index)]  # Taking the first `len(validation_index)` samples
    y_validation_fold = numpy_data['y_val'][:len(validation_index)]

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

def convert_to_native(data):
    """Recursively convert numpy types to native python types."""
    if isinstance(data, dict):
        return {key: convert_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    elif isinstance(data, (np.float32, np.float64)):
        return data.item()  # Convert single value numpy float to Python float
    else:
        return data  

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
            # Add any other callbacks you need here
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


def Cross_validation_training(numpy_data, weight_dict):
    scores = []
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure the model path exists

    # Initialize KFold with the number of splits
    kfold = KFold(n_splits=config['model']['folds'], shuffle=True, random_state=42)

    try:
        for fold_number, (train_index, val_index) in enumerate(kfold.split(numpy_data['x_train']), start=1):
            print(f"Training fold {fold_number}")

            score = Train_fold(train_index, val_index, fold_number, numpy_data, weight_dict)
            scores.append(score)
          
    except Exception as e:
        print(f"An error occurred during cross-validation training: {e}")
    
    return scores  



def main():
    df = load_df();
    numpy_data = gather_numpy_files(DATA_PATH)

    # Calculate weights
    weight_dict = calculate_class_weights(df, 'downsampled_label')

    # Create model
    convolutional_model = create_model()
    convolutional_model.summary()

    # Train model
    Cross_validation_training(numpy_data, weight_dict)
    
    return numpy_data;

numpy = main()

