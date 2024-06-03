# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %pip install scikit-learn -q
# %pip install pandas -q
# %pip install numpy -q
# %pip install matplotlib -q
# %pip install seaborn -q
# %pip install keras -q
# %pip install os -q

# %pip install cvxopt -q
# -

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from sklearn import preprocessing, model_selection
import random
import seaborn as sns
import os
import cvxEDA

# +

MAIN_PATH = os.path.dirname(os.getcwd())
DATA_PATH = MAIN_PATH + "/data"

QUALITY_THRESHOLD = 64
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
NUM_FOLDS = 10
# -

# **Choice options of metrics are as follows:**
# - w_eda
# - cvx_phasic
# - cvx_tonic

METRIC = "w_eda"

# **MODEL CONFIGURATION**
#
# - adjust if necessary. This defines the model's performance

print(f"MAIN_PATH: {MAIN_PATH}")
print(f"DATA_PATH: {DATA_PATH}")

dataset = pd.read_csv("data/merged_data.csv")

dataset.dtypes

dataset

# +
import cvxEDA.src.cvxEDA

def calculate_eda_levels(y):
    fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
    Fs = fs_dict['EDA']
    yn = (y - y.mean()) / y.std()
    [r, p, t, l, d, e, obj] = cvxEDA.src.cvxEDA.cvxEDA(yn, 1. / Fs)
    return r, t, yn



# +
import matplotlib.pyplot as plt

# Define unique_ids
unique_ids = dataset['ID'].unique()

# Initialize the new DataFrame
new_dataframe_eda = pd.DataFrame(columns=["cvx_phasic", "cvx_tonic"])

# Iterate through each unique id
for unique_id in unique_ids:
    # Filter data for each id
    subset_data = dataset[dataset['ID'] == unique_id]
    
    # Calculate EDA levels
    phasic, tonic, yn = calculate_eda_levels(subset_data['w_eda'].values)
    
    # Create a temporary DataFrame to hold the new data
    temp_df = pd.DataFrame({
        "cvx_phasic": phasic, 
        "cvx_tonic": tonic
    })

    new_dataframe_eda = pd.concat([new_dataframe_eda, temp_df], ignore_index=True)

    # # Plotting
    # plt.plot(tonic, label='Tonic')
    # plt.plot(phasic, label='Phasic')
    # plt.plot(subset_data['w_eda'].values, label='EDA')
    
    # plt.xlabel('Time')
    # plt.ylabel('EDA Levels')
    # plt.title(f'Phasic and Tonic EDA for ID: {unique_id}')
    # plt.legend()
    # plt.show()

# -

dataset = pd.concat([dataset, new_dataframe_eda], axis=1)

# +
import pandas as pd

# Function to create sequences DataFrame
def create_sequences_df(merged_df, max_length=32):
    sequences = []
    temp_sequence = []
    eda_sequence = []
    label = None
    time_sequence = []
    current_id = None
    phasic_eda_sequence = []
    tonic_eda_sequence = []

    for index, row in merged_df.iterrows():
        if current_id != row['ID']:
            # New ID encountered, append previous sequence to list
            if temp_sequence:
                sequences.append({
                    'ID': current_id,
                    'w_eda': eda_sequence,
                    'w_temp': temp_sequence,
                    'downsampled_label': label,
                    'Time': time_sequence,
                    'cvx_phasic': phasic_eda_sequence,
                    'cvx_tonic': tonic_eda_sequence
                })
            # Reset sequences for new ID
            temp_sequence = [row['w_temp']]
            eda_sequence = [row['w_eda']]
            label = row['downsampled_labels']
            time_sequence = [row['Time']]
            current_id = row['ID']
            phasic_eda_sequence = [row['cvx_phasic']]
            tonic_eda_sequence = [row['cvx_tonic']]
        else:
            # Append values to sequences
            temp_sequence.append(row['w_temp'])
            eda_sequence.append(row['w_eda'])
            time_sequence.append(row['Time'])
            phasic_eda_sequence.append(row['cvx_phasic'])
            tonic_eda_sequence.append(row['cvx_tonic'])

        # Check if sequence length exceeds max_length
        if len(temp_sequence) >= max_length:
            sequences.append({
                'ID': current_id,
                'w_eda': eda_sequence,
                'w_temp': temp_sequence,
                'downsampled_label': label,
                'Time': time_sequence,
                'cvx_phasic': phasic_eda_sequence,
                'cvx_tonic': tonic_eda_sequence
            })
            # Reset sequences for new ID
            temp_sequence = []
            eda_sequence = []
            label = None
            time_sequence = []
            current_id = None
            phasic_eda_sequence = []
            tonic_eda_sequence = []

    # Append last sequence if it's not empty
    if temp_sequence:
        sequences.append({
            'ID': current_id,
            'w_eda': eda_sequence,
            'w_temp': temp_sequence,
            'downsampled_label': label,
            'Time': time_sequence,
            'cvx_phasic': phasic_eda_sequence,
            'cvx_tonic': tonic_eda_sequence
        })

    # Convert list of dictionaries to DataFrame
    sequences_df = pd.DataFrame(sequences)
    return sequences_df

# Create sequences DataFrame
sequences_df = create_sequences_df(dataset)

# Check the resulting DataFrame
print(sequences_df.head())

# -

sequences_df

print(sequences_df.loc[88])


# +
import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
fig, axes = plt.subplots(2, 8, figsize=(25, 6))  # Increased figure size
axes = axes.flatten()

# Define unique_ids
unique_ids = dataset['ID'].unique()

# Iterate through each unique id
for i, unique_id in enumerate(unique_ids):
    if i < len(unique_ids):
        # Filter data for each id
        subset_data = dataset[dataset['ID'] == unique_id]
        
        # Plotting
        sns.lineplot(x='Time', y='w_eda', data=subset_data, ax=axes[i], color='blue', label='EDA')
        sns.lineplot(x='Time', y='cvx_phasic', data=subset_data, ax=axes[i], color='red', label='phasic EDA')
        # sns.lineplot(x='Time', y='cvx_tonic', data=subset_data, ax=axes[i], color='yellow', label='tonic EDA')                  
        # sns.lineplot(x='Time', y='w_temp', data=subset_data, ax=axes[i], color='red', label='Temp')

        axes[i].set_title(f"Data for {unique_id}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Measurement')
        axes[i].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()


# +
# import matplotlib.pyplot as plt

# # Create subplots
# fig, axes = plt.subplots(16, 3, figsize=(60, 40))  # Increased figure size

# # Define colors for each acceleration component
# colors = ['red', 'green', 'blue']

# # Iterate through each unique id
# for i, unique_id in enumerate(unique_ids):
#     # Filter data for each id
#     subset_data = dataset[dataset['ID'] == unique_id]
    
#     # Iterate through X, Y, and Z accelerations
#     for j, accel_component in enumerate(['X', 'Y', 'Z']):
#         ax = axes[i, j]  # Select the appropriate subplot
        
#         # Plot acceleration component with different color
#         ax.plot(subset_data['Time'], subset_data[accel_component], label=f'{accel_component} Acceleration', color=colors[j])
#         ax.set_title(f"Data for {unique_id} - {accel_component} Acceleration")
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Acceleration')
#         ax.legend()
#         ax.tick_params(axis='x', rotation=45)

# # Adjust layout
# plt.tight_layout()
# plt.show()

# -

sequences_df

print("Before replacing labels")
unique_labels_before = sequences_df['downsampled_label'].unique()
print(unique_labels_before, "\n")
print("Number of unique labels before replacement:", len(unique_labels_before), "\n")


sequences_df['downsampled_label'] = sequences_df['downsampled_label'].apply(lambda x : 1 if x == 2.0 else 0)


# +
from sklearn import preprocessing

print("After replacing labels")
unique_labels_after = sequences_df['downsampled_label'].unique()
print(unique_labels_after)
print("Number of unique labels after replacement:", len(unique_labels_after))

le = preprocessing.LabelEncoder()  # Generates a look-up table
le.fit(sequences_df['downsampled_label'])
sequences_df['downsampled_label'] = le.transform(sequences_df['downsampled_label'])

# -

num_classes = len(sequences_df['downsampled_label'].unique())
print(num_classes)


# +
from collections import Counter

def plot_label_distribution(df):
    # Define class labels
    sorts = {
        0: "No-stress",
        1: "Stress"
    }

    # Count occurrences of each label
    label_counts = Counter(df['downsampled_label'])

    # Extract counts for '0' and '1'
    counts = [label_counts[0], label_counts[1]]
    print("Label distribution:", counts)

    # Define bar labels
    bar_labels = [sorts[0], sorts[1]]

    # Plotting
    plt.bar(bar_labels, counts)
    plt.title("Number of samples per class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


# -

plot_label_distribution(sequences_df)


# +
# import pandas as pd
# from sklearn.utils import resample

# # Separate the majority ('no-stress') and minority ('stress') classes
# df_no_stress = sequences_df[sequences_df['downsampled_label'] == 0]
# df_stress = sequences_df[sequences_df['downsampled_label'] == 1]

# # Downsample the majority class ('no-stress') to match the minority class ('stress')
# df_no_stress_downsampled = resample(df_no_stress,
#                                     replace=False,  # Sample without replacement
#                                     n_samples=len(df_stress),  # Match the number of 'stress' samples
#                                     random_state=42)  # Ensure reproducibility

# # Combine the downsampled 'no-stress' class with the 'stress' class
# sequences_df_balanced = pd.concat([df_no_stress_downsampled, df_stress])

# # Shuffle the combined dataset to mix the samples
# sequences_df_balanced = sequences_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# +
# sequences_df_balanced

# +
# plot_label_distribution(sequences_df_balanced)
# -

# ****Scale and split data****
#
# We perform a simple Min-Max scaling to bring the value-range between 0 and 1.

# +
# Scale the 'w_eda' feature
scaler = preprocessing.MinMaxScaler()
eda_series_list = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in sequences_df[METRIC]]

# Convert the scaled feature back to a list of arrays
eda_array_list = [np.array(series).flatten() for series in eda_series_list]

# Separate the labels
labels_list = [i for i in sequences_df['downsampled_label']]

# Convert the labels list to numpy array
labels_array = np.array(labels_list)

# print(len(combined_series_list))
print(f"EDA list Count:", len(eda_series_list),"\n" "Labels list Count:", len(labels_array))



# +
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow.keras as keras

max_sequence_length = 32  # Choose the desired maximum sequence length

def apply_smote(x_train, y_train):
    # Reshape input features if necessary
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_reshaped, y_train)
    
    # Reshape resampled features back to original shape
    x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
    
    return x_train_resampled, y_train_resampled

# Padding sequences to ensure uniform length
padded_series_list = pad_sequences(eda_series_list, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')

# Splitting data into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    padded_series_list, labels_list, test_size=0.30, random_state=42, shuffle=True
)

# Further splitting the training data into training and validation sets (80% train, 20% val from the original 70% train)
x_train, x_val, y_train, y_val = model_selection.train_test_split(
    x_train, y_train, test_size=0.20, random_state=42, shuffle=True
)

# Convert to numpy arrays
x_train = np.asarray(x_train).astype(np.float32).reshape(-1, max_sequence_length, 1) 
y_train = np.asarray(y_train).astype(np.float32)

x_val = np.asarray(x_val).astype(np.float32).reshape(-1, max_sequence_length, 1) 
y_val = np.asarray(y_val).astype(np.float32)

x_test = np.asarray(x_test).astype(np.float32).reshape(-1, max_sequence_length, 1)
y_test = np.asarray(y_test).astype(np.float32)

# Check lengths of train, validation, and test sets
print(
    f"Length of x_train : {len(x_train)}\nLength of x_val : {len(x_val)}\nLength of x_test : {len(x_test)}\n"
    f"Length of y_train : {len(y_train)}\nLength of y_val : {len(y_val)}\nLength of y_test : {len(y_test)}"
)

# Check the class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Apply SMOTE using the function
x_train_resampled, y_train_resampled = apply_smote(x_train, y_train)

# Check the class distribution after SMOTE
class_distribution_after = Counter(y_train_resampled)
print("Class distribution after SMOTE:", {0: class_distribution_after[0], 1: class_distribution_after[1]})

# +
import matplotlib.pyplot as plt
from collections import Counter

# Class distribution before SMOTE
class_distribution_before = Counter(y_train)
# Class distribution after SMOTE
class_distribution_after = Counter(y_train_resampled)

# Define labels
labels = ['No Stress', 'Stress']

# Plotting
plt.figure(figsize=(10, 5))

# Plot before SMOTE
plt.subplot(1, 2, 1)
plt.bar(labels, class_distribution_before.values(), color='blue')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], labels)

# Plot after SMOTE
plt.subplot(1, 2, 2)
plt.bar(labels, class_distribution_after.values(), color='green')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], labels)

plt.tight_layout()
plt.show()


# +
def plot_dataset_distribution(x_train, y_train, x_test, y_test, x_val, y_val):
    """
    Plots a bar chart showing the sizes of the train, validation, and test sets.

    Parameters:
    - x_train, y_train: Training data and labels.
    - x_val, y_val: Validation data and labels.
    - x_test, y_test: Test data and labels.
    """
    dataset_names = ['Train', 'Test', 'Validation']
    x_lengths = [len(x_train), len(x_test), len(x_val)]
    y_lengths = [len(y_train), len(y_test), len(y_val)]
    
    # Plotting the bar plot
    plt.figure(figsize=(10, 6))
    
    plt.bar(dataset_names, x_lengths, color='b', alpha=0.6, label='X (Features)')
    plt.bar(dataset_names, y_lengths, color='r', alpha=0.6, label='Y (Labels)', bottom=x_lengths)
    
    plt.xlabel('Dataset')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Distribution')
    plt.legend()
    plt.show()


# Plot dataset distribution
plot_dataset_distribution(x_train, y_train, x_test, y_test, x_val, y_val)

# +
from sklearn.model_selection import KFold

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

def SplitDatasetForFolds(train_index, val_index, fold_nr):
    print(f"Training fold {fold_nr}...")

    # Split the data into train and validation sets for this fold.
    x_train_fold = x_train[train_index]
    y_train_fold = y_train[train_index]
    x_val_fold = x_train[val_index]
    y_val_fold = y_train[val_index]

    # Create tf.data.Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_fold, y_train_fold))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val_fold, y_val_fold))

    # Shuffling and batching the datasets
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset, val_dataset


# +
vals_dict = {}
for i in sequences_df['downsampled_label']:
    if i in vals_dict.keys():
        vals_dict[i] += 1
    else:
        vals_dict[i] = 1
total = sum(vals_dict.values())

# Formula used - Naive method where
# weight = 1 - (no. of samples present / total no. of samples)
# So more the samples, lower the weight

weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
print(weight_dict)


# -

# Assuming your one-hot encoded labels are in a variable named 'labels'
binary_labels = np.argmax(sequences_df['downsampled_label'])
print("Shape of binary labels:", binary_labels.shape)


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


# +

def create_model():
    input_layer = keras.Input(shape=(32, 1))
    
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model


# +
conv_model = create_model()
conv_model.summary()

# Save model to JSON
model_json = conv_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# +
# To store history of each fold
history_list = []
fold_number = 1

best_val_accuracy = 0
best_model_filename = ""


for train_index, val_index in kfold.split(x_train):
    # Split data into training and validation sets for this fold.
    train_dataset, test_dataset, val_dataset = SplitDatasetForFolds(train_index, val_index, fold_number)

    # Create a new model instance
    model = create_model()

    # Compile the model
    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    # Set up callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"best_model_fold_{fold_number}.keras", save_best_only=True, monitor="val_binary_accuracy"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_binary_accuracy", factor=0.2, patience=15, min_lr=0.000001),
        keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=10, restore_best_weights=True),
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=25,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=weight_dict
    )

    # Append history
    history_list.append(history.history)

        # Check if this model has the best validation accuracy so far
    if history.history['val_binary_accuracy'][-1] > best_val_accuracy:
        best_val_accuracy = history.history['val_binary_accuracy'][-1]
        best_model_filename = f"best_model_fold_{fold_number}.keras"

    fold_number += 1
    print(f"Training fold {fold_number} completed\n")
    print("------------------------------------------------------------------------------------------------------------------\n")

print("Cross-validation training completed")




# +
# Load the best model
best_model = keras.models.load_model(best_model_filename)

# Now you have the best model, you can evaluate it on the test set
for dataset in [train_dataset, val_dataset]:
    print("Training set" if dataset == train_dataset else "Validation set")
    loss, binary_accuracy, auc, precision, recall = best_model.evaluate(dataset)
    print(f"Loss: {loss}\n Binary Accuracy: {binary_accuracy}\n AUC: {auc}\n Precision: {precision}\n Recall: {recall}\n")

with open(f"metrics.txt", "w") as f:
    f.write(f"Loss: {loss}\n Binary Accuracy: {binary_accuracy}\n AUC: {auc}\n Precision: {precision}\n Recall: {recall}\n") 

# +
import matplotlib.pyplot as plt

# Define a color scheme for metrics
colors = ['b', 'g', 'r', 'y', 'k']

# Plotting the metrics for all folds
def plot_metrics(history_list, metrics, val_metrics, colors):
    num_metrics = len(metrics)
    fig, axs = plt.subplots(nrows=num_metrics, ncols=2, figsize=(28, 20))
    for i, (metric, val_metric) in enumerate(zip(metrics, val_metrics)):
        train_max = max([max(history[metric]) for history in history_list])
        val_max = max([max(history[val_metric]) for history in history_list])
        y_max = max(train_max, val_max)
        for j, history in enumerate(history_list):
            color_index = j % len(colors)  # Get color index for this fold
            color = colors[color_index]     # Get color for this fold
            axs[i, 0].plot(history[metric], label=f'Fold {j+1} {metric}', color=color)
            axs[i, 1].plot(history[val_metric], label=f'Fold {j+1} {val_metric}', linestyle='--', color=color)
        axs[i, 0].set_title(f'{metric.capitalize()} over Folds')
        axs[i, 0].set_xlabel('Epochs')
        axs[i, 0].set_ylabel(metric)
        axs[i, 0].legend()
        axs[i, 0].grid()
        axs[i, 0].set_ylim([0, y_max])  # Set y-axis limit for training plot
        
        axs[i, 1].set_title(f'{val_metric.capitalize()} over Folds')
        axs[i, 1].set_xlabel('Epochs')
        axs[i, 1].set_ylabel(val_metric)
        axs[i, 1].legend()
        axs[i, 1].grid()
        axs[i, 1].set_ylim([0, y_max])  # Set y-axis limit for validation plot
        
    plt.tight_layout()
    plt.show()

# Updated metrics list based on the actual keys from the history dictionary
metrics = ['binary_accuracy', 'loss', 'auc', 'precision', 'recall']
val_metrics = ['val_binary_accuracy', 'val_loss', 'val_auc', 'val_precision', 'val_recall']

# Plot metrics
plot_metrics(history_list, metrics, val_metrics, colors)


# +
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix

# Assuming best_model is already defined and trained
# Generate predictions on the validation set
y_pred_probs = best_model.predict(x_test, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Compute metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_probs)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
print(f"AUC: {auc}")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()


# +
import numpy as np
import matplotlib.pyplot as plt

def view_evaluated_eeg_plots(model, sequences_df, scaler):
    def plot_signals(data, labels, predictions, ids, times):
        total_plots = len(data)
        cols = total_plots // 5
        rows = total_plots // cols
        if total_plots % cols != 0:
            rows += 1
        pos = range(1, total_plots + 1)
        fig = plt.figure(figsize=(40, 30))
        for i, (plot_data, og_label, pred_label, id_, time) in enumerate(zip(data, labels, predictions, ids, times)):
            plt.subplot(rows, cols, pos[i])
            plt.plot(time, plot_data)
            plt.title(f"ID: {id_}\nActual Label: {og_label}\nPredicted Label: {pred_label}")
            fig.subplots_adjust(hspace=0.5)
        plt.show()

    def generate_signals_for_label(label, num_signals=25):
        filtered_df = sequences_df[sequences_df['downsampled_label'] == label]
        sampled_df = filtered_df.sample(n=num_signals, random_state=42)
        data = sampled_df['w_eda']
        times = sampled_df['Time']
        data_array = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in data]
        data_array = np.asarray(data_array).astype(np.float32).reshape(-1, 32, 1)
        labels = sampled_df['downsampled_label'].tolist()
        ids = sampled_df['ID'].tolist()  # Extract IDs
        predictions = (model.predict(data_array, verbose=0) > 0.5).astype(int).flatten()
        return data, labels, predictions, ids, times

    data_0, labels_0, predictions_0, ids_0, times_0 = generate_signals_for_label(0)
    data_1, labels_1, predictions_1, ids_1, times_1 = generate_signals_for_label(1)
    
    print("Plotting signals with label 0:")
    plot_signals(data_0, labels_0, predictions_0, ids_0, times_0)
    
    print("Plotting signals with label 1:")
    plot_signals(data_1, labels_1, predictions_1, ids_1, times_1)

# Call the function with the required arguments
view_evaluated_eeg_plots(best_model, sequences_df, scaler)


# +
import numpy as np
import matplotlib.pyplot as plt


def view_evaluated_eeg_plots(model, sequences_df, scaler, target_id):
    def plot_signals(data, labels, predictions, ids, times):
        total_plots = len(data)
        cols = total_plots // 5
        rows = total_plots // cols
        if total_plots % cols != 0:
            rows += 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3))
        for i, (plot_data, og_label, pred_label, id_, time) in enumerate(zip(data, labels, predictions, ids, times)):
            if len(plot_data) == 0:  # Skip empty plots
                continue
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            color = 'green' if og_label == pred_label else 'red'
            ax.plot(time, plot_data, color=color)
            ax.set_title(f"ID: {id_}\nActual Label: {og_label}\nPredicted Label: {pred_label}")
            ax.set_xlabel('Time')
            ax.set_ylabel('Data')
        plt.tight_layout()
        plt.show()

    def generate_signals_for_id(target_id):
        filtered_df = sequences_df[sequences_df['ID'] == target_id]
        filtered_df = filtered_df.sort_values(by='Time')  # Sort by time
        data = filtered_df['w_eda']
        times = filtered_df['Time']
        data_array = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in data]
        data_array = np.asarray(data_array).astype(np.float32).reshape(-1, 32, 1)
        labels = filtered_df['downsampled_label'].tolist()
        ids = filtered_df['ID'].tolist()  # Extract IDs
        predictions = (model.predict(data_array, verbose=0) > 0.5).astype(int).flatten()
        return data, labels, predictions, ids, times

    data, labels, predictions, ids, times = generate_signals_for_id(target_id)
    
    print(f"Plotting signals for ID: {target_id}")
    plot_signals(data, labels, predictions, ids, times)

    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix')
    plt.show()


