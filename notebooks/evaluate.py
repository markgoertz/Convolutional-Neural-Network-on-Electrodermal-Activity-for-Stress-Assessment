from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from dvclive import Live
from helper import F1Score
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score
import ast

def create_segments(time_indices, values, stress_periods):
    segments = []
    current_segment = {"x": [], "y": [], "color": "blue"}
    for t, v in zip(time_indices, values):
        in_stress = any(start <= t < end for start, end in stress_periods)
        color = "red" if in_stress else "blue"

        if color != current_segment["color"] and current_segment["x"]:
            segments.append(current_segment)
            current_segment = {"x": [], "y": [], "color": color}

        current_segment["x"].append(t)
        current_segment["y"].append(v)
        current_segment["color"] = color

    if current_segment["x"]:
        segments.append(current_segment)
    
    return segments

def plot_physiological_signals(data, model, subject_id):
    
    with Live() as live:
    # Load x_test and y_test from pickle files
        df = pd.read_csv(file_path)

        # Convert list-like strings to actual lists
        for column in ['EDA', 'TEMP', 'BVP', 'ACC']:
            df[column] = df[column].apply(ast.literal_eval)

        # Define sample rates (in Hz)
        sampling_rates = {
            'EDA': 4,   # 4 Hz for EDA
            'TEMP': 4,  # 4 Hz for TEMP
            'BVP': 32,  # 32 Hz for BVP
            'ACC': 32   # 32 Hz for ACC
        }

        # Convert the data into a dictionary format suitable for model.predict
        data_dict = {
            'EDA': np.array(df['EDA'].tolist()).reshape(-1, 32, 1),
            'TEMP': np.array(df['TEMP'].tolist()).reshape(-1, 32, 1),
            'BVP': np.array(df['BVP'].tolist()).reshape(-1, 256, 1),
            'ACC': np.array(df['ACC'].tolist()).reshape(-1, 256, 1)
        }

        # Generate predictions using the model
        y_pred_probs = model.predict([data_dict['EDA'], data_dict['BVP'], data_dict['TEMP'], data_dict['ACC']])
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y = df['labels']

        # Store y_pred in the dataframe
        df['y_pred'] = y_pred
        df['y_pred_probs'] = y_pred_probs

        # Compute the confusion matrix and metrics
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_probs)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        plt.title(f'Confusion Matrix of subject {subject_id}')
        confusion_matrix_path = f"images/evaluation/plots/confusion_matrix_{subject_id}.png"
        plt.savefig(confusion_matrix_path, dpi=120)
        
        live.log_image("confusion_matrix", confusion_matrix_path)
        
    # Prepare data for plotting
        eda_values = [value for segment in df['EDA'] for value in segment]
        temp_values = [value for segment in df['TEMP'] for value in segment]
        bvp_values = [value for segment in df['BVP'] for value in segment]
        acc_values = [value for segment in df['ACC'] for value in segment]

        # Create time axis for each signal
        eda_time_indices = [i / sampling_rates['EDA'] for i in range(len(eda_values))]
        temp_time_indices = [i / sampling_rates['TEMP'] for i in range(len(temp_values))]
        bvp_time_indices = [i / sampling_rates['BVP'] for i in range(len(bvp_values))]
        acc_time_indices = [i / sampling_rates['ACC'] for i in range(len(acc_values))]

        # Determine stress periods
        actual_stress_times = df[df['labels'] == 1].index * 8  # Each index represents 8 seconds
        stress_periods = [(start, start + 8) for start in actual_stress_times]

        # Determine predicted stress periods
        predicted_stress_times = df[df['y_pred'] == 1].index * 8  # Each index represents 8 seconds
        predicted_stress_periods = [(start, start + 8) for start in predicted_stress_times]

        # Create figure
        fig = go.Figure()

        # Plot each physiological signal with color segments
        for signal_name, time_indices, values in zip(
            ['EDA', 'TEMP', 'BVP', 'ACC'],
            [eda_time_indices, temp_time_indices, bvp_time_indices, acc_time_indices],
            [eda_values, temp_values, bvp_values, acc_values]
        ):

            segments = create_segments(time_indices, values)
            for segment in segments:
                fig.add_trace(go.Scatter(
                    x=segment["x"],
                    y=segment["y"],
                    mode='lines',
                    name=signal_name,
                    line=dict(color=segment["color"]),
                    showlegend=True  # Only show "blue" legend once per signal
                ))

        # Add predicted stress periods as reddish-white background
        for start, end in predicted_stress_periods:
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(width=0)
            )

        # Update layout with background color and axis titles
        fig.update_layout(
            title='Physiological Signals Over Time with Stress Moments',
            xaxis_title='Time (seconds)',
            yaxis_title='Signal Value',
            legend_title='Signals',
            template='plotly',
        )

        # Save and log the plot image
        plot_path = "images/evaluation/plots/physiological_signals_plot.png"
        fig.write_image(plot_path)
        live.log_image("physiological_signals", plot_path)

# Evaluation function
def evaluate():
   
    # Load model and helper data
    model_path = Path("models") / "best_model.h5"
    model = load_model(model_path, custom_objects={'F1Score': F1Score})
    print(model.summary())
    
    filepath = Path("data/WESAD/S16") / "S16_unknown_data.csv"
    
    plot_physiological_signals(filepath, model, "S16")
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()
