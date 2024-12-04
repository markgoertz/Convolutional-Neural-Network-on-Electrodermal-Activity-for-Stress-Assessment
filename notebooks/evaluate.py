from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from dvclive import Live
from helper import F1Score
import plotly.graph_objects as go

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

def plot_physiological_signals(x_test_path, y_test_path, model):
    
    with Live() as live:
        # Load x_test and y_test from pickle files
        with open(x_test_path, "rb") as f:
            x_test = pickle.load(f)
        with open(y_test_path, "rb") as f:
            y_test = pickle.load(f)

        # Ensure x_test contains required keys
        required_keys = ['EDA', 'TEMP', 'BVP', 'ACC']
        for key in required_keys:
            if key not in x_test:
                raise ValueError(f"Missing required key '{key}' in x_test pickle file.")

        # Extract signals
        eda, temp, bvp, acc = x_test['EDA'], x_test['TEMP'], x_test['BVP'], x_test['ACC']

        # Reshape and preprocess signals for the model
        data_dict = {
            'EDA': np.array(eda).reshape(-1, 32, 1),
            'TEMP': np.array(temp).reshape(-1, 32, 1),
            'BVP': np.array(bvp).reshape(-1, 256, 1),
            'ACC': np.array(acc).reshape(-1, 256, 1)
        }

        # Make predictions
        y_pred_probs = model.predict([data_dict['EDA'], data_dict['BVP'], data_dict['TEMP'], data_dict['ACC']])
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        # Compute metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_probs)

        # Plot and log the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        confusion_matrix_path = "images/evaluation/plots/confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=120)
        
        live.log_image("confusion_matrix", confusion_matrix_path)

        # Plot the signals with true and predicted stress periods
        eda_time_indices = np.arange(len(eda)).tolist()


        stress_periods = [(i * 8, (i + 1) * 8) for i, label in enumerate(y_test) if label == 1]
        predicted_stress_periods = [(i * 8, (i + 1) * 8) for i, pred in enumerate(y_pred) if pred == 1]

        fig = go.Figure()

        for signal_name, time_indices, values in zip(
            ['EDA'],
            [eda_time_indices],
            [eda]
        ):
            fig.add_trace(go.Scatter(
                x=time_indices,
                y=values,
                mode='lines',
                name=signal_name,
                line=dict(color='blue'),
                showlegend=True
            ))

        for start, end in stress_periods:
            fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1,
                        xref="x", yref="paper", fillcolor="rgba(0,255,0,0.2)", line=dict(width=0))

        for start, end in predicted_stress_periods:
            fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1,
                        xref="x", yref="paper", fillcolor="rgba(255,0,0,0.3)", line=dict(width=0))

        fig.update_layout(title='Physiological Signals with Stress Periods',
                        xaxis_title='Time (seconds)',
                        yaxis_title='Signal Value',
                        legend_title='Signals')

        plot_path = "images/evaluation/plots/physiological_signals_plot.png"
        fig.write_image(plot_path)
        
        live.log_image("physiological_signals", plot_path)

        return fig

# Evaluation function
def evaluate():
   
    # Load model and helper data
    model_path = Path("models") / "best_model.h5"
    model = load_model(model_path, custom_objects={'F1Score': F1Score})
    print(model.summary())
    
    plot_physiological_signals(
        "data/results/x_test_1.pkl",
        "data/results/y_test_1.pkl",
        model,
    )
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()
