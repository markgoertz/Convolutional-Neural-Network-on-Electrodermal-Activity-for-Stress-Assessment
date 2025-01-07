import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import binary_closing

class PhysiologicalSignalAnalyzer:
    def __init__(self, file_path, model, user_id):
        """
        Initialize the analyzer with file path, model, and user ID.
        """
        self.file_path = file_path
        self.model = model
        self.user_id = user_id
        self.data = None
        self.sampling_rates = {
            'EDA': 4,   # 4 Hz for EDA
            'BVP': 32,  # 32 Hz for BVP
            'ACC': 32   # 32 Hz for ACC
        }
        self.predictions = None
        self.metrics = None

    def load_and_preprocess_data(self):
        """
        Load and preprocess data from CSV.
        """
        df = pd.read_csv(self.file_path)

        for column in ['EDA', 'BVP', 'ACC']:
            df[column] = df[column].apply(ast.literal_eval)

        self.data = df

    def predict(self, window_input, min_period_input, center, threshold, gap_tolerance, min_isolated_region_length):
        """
        Predicts and processes physiological signal data to generate final smoothed predictions.
        Parameters:
        window_input (int): The size of the rolling window for computing the rolling mean.
        min_period_input (int): Minimum number of observations in the window required to have a value.
        center (bool): Whether to set the labels at the center of the window.
        threshold (float): The threshold for converting predicted probabilities to binary predictions.
        gap_tolerance (int): The maximum size of gaps to fill in the binary predictions.
        min_isolated_region_length (int): The minimum length of isolated regions to retain in the binary predictions.
        Raises:
        ValueError: If data is not loaded before calling this method.
        Returns:
        None: The function updates the instance's data attribute with the final smoothed predictions.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call `load_and_preprocess_data()` first.")

        # Step 1: Prepare input data for prediction
        data_dict = {
            'EDA': np.array(self.data['EDA'].tolist()).reshape(-1, 32, 1),
            'BVP': np.array(self.data['BVP'].tolist()).reshape(-1, 256, 1),
            'ACC': np.array(self.data['ACC'].tolist()).reshape(-1, 256, 1)
        }

        # Step 2: Predict probabilities and convert them to binary predictions
        y_pred_probs = self.model.predict([data_dict['EDA'], data_dict['BVP'], data_dict['ACC']])
        y_pred = (y_pred_probs > threshold).astype(int).flatten()

        # Add raw predictions to the data
        self.data['y_pred'] = y_pred
        self.data['y_pred_probs'] = y_pred_probs

        # Step 3: Compute rolling mean to smooth predictions
        rolling_mean = self.data['y_pred'].rolling(window=window_input, min_periods=min_period_input, center=center).mean()
        self.data['y_pred_rolling_mean'] = rolling_mean

        # Step 4: Post-process predictions to fill gaps and remove small regions
        # Convert rolling mean into binary predictions
        smoothed_pred = (rolling_mean > threshold).astype(int).values

        # Fill small gaps (e.g., small white spaces between red boxes)
        smoothed_pred = binary_closing(smoothed_pred, structure=np.ones(gap_tolerance))

        # Remove small isolated regions (small red boxes)
        diffs = np.diff(np.concatenate(([0], smoothed_pred, [0])))  # Detect region boundaries
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_isolated_region_length:
                smoothed_pred[start:end] = 0  # Remove small regions

        # Ensure predictions at the start are included if valid
        if smoothed_pred[0] == 1:
            smoothed_pred[:gap_tolerance] = 1

        # Step 5: Update data with final smoothed predictions
        self.data['final_pred'] = smoothed_pred

        # Store final predictions for further use
        self.predictions = smoothed_pred

    def calculate_metrics(self):
        """
        Calculate evaluation metrics and confusion matrix.
        """
        if self.predictions is None:
            raise ValueError("Predictions not generated. Call `predict()` first.")

        y_true = self.data['labels']
        conf_matrix = confusion_matrix(y_true, self.data['y_pred'])
        precision = precision_score(y_true, self.data['y_pred'])
        recall = recall_score(y_true, self.data['y_pred'])
        accuracy = accuracy_score(y_true, self.data['y_pred'])
        auc = roc_auc_score(y_true, self.data['y_pred'])

        self.metrics = {
            'conf_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'auc': auc
        }
        # Print metrics in a readable format
        print(f"Evaluation Metrics:\n")
        # print(f"Confusion Matrix:\n{self.metrics['conf_matrix']}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"AUC: {self.metrics['auc']:.4f}")
        
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix.
        """
        if self.metrics is None:
            raise ValueError("Metrics not calculated. Call `calculate_metrics()` first.")

        conf_matrix = self.metrics['conf_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix of ID: {self.user_id}')
        plt.savefig('val_model_results.png', dpi=120)
        plt.show()

    def plot_signals(self):
        """
        Plot physiological signals with stress moments highlighted.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call `load_and_preprocess_data()` first.")

        # Prepare data for plotting
        eda_values = [value for segment in self.data['EDA'] for value in segment]
        bvp_values = [value for segment in self.data['BVP'] for value in segment]
        acc_values = [value for segment in self.data['ACC'] for value in segment]

        # convert the values into time. Calculated by the number of values / sample-rate.
        eda_time_indices = [i / self.sampling_rates['EDA'] for i in range(len(eda_values))]
        bvp_time_indices = [i / self.sampling_rates['BVP'] for i in range(len(bvp_values))]
        acc_time_indices = [i / self.sampling_rates['ACC'] for i in range(len(acc_values))]

        # Plot the actual labels of the WESAD dataset.    
        actual_stress_times = self.data[self.data['labels'] == 1].index * 8
        stress_periods = [(start, start + 8) for start in actual_stress_times]

        # Plot the model's predicted values. 
        predicted_stress_times = self.data[self.data['final_pred'] == 1].index * 8
        predicted_stress_periods = [(start, start + 8) for start in predicted_stress_times]

        def create_segments(time_indices, values):
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

        fig = go.Figure()

        for signal_name, time_indices, values in zip(
            ['EDA', 'BVP', 'ACC'],
            [eda_time_indices, bvp_time_indices, acc_time_indices],
            [eda_values, bvp_values, acc_values]
        ):
            segments = create_segments(time_indices, values)
            for segment in segments:
                fig.add_trace(go.Scatter(
                    x=segment["x"],
                    y=segment["y"],
                    mode='lines',
                    name=signal_name,
                    line=dict(color=segment["color"]),
                    showlegend=True
                ))

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

        fig.update_layout(
            title=f'Physiological Signals Over Time with Stress Moments of ID: {self.user_id}',
            xaxis_title='Time (seconds)',
            yaxis_title='Signal Value',
            legend_title='Signals',
            template='plotly',
        )

        fig.show()
        
        
    def analyze_predictions(self):
        """
        Analyze the predictions and print indices for different cases, and return the best (max) score for TP/FP and worst (min) score for TN/FN.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call `load_and_preprocess_data()` first.")
        if 'labels' not in self.data.columns or 'y_pred_probs' not in self.data.columns:
            raise ValueError("Required columns 'labels' and 'y_pred_probs' are not present in the data.")

        # Find indices where labels are 1 and y_pred_probs are greater than 0.8 (True Positive)
        indices_1 = self.data[(self.data['labels'] == 1) & (self.data['y_pred_probs'] > 0.8)]
        indices_1 = indices_1.index.tolist()
        best_score_1 = indices_1[indices_1.index(max(indices_1, key=lambda i: self.data.loc[i, 'y_pred_probs']))] if indices_1 else None

        # Find indices where labels are 0 and y_pred_probs are greater than 0.8 (False Positive)
        indices_0 = self.data[(self.data['labels'] == 0) & (self.data['y_pred_probs'] > 0.8)]
        indices_0 = indices_0.index.tolist()
        best_score_0 = indices_0[indices_0.index(max(indices_0, key=lambda i: self.data.loc[i, 'y_pred_probs']))] if indices_0 else None

        # Find indices where labels are 0 and y_pred_probs are less than 0.1 (True Negative)
        indices_0_low = self.data[(self.data['labels'] == 0) & (self.data['y_pred_probs'] < 0.1)]
        indices_0_low = indices_0_low.index.tolist()
        worst_score_0 = indices_0_low[indices_0_low.index(min(indices_0_low, key=lambda i: self.data.loc[i, 'y_pred_probs']))] if indices_0_low else None

        # Find indices where labels are 1 and y_pred_probs are less than 0.2 (False Negative)
        indices_1_low = self.data[(self.data['labels'] == 1) & (self.data['y_pred_probs'] < 0.2)]
        indices_1_low = indices_1_low.index.tolist()
        worst_score_1 = indices_1_low[indices_1_low.index(min(indices_1_low, key=lambda i: self.data.loc[i, 'y_pred_probs']))] if indices_1_low else None

        # Print the indices in a more readable format
        print("True positive Stress: Indices where label is 1 and y_pred_probs > 0.8:")
        print(", ".join(map(str, indices_1)))
        print("\nFalse positive Stress: Indices where label is 0 and y_pred_probs > 0.8:")
        print(", ".join(map(str, indices_0)))
        print("\nTrue negative Stress: Indices where label is 0 and y_pred_probs < 0.1:")
        print(", ".join(map(str, indices_0_low)))
        print("\nFalse Negative Stress: Indices where label is 1 and y_pred_probs < 0.2:")
        print(", ".join(map(str, indices_1_low)))
