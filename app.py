from flask import Flask, request, jsonify
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utilities.Helper import F1Score
import ast

app = Flask(__name__)

# Folder where the uploaded files will be saved
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained Keras model
MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(MAIN_PATH, 'models', 'best_model.h5')
model = load_model(MODEL_PATH, custom_objects={'F1Score': F1Score})

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(file_path):
    # Load the data
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
        'Time': np.array(df['StartTime'].tolist()).reshape(-1, 1),
        'EDA': np.array(df['EDA'].tolist()).reshape(-1, 32, 1),
        'TEMP': np.array(df['TEMP'].tolist()).reshape(-1, 32, 1),
        'BVP': np.array(df['BVP'].tolist()).reshape(-1, 256, 1),
        'ACC': np.array(df['ACC'].tolist()).reshape(-1, 256, 1)
    }

    print(f"Shapes: {data_dict['EDA'].shape}, {data_dict['BVP'].shape}, {data_dict['TEMP'].shape}, {data_dict['ACC'].shape}")
    
    return data_dict

# Route to test the API
@app.route('/')
def home():
    return 'Hello, World!'

# Route to predict the stress levels
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has the 'file' part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # If the file is allowed, save it
    if file and allowed_file(file.filename):
        # Save the file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Preprocess the data from the CSV
        data_dict = preprocess_data(filename)
        
        # Make the prediction
        predictions = {}         
        pred = model.predict([data_dict['EDA'], data_dict['BVP'], data_dict['TEMP'], data_dict['ACC']])
        
        # Extract timestamps and match with the predictions
        timestamps = data_dict['Time'].flatten()  # Convert to 1D array
        
        # Loop through timestamps and predictions to pair them
        for i, timestamp in enumerate(timestamps):
            predictions[str(timestamp)] = float(pred[i][0])  # Assuming prediction is a scalar per time point
        

        return jsonify(predictions), 200
    
    return jsonify({"error": "Invalid file format. Only CSV allowed."}), 400

# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', debug=True)
