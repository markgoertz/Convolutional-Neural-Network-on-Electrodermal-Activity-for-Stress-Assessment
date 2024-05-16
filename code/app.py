from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
model = load_model('D:/Master of Applied IT/code/best_model.keras')
scaler = StandardScaler()  # Initialize StandardScaler for preprocessing

@app.route('/')
def welcome():
    return 'Welcome to Test App API!'

# Preprocess function for input data
def preprocess_input(data):
    max_sequence_length = 128
    num_chunks = len(data) // max_sequence_length + 1
    predictions = []
    for i in range(num_chunks):
        start_idx = i * max_sequence_length
        end_idx = min((i + 1) * max_sequence_length, len(data))
        chunk = data[start_idx:end_idx]
        padded_chunk = pad_sequences([chunk], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
        preprocessed_chunk = np.asarray(padded_chunk).astype(np.float32).reshape(-1, max_sequence_length, 1)
        predictions.append(model.predict(preprocessed_chunk))
    return predictions

def your_label_mapping_function(index):
    labels = ['no-stress', 'stress']  # Assuming these are your original labels
    return labels[index]

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json(force=True)
        input_data_eda = data['w_eda']

        # Preprocess input data
        input_data_eda_preprocessed = preprocess_input(input_data_eda)

        # Perform prediction
        probabilities = model.predict(input_data_eda_preprocessed)

        # Get the index of the maximum probability
        predicted_label_index = np.argmax(probabilities)

        # Map the index to the original label
        predicted_label = your_label_mapping_function(predicted_label_index)  # Replace with your label mapping function

        # Return predicted label
        return jsonify({'predicted_label': predicted_label})

    except Exception as e:
        # Handle exceptions gracefully
        error_message = str(e)
        return jsonify({'error': error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)
