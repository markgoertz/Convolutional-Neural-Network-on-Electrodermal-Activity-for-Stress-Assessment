import pickle
import pandas as pd
import json

def predict(data):

    data = {"success": False}

    pkl_filename = "/d:/Master of Applied IT/code/best_model.keras"  # Update the file path
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    if type(data) == dict:
        df = pd.DataFrame(data)
    else:
        df = data
    
    y_pred = model.predict(df)
    
    if y_pred == 0:
        return 'Extremely Weak'
    elif y_pred == 1:
        return 'Weak'
    elif y_pred == 2:
        return 'Normal'
    elif y_pred == 3:
        return 'Overweight'
    elif y_pred == 4:
        return 'Obesity'
    elif y_pred == 5:
        return 'Extreme Obesity'
