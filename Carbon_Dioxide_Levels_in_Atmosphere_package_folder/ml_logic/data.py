# Retrieve data
import pandas as pd
import numpy as np

def retrieve_data(path:str) -> pd.DataFrame:
    '''
    This function gets a str as input. It represents the path to the raw data (.csv).
    It returns a pandas Dataframe containing the raw data.
    '''
    print("✅ data has been retrieved.")
    return pd.read_csv(path)

def predict_single_value(value, scaler, model):
    '''
    Scales a single value of data, then creates a sequence of 1 unique value.
    Outputs : prediction from model
    '''
    val = np.array([[value]])
    val = scaler.transform(val)
    val = val.reshape(1, 1, 1)
    prediction = model.predict(val)
    prediction = scaler.inverse_transform(prediction)
    print("✅ prediction has been computed.")
    return prediction

def predict_X_test_scaled(X, scaler, fitted_model, dates) -> pd.DataFrame:
    '''
    Predict an array of scaled values.
    '''
    y_pred = fitted_model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    return pd.DataFrame(y_pred, index=dates)
