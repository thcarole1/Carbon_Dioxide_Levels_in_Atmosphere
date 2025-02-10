# Retrieve data
import pandas as pd

def predict_X_test_scaled_api(X, scaler, fitted_model, dates) -> pd.DataFrame:
    '''
    Predict an array of scaled values.
    '''
    y_pred = fitted_model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    return pd.DataFrame(y_pred, index=dates, columns=['predictions'])
