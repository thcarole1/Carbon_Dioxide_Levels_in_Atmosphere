
# Basic imports
import numpy as np
import pandas as pd

# Saving/loading models
import joblib
import pickle

# Scaling the data
from sklearn.preprocessing import StandardScaler

# Create RNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError

def get_split_data(df:pd.DataFrame, train_size : float):
    '''
    This function gets a Pandas Dataframe and an float that represents
    the size of my train dataset relative to all the available data.
    It return a train dataset and a test dataset.
    '''
    index= round(train_size*df.shape[0])
    train = df.iloc[:index]
    test = df.iloc[index:]
    print("✅ data has been plit into train and test data.")
    return train, test

def prepare_train_data(df, look_back=10):
    '''
    Scales the train data, then create sequences of train data.
    Outputs : training data X, training target y and fitted scaler
    '''
    feature = ['Carbon Dioxide (ppm)']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature])
    X, y = [], []
    for i in range(look_back, len(scaled_data)-1):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
    print("✅ Train data been prepared.")
    return np.array(X), np.array(y), scaler

def prepare_test_data(df, look_back=10, scaler=None):
    '''
    Scales the test data, then create sequences of test data.
    Outputs : test data X, test target y and associated dates
    '''
    feature = ['Carbon Dioxide (ppm)']
    scaled_data = scaler.transform(df[feature])
    X, y = [], []
    dates = []
    for i in range(look_back, len(scaled_data)-1):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
        dates.append(df.index[i])
    print("✅ Test data been prepared.")
    return np.array(X), np.array(y), dates

def define_LSTM_model(window, feature):
    '''
    Defines the structure of the deep learning model, then defines compilation parameters.
    Outputs LSTM model instance.
    '''
    # 1- RNN Architecture
    model = Sequential()
    model.add(layers.LSTM(units=500, activation='tanh',
                          input_shape=(window,feature)))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation="linear"))

    # 2- Compilation
    model.compile(loss='mse', optimizer='adam',
                  metrics =['mse', 'mae', RootMeanSquaredError(), MeanAbsolutePercentageError()])
    print("✅ LSTM model created.")
    return model

def train_lstm_model(model, X, y):
    '''
    This function receives an LSTM model, X (scaled) and y(scaled),
    trains the model and returns the fitted model.
    '''
    # Fitting model parameter
    es = EarlyStopping(patience = 30, restore_best_weights=True)

    # Fitting model
    history = model.fit(x=X,
                        y=y,
                        batch_size=16,
                        epochs=1000,
                        verbose=0,
                        callbacks=[es],
                        validation_split=0.3,
                        shuffle=False)
    print("✅ LSTM model fitted.")
    return model

def save_fitted_model(fitted_model):
    '''
    Saving the best fitted model
    '''
    path = "data/processed_data/my_LSTM_model.pkl"
    # joblib.dump(fitted_model, path)
    with open(path, 'wb') as file:
        pickle.dump(fitted_model, file)
    print("✅ LSTM  fitted model has been saved.")

def save_fitted_scaler(scaler):
    '''
    Saving the scaler fitted on train data (if new values, fit the scaler before using the LSTM model)
    '''
    path = "data/processed_data/my_final_scaler.pkl"
    # joblib.dump(scaler, path)
    with open(path, 'wb') as file:
        pickle.dump(scaler, file)
    print("✅ Fitted scaler has been saved.")

def load_fitted_model(path_fitted_model):
    '''
    Reloading the best fitted model
    '''
    print("✅ LSTM  fitted model has been loaded.")
    with open(path_fitted_model, 'rb') as file:
        final_model_reloaded = pickle.load(file)
    return final_model_reloaded

def load_fitted_scaler(path_fitted_scaler):
    '''
    Reloading the saved scaler
    '''
    with open(path_fitted_scaler, 'rb') as file:
        final_scaler_reloaded = pickle.load(file)
    print("✅ Fitted scaler has been loaded.")
    return final_scaler_reloaded
