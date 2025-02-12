# Import from .py files
from ml_logic.data import retrieve_data,predict_single_value,predict_X_test_scaled
from ml_logic.preprocessor import preprocess_data
from ml_logic.model import get_split_data, prepare_train_data,\
                            prepare_test_data, define_LSTM_model,\
                                train_lstm_model, save_fitted_model,\
                                    save_fitted_model,save_fitted_scaler,\
                                        load_fitted_model, load_fitted_scaler


def predict_co2_level():
    # Get the historical data
    path = 'data/raw_data/co2_levels.csv'
    co2_levels = retrieve_data(path)

    # # ******** Preprocessing **********
    co2_levels = preprocess_data(co2_levels)
    # ***********************************

    # Split the data to train set and test set
    train, test = get_split_data(co2_levels, train_size = 0.85)

    # Prepare train data (scaling + sequence creation)
    window = 1 #Observation window
    feature = 1 #Number of features
    X_train_scaled, y_train_scaled, scaler = prepare_train_data(train, window)
    X_test_scaled, y_test_scaled, dates = prepare_test_data(test, window, scaler)

    # Create LSTM model
    model = define_LSTM_model(window, feature)

    # Train the LSTM model
    fitted_model = train_lstm_model(model, X_train_scaled, y_train_scaled)

    # Reloading phase
    path_fitted_model = "data/processed_data/my_LSTM_model.h5"
    path_fitted_scaler = "data/processed_data/my_final_scaler.pkl"

    # Let's save the fitted scaler
    save_fitted_scaler(scaler, path_fitted_scaler)

    # Save fitted model
    save_fitted_model(fitted_model, path_fitted_model)

    model_reloaded = load_fitted_model(path_fitted_model)
    scaler_reloaded = load_fitted_scaler(path_fitted_scaler)

    # # Make prediction (single value)
    # prediction = predict_single_value(400, scaler_reloaded, model_reloaded)

    # Make prediction (X_test_scaled)
    predictions_df = predict_X_test_scaled(X_test_scaled, scaler_reloaded, model_reloaded, dates)
    print(predictions_df)


if __name__ == '__main__':
    try:
       predict_co2_level()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
