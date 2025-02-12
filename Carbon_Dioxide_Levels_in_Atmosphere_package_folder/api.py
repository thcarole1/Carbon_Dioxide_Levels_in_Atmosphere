# Basic libraries
import pandas as pd
import io
import urllib.parse
import zipfile
from io import BytesIO
from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import FileResponse

# Import from .py files
from Carbon_Dioxide_Levels_in_Atmosphere_package_folder.api_functions.data_api import predict_X_test_scaled_api
from Carbon_Dioxide_Levels_in_Atmosphere_package_folder.api_functions.preprocessor_api import preprocess_data_api
from  Carbon_Dioxide_Levels_in_Atmosphere_package_folder.api_functions.model_api import get_split_data_api, \
                                                                                        prepare_test_data_api,\
                                                                                        prepare_train_data_api,\
                                                                                        save_fitted_scaler_api,\
                                                                                        define_LSTM_model_api,\
                                                                                        train_lstm_model_api, \
                                                                                        save_fitted_model_api,\
                                                                                        load_fitted_scaler_api,\
                                                                                        load_fitted_model_api
app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'New project': 'This is the first app of my new project !'}

@app.post("/upload_and_predict")
async def create_upload_file(file: UploadFile):
    # Read file contents as bytes
    contents = await file.read()

    # Convert bytes to a StringIO object so pandas can read it as a file
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

     # # ******** Preprocessing **********
    co2_levels = preprocess_data_api(df)
    # ***********************************

    # Split the data to train set and test set
    train, test = get_split_data_api(co2_levels, train_size = 0.85)

    # Prepare train data (scaling + sequence creation)
    window = 1 #Observation window
    feature = 1 #Number of features
    X_train_scaled, y_train_scaled, scaler = prepare_train_data_api(train, window)
    X_test_scaled, y_test_scaled, dates = prepare_test_data_api(test, window, scaler)

    # Let's save the fitted scaler
    save_fitted_scaler_api(scaler)

    # Create LSTM model
    model = define_LSTM_model_api(window, feature)

    # Train the LSTM model
    fitted_model = train_lstm_model_api(model, X_train_scaled, y_train_scaled)

    # Save fitted model
    save_fitted_model_api(fitted_model)

    # Make prediction (X_test_scaled)
    predictions = predict_X_test_scaled_api(X_test_scaled, scaler, fitted_model, dates)
    print(predictions)

    # Create a ZIP archive in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add Pandas dataframes as CSV
        export_list = [train, test, predictions]
        export_list_str = ['train', 'test', 'predictions']

        for index, dataframe in enumerate(export_list):
            csv_buffer = BytesIO()
            dataframe.to_csv(csv_buffer, index=True)
            csv_buffer.seek(0)
            zip_file.writestr(f"{export_list_str[index]}.csv", csv_buffer.read())

    zip_buffer.seek(0)  # Reset buffer cursor for reading

    # Save the BytesIO to an actual file
    with open("zip_buffer_file", "wb") as f:
        f.write(zip_buffer.getvalue())

    return FileResponse("zip_buffer_file",
                        media_type="application/zip",
                        filename="co2_levels_training_and_prediction_data.zip")

@app.post("/predict")
async def create_upload_file(file: UploadFile):
    # Read file contents as bytes
    contents = await file.read()

    # Convert bytes to a StringIO object so pandas can read it as a file
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

     # # ******** Preprocessing **********
    co2_levels = preprocess_data_api(df)
    # ***********************************

    #Parameters
    window = 1 #Observation window

    #Reload scaler
    path_fitted_scaler = "data/processed_data/my_final_scaler.pkl"
    scaler_reloaded = load_fitted_scaler_api(path_fitted_scaler)

    #Reload fitted model
    path_fitted_model = "data/processed_data/my_LSTM_model.h5"
    model_reloaded = load_fitted_model_api(path_fitted_model)

    X_test_scaled, y_test_scaled, dates = prepare_test_data_api(co2_levels, window,scaler_reloaded)

    # Make prediction (X_test_scaled)
    predictions = predict_X_test_scaled_api(X_test_scaled, scaler_reloaded, model_reloaded, dates)

    # Compute y_test (no scaling)
    y_test = scaler_reloaded.inverse_transform(y_test_scaled)
    y_test = pd.DataFrame(y_test, index=dates, columns=['test'])

    # Create a ZIP archive in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add Pandas dataframes as CSV
        export_list = [y_test, predictions]
        export_list_str = ['y_test', 'predictions']

        for index, dataframe in enumerate(export_list):
            csv_buffer = BytesIO()
            dataframe.to_csv(csv_buffer, index=True)
            csv_buffer.seek(0)
            zip_file.writestr(f"{export_list_str[index]}.csv", csv_buffer.read())

    zip_buffer.seek(0)  # Reset buffer cursor for reading

    # Save the BytesIO to an actual file
    with open("zip_buffer_file", "wb") as f:
        f.write(zip_buffer.getvalue())

    return FileResponse("zip_buffer_file",
                        media_type="application/zip",
                        filename="co2_levels_prediction_data.zip")

@app.get("/predict_csv")
async def predict(csv_data: str = Query(..., description="URL-encoded CSV content")):

    # URL-decode the incoming CSV content
    decoded_csv = urllib.parse.unquote(csv_data)

    # Wrap the decoded CSV content in a StringIO object so that pandas can read it as a file
    csv_io = io.StringIO(decoded_csv)
    try:
        df = pd.read_csv(csv_io)
    except Exception as e:
        return {"error": "Failed to parse CSV", "detail": str(e)}

     # # ******** Preprocessing **********
    co2_levels = preprocess_data_api(df)
    # ***********************************

    #Parameters
    window = 1 #Observation window

    #Reload scaler
    path_fitted_scaler = "data/processed_data/my_final_scaler.pkl"
    scaler_reloaded = load_fitted_scaler_api(path_fitted_scaler)

    #Reload fitted model
    path_fitted_model = "data/processed_data/my_LSTM_model.h5"
    model_reloaded = load_fitted_model_api(path_fitted_model)

    X_test_scaled, y_test_scaled, dates = prepare_test_data_api(co2_levels, window,scaler_reloaded)

    # Make prediction (X_test_scaled)
    predictions = predict_X_test_scaled_api(X_test_scaled, scaler_reloaded, model_reloaded, dates)

    # Compute y_test (no scaling)
    y_test = scaler_reloaded.inverse_transform(y_test_scaled)
    y_test = pd.DataFrame(y_test, index=dates, columns=['test'])

    # Create a ZIP archive in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add Pandas dataframes as CSV
        export_list = [y_test, predictions]
        export_list_str = ['y_test', 'predictions']

        for index, dataframe in enumerate(export_list):
            csv_buffer = BytesIO()
            dataframe.to_csv(csv_buffer, index=True)
            csv_buffer.seek(0)
            zip_file.writestr(f"{export_list_str[index]}.csv", csv_buffer.read())

    zip_buffer.seek(0)  # Reset buffer cursor for reading

    # Save the BytesIO to an actual file
    with open("zip_buffer_file", "wb") as f:
        f.write(zip_buffer.getvalue())

    return FileResponse("zip_buffer_file",
                        media_type="application/zip",
                        filename="co2_levels_prediction_data.zip")
