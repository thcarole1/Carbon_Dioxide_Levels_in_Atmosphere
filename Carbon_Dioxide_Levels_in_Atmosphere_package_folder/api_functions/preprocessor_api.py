# Preprocess data
import pandas as pd
import math

def manage_time_index_api(df)-> pd.DataFrame:
    '''
    This function gets  adataframe as an input.
    Creates a datetime index and drop columns that are no longer useful.
    '''
    df['Year_Month'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')

    # Let's drop year and month column
    df.drop(columns=["Year", "Month","Decimal Date"], inplace=True, axis=1)

    # Let's set Year_Month as new index
    df.set_index("Year_Month", inplace=True)

    print("✅ Time index is set.")
    return df

def drop_unsued_cols_api(df)-> pd.DataFrame:
    '''
    This function gets a pandas dataframe as input then drops unsused columns.
    It returns a Datatframe.
    '''
    df.drop(columns=["Seasonally Adjusted CO2 (ppm)",
                         "Carbon Dioxide Fit (ppm)",
                         "Seasonally Adjusted CO2 Fit (ppm)"], inplace=True, axis=1)
    print("✅ Unused colums have been dropped.")
    return df

def drop_time_series_duplicates_api(df:pd.DataFrame) -> pd.DataFrame:
    '''
    This function gets a Pandas Dataframe as input, and drops duplicates rows if any.
    The index of the input dataframe is related to a date, or duration mark.
    '''
    if df.reset_index().duplicated().sum() > 0 :
        nb_duplicates = df.reset_index().duplicated().sum()
        df.drop_duplicates()
        print(f'{nb_duplicates} have been dropped !')
    else:
        print('No duplicates found !')
    print("✅ Duplicates have been dealt with.")
    return df

def drop_head_missing_values_api(df:pd.DataFrame)-> pd.DataFrame:
    '''
    This function gets a Pandas Dataframe as input. It drops all the
    missing values (i.e nan values) at the head of the dataframe, if any
    '''
    for i in range(len(df)):
        if math.isnan(df.head(1).values[0][0]) == True:
            df.drop(index = df.head(1).index, inplace=True)
        else:
            break
    print("✅ Missing values dealt at the head.")
    return df

def drop_tail_missing_values_api(df:pd.DataFrame)-> pd.DataFrame:
    '''
    This function gets a Pandas Dataframe as input. It drops all the missing
    values (i.e nan values) at the tail of the dataframe, if any
    '''
    for i in range(len(df)):
        if math.isnan(df.tail(1).values[0][0]) == True:
            df.drop(index = df.tail(1).index, inplace=True)
        else:
            break
    print("✅ Missing values dealt at the tail.")
    return df

def interpolate_api(df:pd.DataFrame, method:str):
    '''
    This function gets a Pandas Dataframe and a string as inputs.
    It interpolates the data points based on the specified method.
    '''
    print("✅ Intermediate missing values have been interpolated.")
    return df.interpolate(method)

def preprocess_data_api(df:pd.DataFrame)-> pd.DataFrame:
    '''
    This function gets a Pandas Dataframe as input.
    Computes :
    - manage_time_index
    - drop_unsued_cols
    - drop_time_series_duplicates
    - drop_head_missing_values
    - drop_tail_missing_values
    - interpolate
    Returns a preprocessed pandas dataframe.
    '''

    # Combine year and month into a new column as datetime
    df = manage_time_index_api(df)

    # Drop unused columns
    df = drop_unsued_cols_api(df)

    # Get rid of duplicated rows if any
    df = drop_time_series_duplicates_api(df)

    # *** Get rid of missing values ***
    # Get rid of missing values at head
    df = drop_head_missing_values_api(df)

    # Get rid of missing values at tail
    df = drop_tail_missing_values_api(df)

    # Interpolate intermediate missing values
    df = interpolate_api(df, 'linear')

    print("✅ Data has been preprocessed.")
    return df
