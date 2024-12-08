# src/feature_engineering.py

import pandas as pd
import numpy as np

def preprocess_features(df):
    """
    Adds seasonal and cyclic features to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a datetime index.

    Returns:
    - df (pd.DataFrame): DataFrame with added features.
    """
    # Add hour of day as categorical features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    # Add sine and cosine transformations for cyclic features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    
    # Add trading session as categorical variable
    # Define trading sessions based on hour
    def get_trading_session(hour):
        if 0 <= hour < 8:
            return 0
        elif 8 <= hour < 16:
            return 1
        elif 16 <= hour < 24:
            return 2
    
    df['TradingSession'] = df['Hour'].apply(get_trading_session)
    
    
    # Drop the original Hour, DayOfWeek, Month columns if not needed
    df.drop(['Hour', 'DayOfWeek', 'Month'], axis=1, inplace=True)
    print("Seasonality Done!")
    return df
