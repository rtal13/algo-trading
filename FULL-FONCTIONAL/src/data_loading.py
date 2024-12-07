# src/data_loading.py

import pandas as pd
from src.display import display_load_data

def load_data(file_path, start_index=None, end_index=None):

    # Load the data, parsing the 'Date' column as datetime and setting it as the index
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Sort the DataFrame by index (date)
    df = df.sort_index()
    
    # Remove duplicate timestamps
    if df.index.duplicated().any():
        print("Duplicate timestamps found. Removing duplicates.")
        df = df[~df.index.duplicated(keep='first')]
    
    # Slice rows by numerical indices if start_index or end_index is provided
    if start_index is not None or end_index is not None:
        df = df.iloc[start_index:end_index]
    
    return df
