
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import logging
import src.prepare_data_stats as pds
import matplotlib.pyplot as plt
import os
from src.display import colored_print, print_status, plot_indicator_categories, execute_with_status


def prepare_data(df, horizon=1):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    # Ensure only the 'Close' column is shifted
    shifted = df['Close'].shift(-horizon)  # Extract only the 'Close' column and shift it

    # Remove the last 'horizon' rows from both df and shifted
    df = df.iloc[:-horizon].reset_index(drop=True)
    shifted = shifted.iloc[:-horizon].reset_index(drop=True)

    # Assign the shifted column as the 'Target'
    df['Target'] = shifted
    df.dropna(inplace=True)
    return df

def remove_multicollinearity(df, threshold=0.9, essential_features=None):
    if essential_features is None:
        essential_features = []
    colored_print("Removing Colinearity:")
    colored_print("  Essential features before expansion:" + ' '.join(str(x) for x in essential_features), "yellow")

    # If "Lag" is specified as essential, then mark all columns containing "Lag" as essential
    # Similarly, you can add logic for "Close" or any other substring if needed.
    expanded_essential = set(essential_features)
    for col in df.columns:
        if 'VolumeLag' in essential_features:
            # If "Lag" is in the essential features, include any column that has 'Lag' in its name
            if 'VolumeLag' in col:
                expanded_essential.add(col)
        # If "Close" is considered essential, include all columns containing 'Close'
        if 'CloseLag' in essential_features:
            if 'CloseLag' in col:
                expanded_essential.add(col)

    essential_features = list(expanded_essential)
    colored_print("  Essential features after expansion:" + ' '.join(str(x) for x in essential_features), "magenta")

    corr_matrix = df.drop('Target', axis=1).corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs of features with correlation greater than threshold
    to_drop = []
    for column in upper_triangle.columns:
        high_corr = upper_triangle[column][upper_triangle[column] > threshold].index.tolist()
        if high_corr:
            for correlated_feature in high_corr:
                # Decide which feature to drop
                if column not in essential_features and correlated_feature not in essential_features:
                    # By default, we drop 'column' or 'correlated_feature', 
                    # but you can refine the logic if needed.
                    to_drop.append(column)
                    break
                elif column not in essential_features and correlated_feature in essential_features:
                    to_drop.append(column)
                    break
                elif correlated_feature not in essential_features and column in essential_features:
                    to_drop.append(correlated_feature)
                    # break if you want to drop just one from the pair
                    break

    to_drop = sorted(set(to_drop))
    colored_print(f"Features to drop due to high correlation (>{threshold}):", "re")
    colored_print(to_drop, "red")

    df.drop(columns=to_drop, inplace=True, errors='ignore')
    
    return df

def treat_outliers_iqr(df, multiplier=3.0):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    changes = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        original_values = df[col].copy()
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        
        changed_count = (original_values != df[col]).sum()
        if changed_count > 0:
            changes[col] = changed_count
    
    print("Outlier Correction Summary using IQR method:")
    if changes:
        for col, count in changes.items():
            print(f"{col}: {count} values were corrected.")
    else:
        print("No outliers were corrected.")
    
    return df

def split_data(df, test_size=0.05):
    """
    Splits the data into training and test sets chronologically.
    """
    split_index = int(len(df) * (1 - test_size))
    X = df.drop('Target', axis=1)
    y = df['Target']

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Debugging: Check size using f-strings
    colored_print(f"X_train_len_split: {len(X_train)} {X_train.shape}")
    colored_print(f"y_train_len_split: {len(y_train)} {y_train.shape}")
    colored_print(f"X_test_len_split: {len(X_test)} {X_test.shape}")
    colored_print(f"y_test_len_split: {len(y_test)} {y_test.shape}")

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, y_train, y_test , scaler_type=None):
    """
    Scales features and target variable using StandardScaler.
    """
    if scaler_type != None:
        if scaler_type == "Standard":
            colored_print("Standardize data", "green")
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()

        if scaler_type == 'MinMax':
            colored_print("MinMax data", "green")
            feature_scaler = MinMaxScaler()
            target_scaler = StandardScaler()
            
        if scaler_type == "Normalize":
            colored_print("Normalize data", "green")
            feature_scaler = Normalizer()
            target_scaler = Normalizer()
            
        X_train_scaled = pd.DataFrame(feature_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(feature_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        y_train_scaled = pd.Series(target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
        y_test_scaled = pd.Series(target_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled , target_scaler
    else:
        colored_print("No scaler SELECTED", "red")
        return X_train, X_test, y_train, y_test , None


def create_sequences(features, targets, sequence_length, horizon):
    X, y = [], []
    print("range", len(features) - sequence_length - horizon)
    for i in range(len(features) - sequence_length - horizon):
        X_seq = features.iloc[i:(i + sequence_length)]  # Select rows by position
        y_target = targets.iloc[i + sequence_length]     # Target corresponding to the end of the sequence
        
        if i == 0:
            print("features:", features.iloc[i + sequence_length]["Close"], 
                  "target:", targets.iloc[i + sequence_length - horizon])
        
        X.append(X_seq.values)  # Convert the selected DataFrame slice to a NumPy array
        y.append(y_target)
        
    return np.array(X), np.array(y)


def keep_only_essential_features(df, essential_features):
    """
    Drops all features except the essential features.
    """
    expanded_essential = set()
    df_cols = set(df.columns)

    # Identify all columns that contain 'CloseLag' or 'VolumeLag'
    close_lag_cols = [c for c in df.columns if 'CloseLag' in c]
    volume_lag_cols = [c for c in df.columns if 'VolumeLag' in c]

    for feat in essential_features:
        # If the feature is a pattern placeholder
        if feat == 'CloseLag':
            expanded_essential.update(close_lag_cols)
        elif feat == 'VolumeLag':
            expanded_essential.update(volume_lag_cols)
        else:
            # Directly essential if it exists
            if feat in df_cols:
                expanded_essential.add(feat)

    # Always keep 'Target' if it exists
    if 'Target' in df_cols:
        expanded_essential.add('Target')

    # Convert back to list and select only existing columns
    final_features = [f for f in expanded_essential if f in df_cols]

    # Print the final list of columns before selecting
    colored_print("Remaining columns before splitting: " + str(sorted(final_features)), "green")

    # Subset the DataFrame
    df = df[final_features]
    return df


def prepare_data_full(df, SEQUENCE_LENGTH=60, horizon=1, scale=None):
    
    essential_features = ['Close', "High", "Low", "Open", "Volume", "CloseLag", "VolumeLag", "MACD", "RSI", "ROC", "VO", "ATR"]
    
    df = execute_with_status("Prepare the data:", prepare_data, df, horizon=horizon)
    df = execute_with_status("Handle multicollinearity:", remove_multicollinearity, df, threshold=0.8, essential_features=essential_features)
    df = execute_with_status("Keep only essential features:",keep_only_essential_features,df,essential_features)
    plot_indicator_categories(df)
    
    # Data Splitting
    X_train, X_test, y_train, y_test = execute_with_status("SPLIT the data:", split_data, df, test_size=0.04)
    
    # Scale data after splitting
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, target_scaler = execute_with_status("SCALE the data:", scale_data, X_train, X_test,y_train,y_test, scale)

    #pds.analyse_prepared_data(X_train_scaled.assign(Target=y_train_scaled), essential_features)

    # Create sequences using scaled targets
    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled, y_train_scaled, SEQUENCE_LENGTH, horizon
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled, y_test_scaled, SEQUENCE_LENGTH, horizon
    )
    
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, target_scaler