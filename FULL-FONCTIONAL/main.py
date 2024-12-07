import os
import time
import numpy as np
import pandas as pd
import torch
from src.data_loading import load_data
from src.feature_engineering import preprocess_features
from src.indicators import add_technical_indicators
from src.prepare_data import prepare_data_full
from src.display import print_status, colored_print
from src.model import init_model
from src.training import train_model
from src.evaluate import evaluate_model

np.set_printoptions(precision=16)  # For NumPy
pd.set_option('display.precision', 16)  # For pandas

def main():
    # Configure paths
    data_file = os.path.join('data', 'EURUSD_M1.csv')
    now = time.time()
    folder_name = 'Epoch_{}'.format(now)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    SEQUENCE_LENGTH = 60  # Adjust as needed
    START_INDEX = 1
    END_INDEX = 20000
    HORIZON = 1  # Forecast horizon
    BATCH_SIZE = 8
    EPOCHS = 250
    LR = 0.001  # Learning rate
    
    indicator_types = {
        "trend": True,
        "momentum": True,
        "volume": True,
        "volatility": True,
        "additional": True,
        "vol_oscillator": True,
        "pivot_point": True,
        "lag": True,
        "rolling": True,
        "polynomial": True,
    }
    
    # Load the data
    colored_print("Loading data:")
    df = load_data(file_path=data_file, start_index=START_INDEX, end_index=END_INDEX)
    print_status(True)
    colored_print("Process seasonality:")
    df = preprocess_features(df)
    print_status(True)
    colored_print('Adding Technical Indicators:')
    df = add_technical_indicators(df, HORIZON, indicator_types=indicator_types)
    print_status(True)
    
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, target_scaler = prepare_data_full(df, SEQUENCE_LENGTH, HORIZON, scale="Standard")
    print("X_train_seq shape:", X_train_seq.shape)
    print("y_train_seq shape:", y_train_seq.shape)
    print("X_test_seq shape:", X_test_seq.shape)
    print("y_test_seq shape:", y_test_seq.shape)

    # Reshape targets to (N, 1)
    y_train_seq = y_train_seq.reshape(-1, 1)
    y_test_seq = y_test_seq.reshape(-1, 1)

    num_features = X_train_seq.shape[2]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_seq).float()
    y_train_tensor = torch.from_numpy(y_train_seq).float()
    X_test_tensor = torch.from_numpy(X_test_seq).float()
    y_test_tensor = torch.from_numpy(y_test_seq).float()
    
    model, criterion, optimizer = init_model(
        input_dim=num_features,
        lstm_hidden_dim=64,
        lstm_layers=1,
        transformer_dim=64,
        nhead=4,
        num_transformer_layers=2,
        fc_dim=1,
        dropout=0.1,
        lr=LR
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    model.to(device)
    model = train_model(
        model, criterion, optimizer, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        patience=10, gradient_clip_value=1.0, folder_name=folder_name, 
        scheduler_patience=5, target_scaler=target_scaler
    )
    
    # Evaluate only if we have test samples
    if X_test_tensor.size(0) > 0:
        evaluate_model(model, criterion, X_test_tensor, y_test_tensor, target_scaler, folder_name, epoch_number=EPOCHS)
    else:
        print("No test samples available. Skipping evaluation.")

if __name__ == '__main__':
    main()