import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ta
import os
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from matplotlib.patches import Rectangle
from scipy import stats
from itertools import product
import random
import copy
import math

# ================================
# Globals
# ================================

TICKER = "EURUSD=X"
PLOT_FOLDER = None
DOWNLOAD_DAYS = 1  # Increased to get more data
SEQ_LEN = 26
FUTURE_STEPS = 1
EPOCHS = 50  # Adjusted for experimentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_PERCENTAGE = 0.2  # Increased validation set size
MODE = 0  # Set to 1 to download data from Yahoo Finance
MAX_FILE_ROWS = 3000  # Limit the number of rows to load from the CSV file
LOG_FILE = "epoch_summary"
LAMBDA1 = 0.7  # Prediction loss weight
LAMBDA2 = 0.3  # Volatility regularization weight
EARLY_STOPPING_PATIENCE = 5

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ================================
# Helper Functions
# ================================

def augment_data(X, y, noise_level=0.0001):
    noise = np.random.normal(0, noise_level, X.shape)
    X_augmented = X + noise
    return X_augmented, y

def debug_normalization(scaler, data, column_name):
    """
    Debug and verify normalization and inverse transformation.
    """
    try:
        # Select the specific column
        column_data = data[[column_name]]
        
        print("Before normalization:")
        print(column_data.head(FUTURE_STEPS + 20))

        # Perform normalization
        normalized = scaler.transform(column_data)
        print("\nAfter normalization:")
        print(normalized[:FUTURE_STEPS + 20])

        # Perform inverse transformation
        denormalized = scaler.inverse_transform(normalized)
        print("\nAfter inverse transform:")
        print(denormalized[:FUTURE_STEPS + 20])

    except Exception as e:
        print(f"[ERROR] Debugging normalization failed. Error: {e}")

def log_epoch_data(file_path, epoch, fold, train_loss, val_loss, train_rmse, val_rmse, val_pip_diff,
                   train_sharpe, val_sharpe, train_profit_factor, val_profit_factor,
                   train_max_drawdown, val_max_drawdown, train_directional_accuracy,
                   val_directional_accuracy, train_mape, val_mape, train_r2, val_r2,
                   train_mae, val_mae, train_alpha, train_beta, train_omega):
    """
    Log epoch data to a CSV file with timestamped filenames and fold headers.
    """
    # Generate timestamped filename
    folder = get_plot_folder()
    file_path = f"{folder}/{file_path}.csv"

    # Define data for the current epoch
    epoch_data = {
        "Epoch": epoch,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Train RMSE": train_rmse,
        "Validation RMSE": val_rmse,
        "Validation Pip Difference": val_pip_diff,
        "Train Sharpe Ratio": train_sharpe,
        "Validation Sharpe Ratio": val_sharpe,
        "Train Profit Factor": train_profit_factor,
        "Validation Profit Factor": val_profit_factor,
        "Train Max Drawdown": train_max_drawdown,
        "Validation Max Drawdown": val_max_drawdown,
        "Train Directional Accuracy (%)": train_directional_accuracy,
        "Validation Directional Accuracy (%)": val_directional_accuracy,
        "Train MAPE": train_mape,
        "Validation MAPE": val_mape,
        "Train R²": train_r2,
        "Validation R²": val_r2,
        "Train MAE": train_mae,
        "Validation MAE": val_mae,
        "Train Alpha": train_alpha,
        "Train Beta": train_beta,
        "Train Omega": train_omega
    }

    # Create a DataFrame for the current epoch
    epoch_df = pd.DataFrame([epoch_data])

    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            # Add a fold header if this is the first epoch in a new fold
            if epoch == 1:
                f.write(f"\n---- Fold {fold} ----\n")
        # Append data to the existing file
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, epoch_df], ignore_index=True)
    else:
        # Add a fold header for the first fold
        with open(file_path, 'w') as f:
            f.write(f"---- Fold {fold} ----\n")
        updated_df = epoch_df

    # Save the updated DataFrame to the file
    updated_df.to_csv(file_path, index=False)
    print(f"Epoch {epoch} data for fold {fold} logged to {file_path}.")

def get_plot_folder():
    """Generate a unique folder name for saving plots."""
    global PLOT_FOLDER
    if PLOT_FOLDER is None:
        PLOT_FOLDER = f"plots_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(PLOT_FOLDER, exist_ok=True)
        print(f"Created plot folder: {PLOT_FOLDER}")
        
    return PLOT_FOLDER

def load_forex_data(file_path):
    """
    Load EURUSD M1 data from a CSV file and prepare it for preprocessing.
    """
    # Load the data
    data = pd.read_csv(file_path)
    limited_data = data.iloc[:MAX_FILE_ROWS].copy()

    # Convert 'Date' to datetime and set as the index
    limited_data['Date'] = pd.to_datetime(limited_data['Date'])
    limited_data.set_index('Date', inplace=True)

    return limited_data

def suppress_discontinuities(data, threshold=1e6, smooth_window=5):
    """
    Suppress straight lines caused by discontinuities in technical indicators.
    """
    cleaned_data = data.copy()

    for column in cleaned_data.columns:
        # Suppress extreme values
        cleaned_data[column] = np.where(
            np.abs(cleaned_data[column]) > threshold, np.nan, cleaned_data[column]
        )

        # Fill forward and backward to handle NaNs
        cleaned_data[column] = cleaned_data[column].fillna(method='ffill').fillna(method='bfill')

        # Smooth the data with a rolling window
        cleaned_data[column] = cleaned_data[column].rolling(window=smooth_window, min_periods=1).mean()

    return cleaned_data

def preprocess_data(data):
    """
    Add technical indicators, time-based features, lagged features, and handle missing values.
    """
    print("Data before preprocessing:")
    print(data.head(30))

    # ================================
    # Add Momentum Indicators
    # ================================
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['Stochastic_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stochastic_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'], lbp=14)

    # ================================
    # Add Trend Indicators
    # ================================
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
    data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
    data['Aroon_Up'] = ta.trend.aroon_up(data['High'], data['Low'], window=25)
    data['Aroon_Down'] = ta.trend.aroon_down(data['High'], data['Low'], window=25)

    # ================================
    # Add Volatility Indicators
    # ================================
    bollinger_band = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Bollinger_Band_Width'] = bollinger_band.bollinger_wband()
    data['Bollinger_Band_PctB'] = bollinger_band.bollinger_pband()
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

    # ================================
    # Add Volume Indicators (if applicable)
    # ================================
    if 'Volume' in data.columns and data['Volume'].nunique() > 1:
        data['On_Balance_Volume'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['Chaikin_Money_Flow'] = ta.volume.chaikin_money_flow(
            data['High'], data['Low'], data['Close'], data['Volume'], window=20
        )
        data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'], window=14)

    # ================================
    # Add Time-Based Features
    # ================================
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month

    # ================================
    # Add Target Column
    # ================================
    data['Target'] = data['Close'].shift(-FUTURE_STEPS)  # Shift 'Close' to create future prediction target

    # ================================
    # Handle Missing Values
    # ================================
    data = data.dropna().copy()  # Drop rows with any NaN values
    data.ffill(inplace=True)  # Fill forward any remaining NaN values
    max_lookback = 40  # Remove the first `max_lookback` rows to ensure data integrity
    data = data.iloc[max_lookback:].copy()

    # ================================
    # Define Initial Features
    # ================================
    features = [
        'Close', 'RSI', 'MACD', 'Bollinger_Band_PctB', 'ATR', 'Hour', 'DayOfWeek', 'Month',
        'ADX', 'Aroon_Up', 'Aroon_Down', 'Bollinger_Band_Width'
    ]

    # ================================
    # Add Lagged Features
    # ================================
    for lag in range(1, 10):  # Add lag features for 'Close' from 1 to 9
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        features.append(f'Close_lag_{lag}')

    # Remove rows with NaN values caused by lagged features
    data = data.dropna(subset=features).copy()

    print("Data shape after dropping NaNs from features and target:", data.shape)

    # Check for NaNs or Infs in the data
    if data[features + ['Target']].isnull().values.any():
        print("Data contains NaN values.")
    if np.isinf(data[features + ['Target']].values).any():
        print("Data contains Inf values.")

    print("Data after preprocessing:")
    print(data.head(30))
    print(data.isna().sum())

    return data, features

def scale_features(data, features, target):
    """
    Scale the features and target using RobustScaler and debug normalization.
    """
    # Create scalers
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    # Ensure only specified features and target columns are used
    features_to_scale = data[features].copy()
    target_to_scale = data[target].copy()

    # Fit and transform features and target
    scaled_features = feature_scaler.fit_transform(features_to_scale)
    scaled_target = target_scaler.fit_transform(target_to_scale)

    # Create DataFrame for scaled data
    scaled_features_df = pd.DataFrame(scaled_features, columns=features, index=data.index)
    scaled_features_df['Target'] = scaled_target

    # Debug normalization for the target column
    debug_normalization(
        scaler=target_scaler,
        data=data,
        column_name="Target"  # Use 'Target' for both original and normalized data
    )

    # Log scaled features for debugging
    print("Scaled features:")
    print(scaled_features_df.head(5))

    # Check for NaNs or Infs in scaled data
    if np.isnan(scaled_features_df.values).any():
        print("Scaled data contains NaN values.")
    if np.isinf(scaled_features_df.values).any():
        print("Scaled data contains Inf values.")

    return scaled_features_df, feature_scaler, target_scaler

def create_sequences(data, seq_len, future_steps):
    X, y, indices = [], [], []
    data_values = data.values  # Convert DataFrame to NumPy array
    data_index = data.index  # Get the index (time) from the DataFrame
    for i in range(seq_len, len(data_values) - future_steps + 1):
        X.append(data_values[i - seq_len:i, :-1])  # Exclude 'Target' column
        y.append(data_values[i + future_steps - 1, -1])  # 'Target' is the last column
        indices.append(data_index[i + future_steps - 1])  # Get the time index for y
    return np.array(X), np.array(y), indices

def save_plot(folder, y_actual, y_predicted, epoch, model_name, scaler):
    """
    Save plots of actual vs predicted values with enhancements:
    - Highlight divergences
    - Add trendlines
    - Annotate correlation coefficient
    - Include shaded areas for significant differences.
    """
    # Calculate trendlines
    y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()
    y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
    trendline_actual = np.poly1d(np.polyfit(range(len(y_actual)), y_actual, 1))(range(len(y_actual)))
    trendline_predicted = np.poly1d(np.polyfit(range(len(y_predicted)), y_predicted, 1))(range(len(y_predicted)))
    
    # Correlation coefficient
    correlation, _ = pearsonr(y_actual, y_predicted)
    
    # Define significant divergence threshold
    divergence_threshold = 50  # Define pip threshold (customize as needed)
    divergences = np.abs((y_predicted - y_actual) * 10000) > divergence_threshold

    # Plot Actual vs Predicted with Trendlines
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual, label="Actual", color='blue')
    plt.plot(y_predicted, label="Predicted", color='red', linestyle='--')
    plt.plot(trendline_actual, label="Actual Trendline", color='cyan', linestyle='-.')
    plt.plot(trendline_predicted, label="Predicted Trendline", color='orange', linestyle='-.')
    plt.legend()
    plt.title(f"Predicted vs. Actual Prices with Trendlines - Epoch {epoch}")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    
    # Annotate correlation
    plt.text(0.05, 0.9, f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    # Save Actual vs Predicted plot
    plot_filename_actual = f"{folder}/plot_actual_vs_predicted_epoch_{epoch}_{model_name}.png"
    plt.savefig(plot_filename_actual)
    plt.close()

    # Plot Pip Difference with Highlighted Divergences
    pip_difference = (y_predicted - y_actual) * 10000
    plt.figure(figsize=(14, 6))
    plt.plot(pip_difference, label="Pip Difference", color='green')
    plt.plot(np.poly1d(np.polyfit(range(len(pip_difference)), pip_difference, 1))(range(len(pip_difference))),
             label="Pip Difference Trendline", color='magenta', linestyle='-.')

    # Highlight divergence regions
    for idx, diverged in enumerate(divergences):
        if diverged:
            plt.gca().add_patch(Rectangle((idx - 0.5, -divergence_threshold), 1, 2 * divergence_threshold,
                                          color='red', alpha=0.2))

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.title(f"Pip Difference with Trendline - Epoch {epoch}")
    plt.xlabel("Time Step")
    plt.ylabel("Pip Difference")

    # Save Pip Difference plot
    plot_filename_pip = f"{folder}/plot_pip_difference_epoch_{epoch}_{model_name}.png"
    plt.savefig(plot_filename_pip)
    plt.close()

    print(f"Plots saved: {plot_filename_actual}, {plot_filename_pip}")

def train_model(model, optimizer, criterion, X_train, y_train, batch_size=64):
    model.train()
    train_loss = 0.0
    y_train_pred, y_train_actual = [], []

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for X_batch, y_batch in tqdm(train_loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        prediction = model(X_batch)
        if torch.isnan(prediction).any() or torch.isinf(prediction).any():
            print("Model outputs contain NaN or Inf values.")
            continue  # Skip this batch
        loss = criterion(prediction, y_batch)
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss is NaN or Inf.")
            continue  # Skip this batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)  # Accumulate total loss

        y_train_pred.extend(prediction.detach().cpu().numpy().squeeze())
        y_train_actual.extend(y_batch.detach().cpu().numpy().squeeze())

    train_loss /= len(train_loader.dataset)
    return train_loss, np.array(y_train_pred), np.array(y_train_actual)

def validate_model(model, criterion, X_val, y_val, batch_size=64):
    model.eval()
    val_loss = 0.0
    y_val_pred, y_val_actual = [], []

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            prediction = model(X_batch)
            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                print("Model outputs contain NaN or Inf values.")
                continue  # Skip this batch
            loss = criterion(prediction, y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Validation loss is NaN or Inf.")
                continue  # Skip this batch
            val_loss += loss.item() * X_batch.size(0)

            y_val_pred.extend(prediction.cpu().numpy().squeeze())
            y_val_actual.extend(y_batch.cpu().numpy().squeeze())

    val_loss /= len(val_loader.dataset)
    return val_loss, np.array(y_val_pred), np.array(y_val_actual)

def calculate_metrics(y_actual, y_pred, scaler, prefix=""):
    """
    Calculate a comprehensive set of metrics for model evaluation.
    """
    # Denormalize if scaler is provided
    if scaler:
        y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
    returns = y_pred - y_actual
    # Core error metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100  # Avoid division by zero
    r2 = r2_score(y_actual, y_pred)

    # Pip Difference
    pip_diff, mean_abs_pip_diff = calculate_pip_difference(y_pred, y_actual)

    # Directional Accuracy (percentage of correct directional predictions)
    if len(y_actual) > 1:  # Ensure at least two points for directionality
        correct_directions = (np.sign(y_actual[1:] - y_actual[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        directional_accuracy = np.mean(correct_directions) * 100
    else:
        directional_accuracy = None

    # Sharpe Ratio Proxy
    if returns is not None and len(returns) > 1:
        avg_return = np.mean(returns)
        return_std = np.std(returns) + 1e-8  # Avoid division by zero
        sharpe_ratio = avg_return / return_std if return_std != 0 else 0
    else:
        sharpe_ratio = None

    # Max Drawdown
    if returns is not None and len(returns) > 1:
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-8)  # Avoid division by zero
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = None

    # Profit Factor Proxy
    if returns is not None and len(returns) > 1:
        profits = np.sum(returns[returns > 0])
        losses = -np.sum(returns[returns < 0])
        profit_factor = profits / losses if losses != 0 else 0.0  # Set default to 0.0
    else:
        profit_factor = 0.0  # Set default to 0.0

    # Combine all metrics into a dictionary
    metrics = {
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}MAPE": mape,
        f"{prefix}R²": r2,
        f"{prefix}Mean Absolute Pip Difference": mean_abs_pip_diff,
        f"{prefix}Directional Accuracy (%)": directional_accuracy,
        f"{prefix}Sharpe Ratio": sharpe_ratio,
        f"{prefix}Max Drawdown": max_drawdown,
        f"{prefix}Profit Factor": profit_factor,
    }
    return metrics

def calculate_pip_difference(y_pred, y_actual, scaler=None):
    """Calculate pip difference and mean absolute pip difference."""
    # Denormalize predictions and actuals if scaler is provided
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()

    # Calculate pip difference
    pip_diff = (y_pred - y_actual) * 10000  # Convert to pips
    abs_pip_diff = np.abs(pip_diff).mean()
    return pip_diff, abs_pip_diff

def save_epoch_metrics(metrics_df, output_filename):
    """Save epoch metrics to an Excel file."""
    with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
        metrics_df.to_excel(writer, sheet_name="Epoch Metrics", index=False)

        # Formatting in Excel
        workbook = writer.book
        worksheet = writer.sheets["Epoch Metrics"]

        # Header formatting
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#D7E4BC",
            "border": 1
        })

        # Apply formatting to headers
        for col_num, value in enumerate(metrics_df.columns):
            worksheet.write(0, col_num, value, header_format)

        # Adjust column widths
        for col_num, col_name in enumerate(metrics_df.columns):
            worksheet.set_column(col_num, col_num, len(col_name) + 5)

    print(f"Training summary saved to: {output_filename}")

def analyze_results(metrics_df):
    """Analyze results with rolling averages, delta changes, and highlight best/worst epochs."""
    metrics_df['Rolling RMSE'] = metrics_df['Validation RMSE'].rolling(window=3).mean()
    metrics_df['Rolling Pip Difference'] = metrics_df['Validation Pip Difference'].rolling(window=3).mean()

    metrics_df['RMSE Change'] = metrics_df['Validation RMSE'].diff()
    metrics_df['Pip Difference Change'] = metrics_df['Validation Pip Difference'].diff()

    best_epoch = metrics_df.loc[metrics_df['Validation RMSE'].idxmin()]
    worst_epoch = metrics_df.loc[metrics_df['Validation RMSE'].idxmax()]

    print("\nBest Epoch:")
    print(best_epoch)

    print("\nWorst Epoch:")
    print(worst_epoch)

    print("\nRolling Averages and Delta Changes:")
    print(metrics_df[['Epoch', 'Rolling RMSE', 'RMSE Change', 'Rolling Pip Difference', 'Pip Difference Change']])

    # Plot distribution of Pip Differences
    sns.histplot(metrics_df['Validation Pip Difference'], bins=30, kde=True)
    plt.title("Distribution of Pip Differences")
    plt.xlabel("Pip Difference")
    plt.ylabel("Frequency")
    plt.show()
    
def save_predictions_and_metrics(y_actual, y_pred, time_index, metrics, epoch, model_name, scaler):
    """
    Save predictions, actuals, time, and metrics for each epoch to an Excel file.
    """
    # Inverse transform y_actual and y_pred
    y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Convert time_index to pandas Series if necessary
    if not isinstance(time_index, pd.Series):
        time_index = pd.Series(time_index)
    time_index = pd.to_datetime(time_index)

    # Remove timezone information
    time_index = time_index.dt.tz_localize(None)

    # Create a DataFrame to store predictions and actual values
    predictions_df = pd.DataFrame({
        "Time": time_index,
        "Actual Price": y_actual,
        "Predicted Price": y_pred,
        "Pip Difference": (y_pred - y_actual) * 10000,
    })

    # Add metrics as a summary at the bottom of the DataFrame
    metrics_summary = pd.DataFrame.from_dict([metrics])
    metrics_summary.index = ["Metrics"]
    predictions_df = pd.concat([predictions_df, metrics_summary])
    folder = get_plot_folder()
    # Generate a filename
    filename = f"{folder}/predictions_metrics_epoch_{epoch}_{model_name}.xlsx"

    # Save to Excel
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        predictions_df.to_excel(writer, sheet_name=f"Epoch {epoch}", index=False)

        # Format the sheet
        workbook = writer.book
        worksheet = writer.sheets[f"Epoch {epoch}"]
        header_format = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "fg_color": "#D7E4BC",
            "border": 1
        })

        # Apply header formatting
        for col_num, value in enumerate(predictions_df.columns):
            worksheet.write(0, col_num, value, header_format)

        # Adjust column width
        for col_num, value in enumerate(predictions_df.columns):
            worksheet.set_column(col_num, col_num, len(str(value)) + 5)

    print(f"Saved predictions and metrics for epoch {epoch} to: {filename}")

# ================================
# Model Definitions
# ================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LSTMTransformer(nn.Module):
    def __init__(
        self, input_dim, lstm_hidden_size=128, lstm_layers=1, 
        nhead=4, num_transformer_layers=2, dim_feedforward=256,
        dropout=0.1, output_dim=1
    ):
        super(LSTMTransformer, self).__init__()
        dropout = dropout if lstm_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_dim, lstm_hidden_size, num_layers=lstm_layers, 
            batch_first=True, dropout=dropout, bidirectional=False
        )
        
        self.positional_encoding = PositionalEncoding(lstm_hidden_size, dropout)
        
        # Enable batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.fc = nn.Linear(lstm_hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden_size]
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.positional_encoding(lstm_out)  # No need to permute now
        transformer_out = self.transformer_encoder(lstm_out)  # [batch_size, seq_len, lstm_hidden_size]
        # Use the last time step's output
        output = transformer_out[:, -1, :]
        output = self.fc(output)
        return output

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)
        # Optionally add a volatility penalty
        if predictions.size(0) > 1:
            volatility_penalty = torch.mean(torch.abs(predictions[1:] - predictions[:-1]))
        else:
            volatility_penalty = 0.0  # No penalty if only one sample
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss + 0.01 * volatility_penalty

# ================================
# Main Workflow
# ================================

# Download data
if MODE == 0:
    data = load_forex_data("EURUSD_M1.csv")
    print("Data loaded successfully", len(data))
else:
    # Adjust the date range to get more data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DOWNLOAD_DAYS)  # Get data from the past few days

    # Ensure dates are weekdays
    while end_date.weekday() > 4:  # Saturday or Sunday
        end_date -= timedelta(days=1)

    while start_date.weekday() > 4:
        start_date -= timedelta(days=1)

    data = yf.download(
        TICKER,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval="1m",
        progress=False
    )

    if data.empty or data['Volume'].sum() == 0:
        print("Downloaded data is empty or has zero volume. Trying a higher interval.")
        # Try a higher interval
        data = yf.download(
            TICKER,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="5m",
            progress=False
        )

    if data.empty or data['Volume'].sum() == 0:
        print("No suitable data found. Consider changing the data source or date range.")
        exit()
    else:
        print("Data downloaded successfully:", data.shape)
        print(data.head())

# Proceed only if data is available
if not data.empty:
    # Preprocess data
    data, features = preprocess_data(data)
    print("Data shape after preprocessing:", data.shape)

    # Define features and target
    target = ['Target']

    # Drop rows with NaNs in features or target
    data.dropna(subset=features + target, inplace=True)
    print("Data shape after dropping NaNs from features and target:", data.shape)

    if data.empty:
        print("No data left after preprocessing. Please adjust your preprocessing steps.")
        exit()

    # Scale features and target
    scaled_data, feature_scaler, target_scaler = scale_features(data, features, target)

    # Create sequences
    X, y, indices = create_sequences(scaled_data, SEQ_LEN, FUTURE_STEPS)

    # Check for sequences
    if len(X) == 0:
        print("No sequences created. Check SEQ_LEN and ensure it's appropriate for the dataset size.")
        exit()

    print(f"Number of sequences created: {len(X)}")

# Split data into training and testing sets
kf = KFold(n_splits=5)
fold = 1

# Initialize hyperparameter grid
hyperparams_grid = {
    'lstm_hidden_size': [128, 256],
    'lstm_layers': [2, 3],
    'nhead': [4, 8],
    'num_transformer_layers': [2, 4],
    'dim_feedforward': [256, 512],
    'dropout': [0.1, 0.2],
    'learning_rate': [0.0005, 0.0001],  # Reduced learning rates
    'weight_decay': [0.001, 0.0001]  # Reduced weight decay
}

best_overall_val_loss = float('inf')
best_hyperparams = None
best_model_state = None

input_dim = X.shape[2]

for params in product(*hyperparams_grid.values()):
    params_dict = dict(zip(hyperparams_grid.keys(), params))
    print(f"\nTraining with hyperparameters: {params_dict}")

    # Cross-validation
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold}")

        # Re-initialize model for each fold
        model = LSTMTransformer(
            input_dim=input_dim,
            lstm_hidden_size=params_dict['lstm_hidden_size'],
            lstm_layers=params_dict['lstm_layers'],
            nhead=params_dict['nhead'],
            num_transformer_layers=params_dict['num_transformer_layers'],
            dim_feedforward=params_dict['dim_feedforward'],
            dropout=params_dict['dropout']
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=params_dict['learning_rate'],
            weight_decay=params_dict['weight_decay']
        )

        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = CustomLoss(alpha=0.7)

        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        time_indices_train = np.array(indices)[train_index]
        time_indices_val = np.array(indices)[val_index]

        # Remove data augmentation to prevent NaN issues
        # X_train_augmented, y_train_augmented = augment_data(X_train_fold, y_train_fold)
        X_train_augmented, y_train_augmented = X_train_fold, y_train_fold

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_augmented, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_augmented, dtype=torch.float32).unsqueeze(-1)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(-1)

        # Move tensors to device
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_val_tensor = X_val_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)

        # Reset early stopping variables
        early_stopping_counter = 0
        best_val_loss = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())  # Initialize here

        # Track best validation RMSE
        epoch_metrics = []

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")

            train_loss, y_train_pred, y_train_actual = train_model(
                model, optimizer, criterion, X_train_tensor, y_train_tensor
            )

            val_loss, y_val_pred, y_val_actual = validate_model(
                model, criterion, X_val_tensor, y_val_tensor
            )

            # Calculate metrics
            train_metrics = calculate_metrics(
                y_train_actual, y_train_pred, scaler=target_scaler, prefix="Train "
            )
            val_metrics = calculate_metrics(
                y_val_actual, y_val_pred, scaler=target_scaler, prefix="Validation "
            )

            # Log metrics
            log_epoch_data(
                file_path=LOG_FILE,
                epoch=epoch + 1,
                fold=fold,
                train_loss=train_loss,
                val_loss=val_loss,
                train_rmse=train_metrics["Train RMSE"],
                val_rmse=val_metrics["Validation RMSE"],
                val_pip_diff=val_metrics["Validation Mean Absolute Pip Difference"],
                train_sharpe=train_metrics["Train Sharpe Ratio"],
                val_sharpe=val_metrics["Validation Sharpe Ratio"],
                train_profit_factor=train_metrics["Train Profit Factor"],
                val_profit_factor=val_metrics["Validation Profit Factor"],
                train_max_drawdown=train_metrics["Train Max Drawdown"],
                val_max_drawdown=val_metrics["Validation Max Drawdown"],
                train_directional_accuracy=train_metrics["Train Directional Accuracy (%)"],
                val_directional_accuracy=val_metrics["Validation Directional Accuracy (%)"],
                train_mape=train_metrics["Train MAPE"],
                val_mape=val_metrics["Validation MAPE"],
                train_r2=train_metrics["Train R²"],
                val_r2=val_metrics["Validation R²"],
                train_mae=train_metrics["Train MAE"],
                val_mae=val_metrics["Validation MAE"],
                train_alpha=None,
                train_beta=None,
                train_omega=None
            )

            # Scheduler step
            if not np.isnan(val_loss) and not np.isinf(val_loss):
                scheduler.step(val_loss)

            # Early stopping logic
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"Validation loss is not a finite number at epoch {epoch + 1}. Skipping model state update.")
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1} due to NaN loss.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    else:
                        print("Warning: best_model_state is None. Cannot load best model state.")
                    break
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    else:
                        print("Warning: best_model_state is None. Cannot load best model state.")
                    break

            val_abs_pip_diff = calculate_pip_difference(y_val_pred, y_val_actual, target_scaler)[1]
            # Print metrics
            print(f"\nLSTM-TRANSFORMER - Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}")
            print(f"  Train RMSE: {train_metrics['Train RMSE']:.5f}, Validation RMSE: {val_metrics['Validation RMSE']:.5f}")
            print(f"  Train MAE: {train_metrics['Train MAE']:.5f}, Validation MAE: {val_metrics['Validation MAE']:.5f}")
            print(f"  Train MAPE: {train_metrics['Train MAPE']:.5f}%, Validation MAPE: {val_metrics['Validation MAPE']:.5f}%")
            print(f"  Train R²: {train_metrics['Train R²']:.5f}, Validation R²: {val_metrics['Validation R²']:.5f}")
            print(f"  Train Directional Accuracy: {train_metrics['Train Directional Accuracy (%)']:.2f}%, Validation Directional Accuracy: {val_metrics['Validation Directional Accuracy (%)']:.2f}%")
            print(f"  Train Sharpe Ratio: {train_metrics['Train Sharpe Ratio']:.5f}, Validation Sharpe Ratio: {val_metrics['Validation Sharpe Ratio']:.5f}")
            print(f"  Train Max Drawdown: {train_metrics['Train Max Drawdown']:.5f}, Validation Max Drawdown: {val_metrics['Validation Max Drawdown']:.5f}")
            print(f"  Train Profit Factor: {train_metrics['Train Profit Factor']:.5f}, Validation Profit Factor: {val_metrics['Validation Profit Factor']:.5f}")
            print(f"  Validation Mean Absolute Pip Difference: {val_abs_pip_diff:.2f} pips")

            # Denormalize validation predictions and actuals for inspection
            y_val_pred_denorm = target_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
            y_val_actual_denorm = target_scaler.inverse_transform(y_val_actual.reshape(-1, 1)).flatten()

            print("Sample denormalized predictions:", y_val_pred_denorm[:5])
            print("Sample denormalized actuals:", y_val_actual_denorm[:5])

            # Save predictions and metrics
            save_predictions_and_metrics(
                y_actual=y_val_actual,
                y_pred=y_val_pred,
                time_index=time_indices_val,
                metrics=val_metrics,
                epoch=epoch + 1,
                model_name="LSTM-TRANSFORMER",
                scaler=target_scaler
            )

            # Store metrics for reporting
            epoch_metrics.append({
                "Epoch": epoch + 1,
                "Model": "LSTM-TRANSFORMER",
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                **train_metrics,
                **val_metrics,
                "Validation Pip Difference": val_abs_pip_diff
            })

            save_plot(get_plot_folder(), y_val_actual, y_val_pred, epoch + 1, "LSTM-TRANSFORMER", target_scaler)

        # Residual Analysis
        residuals = y_val_actual - y_val_pred
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Distribution')
        plt.xlabel('Residual')
        plt.show()

        # After each fold
        fold += 1

print(f"\nBest hyperparameters: {best_hyperparams}")
# Load the best model state

if best_model_state is None:
    print("Warning: best_model_state is None. Using the current model state instead.")
    best_model_state = copy.deepcopy(model.state_dict())

# Save metrics
metrics_df = pd.DataFrame(epoch_metrics)
output_filename = f"training_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
save_epoch_metrics(metrics_df, output_filename)
analyze_results(metrics_df)
