import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ta
import os
from datetime import datetime, timedelta
import seaborn as sns
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
DOWNLOAD_DAYS = 3  # Increased to get more data only can download 5 day max for 11min interval
SEQ_LEN = 26
FUTURE_STEPS = 1
EPOCHS = 50  # Reduced for quicker experimentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_PERCENTAGE = 0.2  # Increased validation set size
MODE = 1  # Set to 1 to download data from Yahoo Finance
LOG_FILE = "epoch_summary.csv"
LAMBDA1 = 0.7  # Prediction loss weight
LAMBDA2 = 0.3  # Volatility regularization weight
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.001  # Starting learning rate
WEIGHT_DECAY = 0.01  # Starting weight decay

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

def augment_data(X, y, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    X_augmented = X + noise
    return X_augmented, y

def debug_normalization(scaler, data, column_name):
    """
    Debug and verify normalization and inverse transformation.

    Parameters:
    - scaler (MinMaxScaler): The scaler used for normalization.
    - data (pd.DataFrame): The data containing the column to be checked.
    - column_name (str): Column name to be normalized and inverse transformed.

    Returns:
    None
    """
    try:
        # Ensure input data is a DataFrame
        values = data[[column_name]]
        print("Before normalization:")
        print(values.head(FUTURE_STEPS + 5).values)

        # Perform normalization
        normalized = scaler.transform(values)
        print("\nAfter normalization:")
        print(normalized[:FUTURE_STEPS + 5])

        # Perform inverse transformation
        denormalized = scaler.inverse_transform(normalized)
        print("\nAfter inverse transform:")
        print(denormalized[:FUTURE_STEPS + 5])

    except Exception as e:
        print(f"[ERROR] Debugging normalization failed. Error: {e}")

def log_epoch_data(file_path, epoch, fold, train_loss, val_loss, train_rmse, val_rmse, val_pip_diff,
                   train_sharpe, val_sharpe, train_profit_factor, val_profit_factor,
                   train_max_drawdown, val_max_drawdown, train_directional_accuracy,
                   val_directional_accuracy, train_mape, val_mape, train_r2, val_r2,
                   train_mae, val_mae, train_alpha, train_beta, train_omega):
    """
    Log epoch data to a CSV file with timestamped filenames and fold headers.

    Parameters:
    - file_path (str): Base path for the log file (without timestamp).
    - epoch (int): Epoch number.
    - fold (int): Fold number for cross-validation.
    - train_loss, val_loss, etc.: Metrics to log.
    """
    # Generate timestamped filename
    folder = get_plot_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    Parameters:
        file_path (str): Path to the EURUSD_M1.csv file.

    Returns:
        pd.DataFrame: DataFrame ready for preprocessing.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Convert 'Date' to datetime and set as the index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    return data

def suppress_discontinuities(data, threshold=1e6, smooth_window=5):
    """
    Suppress straight lines caused by discontinuities in technical indicators.

    Parameters:
    - data (pd.DataFrame): DataFrame containing technical indicator columns.
    - threshold (float): Threshold to identify outliers or invalid values.
    - smooth_window (int): Window size for smoothing.

    Returns:
    pd.DataFrame: Cleaned DataFrame with discontinuities suppressed.
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
    Add technical indicators, time-based features, and handle missing values.

    Parameters:
    data (pd.DataFrame): A DataFrame containing OHLCV data with a DateTime index.

    Returns:
    pd.DataFrame: DataFrame with additional features for model training.
    """
    print("Data before preprocessing:")
    print(data.head(30))

    # Add momentum indicators
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['Stochastic_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stochastic_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'], lbp=14)

    # Add trend indicators
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
    data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
    data['Aroon_Up'] = ta.trend.aroon_up(data['High'], data['Low'], window=25)
    data['Aroon_Down'] = ta.trend.aroon_down(data['High'], data['Low'], window=25)

    # Add volatility indicators
    bollinger_band = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Bollinger_Band_Width'] = bollinger_band.bollinger_wband()
    data['Bollinger_Band_PctB'] = bollinger_band.bollinger_pband()
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

    # Add volume indicators
    if 'Volume' in data.columns and data['Volume'].nunique() > 1:
        data['On_Balance_Volume'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['Chaikin_Money_Flow'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'], window=20)
        data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'], window=14)
    
    # Add time-based features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month

    data['Target'] = data['Close'].shift(-FUTURE_STEPS)

    # Add lagged features
    for lag in range(1, 4):
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)

    # Drop rows with NaN values (due to indicators, lagging, or shifting)
    data = data.dropna().copy()  # Drop and copy explicitly to avoid warnings
    # Fill forward missing values caused by indicator calculations
    data.ffill(inplace=True)

    # Drop rows with missing values only after ensuring enough data
    max_lookback = 40  # Maximum look-back period used by any indicator
    data = data.iloc[max_lookback:]  # Remove only the first N rows

    print("Data after preprocessing:")
    print(data.head(30))
    print(data.isna().sum())
    return data


def plot_indicators(data, train_split_index, indicators):
    """
    Plot selected indicators and visualize the train/validation split.

    Parameters:
    data (pd.DataFrame): DataFrame with technical indicators and a DateTime index.
    train_split_index (int): Index to split train and validation data.
    indicators (list): List of indicator columns to plot.
    """
    # Split data into train and validation
    train_data = data.iloc[:train_split_index]
    validation_data = data.iloc[train_split_index:]

    # Generate Buy/Sell Signals
    buy_signals = data[(data['RSI'] < 30) & (data['Close'] < data['EMA_12'])].index
    sell_signals = data[(data['RSI'] > 70) & (data['Close'] > data['EMA_12'])].index

    # Convert signal indices to integer positions
    buy_signals_idx = data.index.get_indexer(buy_signals)
    sell_signals_idx = data.index.get_indexer(sell_signals)

    # Plot Close price
    plt.figure(figsize=(14, 10))

    # Close Price Plot
    plt.subplot(3, 1, 1)
    plt.plot(train_data.index, train_data['Close'], label='Train Close Price')
    plt.plot(validation_data.index, validation_data['Close'], label='Validation Close Price')
    plt.scatter(data.index[buy_signals_idx], data['Close'].iloc[buy_signals_idx], color='green', label='Buy Signal', marker='^')
    plt.scatter(data.index[sell_signals_idx], data['Close'].iloc[sell_signals_idx], color='red', label='Sell Signal', marker='v')
    plt.axvline(x=data.index[train_split_index], color='orange', linestyle='--', label='Train/Validation Split')
    plt.legend()
    plt.title("Close Price with Buy/Sell Signals")

    # Indicators Plot
    plt.subplot(3, 1, 2)
    for ind in indicators:
        plt.plot(data.index, data[ind], label=ind)
    plt.legend()
    plt.title("Technical Indicators")

    # Zoomed Plot
    plt.subplot(3, 1, 3)
    random_start = np.random.randint(0, len(data) - 100)
    zoom_start, zoom_end = random_start, random_start + 100
    
    plt.plot(data.index[zoom_start:zoom_end], data['Close'].iloc[zoom_start:zoom_end], label='Close Price (Zoomed)')
    zoom_buy_signals_idx = buy_signals_idx[(buy_signals_idx >= zoom_start) & (buy_signals_idx < zoom_end)]
    zoom_sell_signals_idx = sell_signals_idx[(sell_signals_idx >= zoom_start) & (sell_signals_idx < zoom_end)]
    plt.scatter(data.index[zoom_buy_signals_idx], data['Close'].iloc[zoom_buy_signals_idx], color='green', label='Buy Signal (Zoomed)', marker='^')
    plt.scatter(data.index[zoom_sell_signals_idx], data['Close'].iloc[zoom_sell_signals_idx], color='red', label='Sell Signal (Zoomed)', marker='v')
    plt.legend()
    plt.title("Zoomed-In Close Price with Signals")

    plt.tight_layout()
    plt.show()

def remove_outliers(data, features, z_thresh=3):
    """
    Remove outliers from the data based on z-score threshold.

    Parameters:
    - data (pd.DataFrame): Input data.
    - features (list): List of feature columns to check for outliers.
    - z_thresh (float): Z-score threshold to identify outliers.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    z_scores = np.abs(stats.zscore(data[features]))
    filtered_entries = (z_scores < z_thresh).all(axis=1)
    return data[filtered_entries]

def scale_features(data, features, target):
    """
    Scale the features and target using MinMaxScaler and debug normalization.

    Parameters:
    - data (pd.DataFrame): Input data with features and target columns.
    - features (list): List of feature column names.
    - target (list): List containing the target column name.

    Returns:
    - scaled_features_df (pd.DataFrame): DataFrame with scaled features and target.
    - feature_scaler (MinMaxScaler): Scaler for features.
    - target_scaler (MinMaxScaler): Scaler for the target column.
    """
    # Create scalers
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    # Copy data to avoid modifying the original DataFrame
    data = data.copy()

    # Fit scalers and transform features and target
    scaled_features = feature_scaler.fit_transform(data[features])
    scaled_target = target_scaler.fit_transform(data[target])

    # Create DataFrame for scaled data
    scaled_features_df = pd.DataFrame(scaled_features, columns=features, index=data.index)
    scaled_features_df['Target'] = scaled_target

    # Debug normalization for the target column
    debug_normalization(
        scaler=target_scaler,
        data=data,
        column_name="Target"  # Use 'Target' for both original and normalized data
    )

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

def train_model(model, optimizer, criterion, X_train, y_train):
    model.train()
    train_loss = 0.0
    y_train_pred, y_train_actual = [], []

    for i in tqdm(range(len(X_train)), desc="Training"):
        optimizer.zero_grad()
        prediction = model(X_train[i:i + 1])

        loss = criterion(prediction, y_train[i:i + 1])
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

        y_train_pred.append(prediction.detach().cpu().numpy().squeeze())
        y_train_actual.append(y_train[i].detach().cpu().numpy().squeeze())

    train_loss /= len(X_train)
    return train_loss, np.array(y_train_pred), np.array(y_train_actual)

def validate_model(model, criterion, X_val, y_val):
    model.eval()
    val_loss = 0.0
    y_val_pred, y_val_actual = [], []

    with torch.no_grad():
        for i in range(len(X_val)):
            prediction = model(X_val[i:i + 1])
            loss = criterion(prediction, y_val[i:i + 1])
            val_loss += loss.item()

            y_val_pred.append(prediction.cpu().numpy().squeeze())
            y_val_actual.append(y_val[i].cpu().numpy().squeeze())

    val_loss /= len(X_val)
    return val_loss, np.array(y_val_pred), np.array(y_val_actual)

def calculate_metrics(y_actual, y_pred, scaler, prefix=""):
    """
    Calculate a comprehensive set of metrics for model evaluation.
    
    Parameters:
    - y_actual (np.array): Actual values (normalized or raw).
    - y_pred (np.array): Predicted values (normalized or raw).
    - prefix (str): Prefix for the metric names (e.g., "Train " or "Validation ").
    - returns (np.array, optional): Trading returns for Sharpe ratio and profit factor.
    - scaler (MinMaxScaler, optional): Scaler to denormalize the data.

    Returns:
    dict: A dictionary of calculated metrics with their values.
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
        return_std = np.std(returns)
        sharpe_ratio = avg_return / return_std if return_std != 0 else 0
    else:
        sharpe_ratio = None

    # Max Drawdown
    if returns is not None and len(returns) > 1:
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = None

    # Profit Factor Proxy
    if returns is not None and len(returns) > 1:
        profits = np.sum(returns[returns > 0])
        losses = -np.sum(returns[returns < 0])
        profit_factor = profits / losses if losses != 0 else None
    else:
        profit_factor = None

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

    Args:
        y_actual (array): Actual values (normalized).
        y_pred (array): Predicted values (normalized).
        time_index (DatetimeIndex): Time associated with the predictions.
        metrics (dict): Metrics for the epoch.
        epoch (int): Current epoch number.
        model_name (str): Name of the model.
        scaler (MinMaxScaler): Scaler used for inverse transformation.
    """
    # Inverse transform y_actual and y_pred
    y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # Debugging the time_index
    print("Time Index Type:", type(time_index))
    time_index = pd.to_datetime(time_index)

    if isinstance(time_index, pd.DatetimeIndex):
        print("Time Index Timezone Info:", time_index.tz)
    # Ensure time_index is timezone-unaware
    if hasattr(time_index, "tz_localize"):
        time_index = time_index.tz_localize(None)  # Remove timezone information

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
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden_size, num_layers=lstm_layers, 
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.positional_encoding = PositionalEncoding(lstm_hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout
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
        lstm_out = self.positional_encoding(lstm_out)
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch_size, lstm_hidden_size]
        transformer_out = self.transformer_encoder(lstm_out)
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch_size, seq_len, lstm_hidden_size]
        # Use the last time step's output
        output = transformer_out[:, -1, :]
        output = self.fc(output)
        return output

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
    start_date = end_date - timedelta(days=DOWNLOAD_DAYS)  # Adjust as needed
    data = yf.download(
        TICKER,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval="1m",
        progress=False  # Suppress download progress bar
    )
    if data.empty:
        print("No data downloaded. Please check the date range and ticker symbol.")
        exit()
    else:
        print("Data downloaded successfully", len(data))

# Proceed only if data is available
if not data.empty:
    # Preprocess data
    data = preprocess_data(data)
    print("Data shape after preprocessing:", data.shape)

    # Create 'Target' column by shifting 'Close' values
    data.loc[:, 'Target'] = data['Close'].shift(-FUTURE_STEPS)
    data.dropna(subset=['Target'], inplace=True)
    print("Data shape after creating Target and dropping NaNs:", data.shape)

    # Define features and target
    features = ['Close', 'RSI', 'MACD', 'Bollinger_Band_PctB', 'ATR', 'Volume', 'Hour', 'DayOfWeek', 'Month',
                'ADX', 'Aroon_Up', 'Aroon_Down', 'Bollinger_Band_Width']
    target = ['Target']

    # Remove highly correlated features
    corr_matrix = data[features].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    print(f"Removing highly correlated features: {high_corr_features}")
    features = [f for f in features if f not in high_corr_features]
    print("Features after removing correlated ones:", features)

    # Add lagged features
    for lag in range(1, 4):
        data.loc[:, f'Close_lag_{lag}'] = data['Close'].shift(lag)
        features.append(f'Close_lag_{lag}')

    print("Features after adding lagged features:", features)

    # Drop NaNs introduced by shifting
    data.dropna(subset=features + target, inplace=True)
    print("Data shape after dropping NaNs from features and target:", data.shape)

    # Optionally, remove outliers
    # data = remove_outliers(data, features + target, z_thresh=3)
    # print("Data shape after removing outliers:", data.shape)
    if data.empty:
        print("No data left after preprocessing. Please adjust your preprocessing steps.")
        exit()

# Scale features and create sequences
scaled_data, feature_scaler, target_scaler = scale_features(data, features, target)
X, y, indices = create_sequences(scaled_data, SEQ_LEN, FUTURE_STEPS)

# Calculate total number of sequences
if len(X) == 0:
    print("No sequences created. Check SEQ_LEN and ensure it's appropriate for the dataset size.")
    exit()
total_sequences = len(X)

# Split data into training and testing sets
kf = KFold(n_splits=5)
fold = 1

# Initialize hyperparameter grid
hyperparams_grid = {
    'lstm_hidden_size': [64, 128],
    'lstm_layers': [1, 2],
    'nhead': [4, 8],
    'num_transformer_layers': [2, 4],
    'dim_feedforward': [128, 256],
    'dropout': [0.1, 0.2],
    'learning_rate': [0.001],
    'weight_decay': [0.01]
}

best_overall_val_loss = float('inf')
best_hyperparams = None
best_model_state = None

for params in product(*hyperparams_grid.values()):
    params_dict = dict(zip(hyperparams_grid.keys(), params))
    print(f"\nTraining with hyperparameters: {params_dict}")

    # Initialize model with current hyperparameters
    input_dim = X.shape[2]
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
    criterion = nn.SmoothL1Loss()

    # Cross-validation
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold}")

        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        time_indices_train = np.array(indices)[train_index]
        time_indices_val = np.array(indices)[val_index]

        # Apply data augmentation to the training fold
        X_train_augmented, y_train_augmented = augment_data(X_train_fold, y_train_fold)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_augmented, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_augmented, dtype=torch.float32).unsqueeze(-1).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(-1).to(device)

        # Reset early stopping variables
        early_stopping_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

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

            # Log metrics (set GARCH parameters to None)
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
            scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)
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

    # After training with the current hyperparameters
    if best_val_loss < best_overall_val_loss:
        best_overall_val_loss = best_val_loss
        best_hyperparams = params_dict
        final_model_state = best_model_state  # Save the best model state

print(f"\nBest hyperparameters: {best_hyperparams}")
# Load the best model state
model.load_state_dict(final_model_state)

# Save metrics
metrics_df = pd.DataFrame(epoch_metrics)
output_filename = f"training_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
save_epoch_metrics(metrics_df, output_filename)
analyze_results(metrics_df)