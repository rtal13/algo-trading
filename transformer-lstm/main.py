import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ta
import os
from datetime import datetime
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import pearsonr
from matplotlib.patches import Rectangle



# ================================
# Globals
# ================================

PLOT_FOLDER = None
SEQ_LEN = 27
FUTURE_STEPS = 2
EPOCHS = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_PERCENTAGE = 0.05
MODE = 0


# ================================
# Helper Functions
# ================================

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

def preprocess_data(data):
    """
    Add technical indicators, time-based features, and scale data.

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


def scale_features(data, features, target):
    """Scale the features and target using MinMaxScaler."""
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    data = data.copy()  # Ensure `data` is not a view
    data["Target"] = data["Close"].shift(-FUTURE_STEPS)
    scaled_features = feature_scaler.fit_transform(data[features])
    scaled_target = target_scaler.fit_transform(data[target])

    scaled_features_df = pd.DataFrame(scaled_features, columns=features, index=data.index)
    scaled_features_df['Target'] = scaled_target

    return scaled_features_df, feature_scaler, target_scaler


def create_sequences(data, seq_len, future_steps):
    """Generate sequences for training/testing."""
    X, y = [], []
    for i in range(seq_len, len(data) - future_steps):
        X.append(data.iloc[i - seq_len:i].values)
        y.append(data['Target'].iloc[i])
    return np.array(X), np.array(y)


def save_plot(folder, y_actual, y_predicted, epoch, model_name):
    """
    Save plots of actual vs predicted values with enhancements:
    - Highlight divergences
    - Add trendlines
    - Annotate correlation coefficient
    - Include shaded areas for significant differences.
    """
    # Calculate trendlines
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
    """Train the model and return the average training loss and predictions."""
    model.train()
    train_loss = 0.0
    y_train_pred, y_train_actual = [], []

    for i in tqdm(range(len(X_train)), desc="Training"):
        optimizer.zero_grad()
        outputs = model(X_train[i:i + 1])
        loss = criterion(outputs, y_train[i:i + 1])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Collect predictions and actuals for metrics
        y_train_pred.append(outputs.detach().cpu().numpy().squeeze())
        y_train_actual.append(y_train[i].detach().cpu().numpy().squeeze())

    train_loss /= len(X_train)
    return train_loss, np.array(y_train_pred), np.array(y_train_actual)


def validate_model(model, criterion, X_test, y_test):
    """Validate the model and return the average validation loss and predictions."""
    model.eval()
    val_loss = 0.0
    y_val_pred, y_val_actual = [], []

    with torch.no_grad():
        for i in range(len(X_test)):
            outputs = model(X_test[i:i + 1])
            val_loss += criterion(outputs, y_test[i:i + 1]).item()
            y_val_pred.append(outputs.detach().cpu().numpy().squeeze())
            y_val_actual.append(y_test[i].detach().cpu().numpy().squeeze())

    val_loss /= len(X_test)
    return val_loss, np.array(y_val_pred), np.array(y_val_actual)


def calculate_metrics(y_actual, y_pred, prefix=""):
    """Calculate RMSE, R², and MAE for given predictions."""
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)

    metrics = {
        f"{prefix}RMSE": rmse,
        f"{prefix}R²": r2,
        f"{prefix}MAE": mae
    }
    return metrics


def calculate_pip_difference(y_pred, y_actual):
    """Calculate pip difference and mean absolute pip difference."""
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
    
def save_predictions_and_metrics(y_actual, y_pred, time_index, metrics, epoch, model_name):
    """
    Save predictions, actuals, time, and metrics for each epoch to an Excel file.

    Args:
        y_actual (array): Actual values.
        y_pred (array): Predicted values.
        time_index (DatetimeIndex): Time or date associated with the predictions.
        metrics (dict): Metrics for the epoch.
        epoch (int): Current epoch number.
        model_name (str): Name of the model.
    """
    # Ensure time_index is timezone-unaware
    if hasattr(time_index, "tz_localize"):
        time_index = time_index.tz_localize(None)

    # Create a DataFrame to store predictions and actual values
    predictions_df = pd.DataFrame({
        "Time": time_index,
        "Actual Price": y_actual.flatten(),
        "Predicted Price": y_pred.flatten(),
        "Pip Difference": (y_pred.flatten() - y_actual.flatten()) * 10000,
    })

    # Add metrics as a summary at the bottom of the DataFrame
    metrics_summary = pd.DataFrame.from_dict([metrics])
    metrics_summary.index = ["Metrics"]
    predictions_df = pd.concat([predictions_df, metrics_summary])

    # Generate a filename
    filename = f"predictions_metrics_epoch_{epoch}_{model_name}.xlsx"

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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        return self.norm2(src + src2)


class TransformerLSTM(nn.Module):
    def __init__(self, seq_len, d_model=64, num_heads=4, num_encoder_layers=2, dim_feedforward=128, lstm_units=64, input_dim=8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward) for _ in range(num_encoder_layers)
        ])
        self.lstm = nn.LSTM(d_model, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, 1)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# ================================
# Main Workflow
# ================================

# Download data
if (MODE == 0):
    data = load_forex_data("EURUSD_M1.csv")
    print("Data loaded successfully", len(data))
else:
    data = yf.download("AAPL", start="2024-11-12", end="2024-11-19", interval="1m")
    print("Data downloaded successfully", len(data))

# Preprocess data
data = preprocess_data(data)
indicators = ['RSI', 'MACD', 'Bollinger_Band_PctB', 'ATR', 'Volume', 'Stochastic_K', 'Stochastic_D', 'Williams_R', 'EMA_12', 'EMA_26', 'ADX', 'Aroon_Up', 'Aroon_Down', 'Bollinger_Band_Width']
plot_indicators(data, train_split_index=int(len(data) * 0.8), indicators=indicators)

# Define features and target

features = ['Close', 'RSI', 'MACD', 'Bollinger_Band_PctB', 'ATR', 'Volume', 'Hour', 'DayOfWeek', 'Month', 'Stochastic_K', 'Stochastic_D', 'Williams_R', 'EMA_12', 'EMA_26', 'ADX', 'Aroon_Up', 'Aroon_Down', 'Bollinger_Band_Width']
target = ['Target']

# Scale features and create sequences
scaled_data, feature_scaler, target_scaler = scale_features(data, features, target)
X, y = create_sequences(scaled_data, SEQ_LEN, FUTURE_STEPS)

# Split data into training and testing sets
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_PERCENTAGE, random_state=42)

# Initialize models
input_dim = X_train.shape[2]
models = {
    "Transformer-LSTM": TransformerLSTM(seq_len=SEQ_LEN, d_model=64, input_dim=input_dim).to(device)
    }
optimizers = {name: optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
schedulers = {name: ReduceLROnPlateau(optimizer, patience=3, factor=0.5) for name, optimizer in optimizers.items()}
criterion = nn.MSELoss()

# ================================
# Main Training Loop
# ================================

# Track best validation RMSE
best_val_rmse = float('inf')
epoch_metrics = []  # To store metrics for the report

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    for name, model in models.items():
        optimizer = optimizers[name]
        scheduler = schedulers[name]

        # Train the model
        train_loss, y_train_pred, y_train_actual = train_model(model, optimizer, criterion, X_train, y_train)

        # Validate the model
        val_loss, y_val_pred, y_val_actual = validate_model(model, criterion, X_test, y_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train_actual, y_train_pred, prefix="Train ")
        val_metrics = calculate_metrics(y_val_actual, y_val_pred, prefix="Validation ")
        _, val_abs_pip_diff = calculate_pip_difference(y_val_pred, y_val_actual)

        # Scheduler step
        scheduler.step(val_loss)

        # Update best validation RMSE
        if val_metrics["Validation RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["Validation RMSE"]
            torch.save(model.state_dict(), f"best_model_{name}.pth")
            print(f"Saved new best model for {name} with RMSE: {best_val_rmse:.5f}")

        # Print metrics
        print(f"\n{name} - Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}")
        print(f"  Train RMSE: {train_metrics['Train RMSE']:.5f}, Validation RMSE: {val_metrics['Validation RMSE']:.5f}")
        print(f"  Train R²: {train_metrics['Train R²']:.5f}, Validation R²: {val_metrics['Validation R²']:.5f}")
        print(f"  Validation Mean Absolute Pip Difference: {val_abs_pip_diff:.2f} pips")

        save_predictions_and_metrics(
            y_actual=target_scaler.inverse_transform(y_val_actual.reshape(-1, 1)),
            y_pred=target_scaler.inverse_transform(y_val_pred.reshape(-1, 1)),
            time_index=scaled_data.index[-len(y_val_actual):],
            metrics=val_metrics,
            epoch=epoch + 1,
            model_name=name
        )
        # Store metrics for reporting
        epoch_metrics.append({
            "Epoch": epoch + 1,
            "Model": name,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            **train_metrics,
            **val_metrics,
            "Validation Pip Difference": val_abs_pip_diff
        })

        # Save plot for the epoch
        save_plot(
            get_plot_folder(),
            y_val_actual,
            y_val_pred,
            epoch + 1,
            name
        )

# ================================
# Generate Report
# ================================

metrics_df = pd.DataFrame(epoch_metrics)
output_filename = f"training_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
save_epoch_metrics(metrics_df, output_filename)
analyze_results(metrics_df)