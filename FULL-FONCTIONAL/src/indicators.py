# src/indicators.py
import ta
from ta.momentum import RSIIndicator, ROCIndicator, TSIIndicator, UltimateOscillator
from ta.trend import MACD, EMAIndicator, CCIIndicator, ADXIndicator, IchimokuIndicator
from ta.volume import ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from src.display import colored_print, print_indicators

def add_polynomial_and_interaction_features(df, features, degree=2, interaction_only=False, include_bias=False):
    """
    Adds polynomial and interaction features for specific features.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features.
    - features (list): List of feature names to include in polynomial feature generation.
    - degree (int): The degree of the polynomial features.
    - interaction_only (bool): If True, only interaction features are produced.
    - include_bias (bool): If True, includes a bias column (column of ones).
    
    Returns:
    - df_poly (pd.DataFrame): DataFrame with the new polynomial features added.
    """
    # Initialize PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    
    # Fit and transform the data
    poly_features = poly.fit_transform(df[features])
    
    # Get feature names
    poly_feature_names = poly.get_feature_names_out(features)
    
    # Create a DataFrame with the new features
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df_poly = df_poly.add_prefix('poly_')
    # Drop the original features to prevent duplication if desired
    # df_poly = df_poly.drop(columns=features, errors='ignore')
    
    # Concatenate with the original DataFrame
    df = pd.concat([df, df_poly], axis=1)
    df = df.dropna()
    return df

def add_rolling_statistics(df, windows=[5, 10, 15]):
    """
    Adds rolling mean and standard deviation of 'Close' price.
    """
    for window in windows:
        df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
    df = df.dropna()
    return df

def add_lag_features(df, lags=3):
    """
    Adds lagged versions of features to capture temporal dependencies.
    """
    for lag in range(1, lags + 1):
        df[f'CloseLag_{lag}'] = df['Close'].shift(lag)
        df[f'VolumeLag_{lag}'] = df['Volume'].shift(lag)
    df = df.dropna()
    return df

def calculate_psar(df, step=0.02, max_step=0.2):
    psar = df['Low'].copy()
    psar.iloc[0] = df['Low'].iloc[0]
    bullish = True
    af = step
    ep = df['High'].iloc[0]

    for i in range(1, len(df)):
        previous_psar = psar.iloc[i-1]
        if bullish:
            psar.iloc[i] = previous_psar + af * (ep - previous_psar)
            psar.iloc[i] = min(psar.iloc[i], df['Low'].iloc[i-1], df['Low'].iloc[i])
            if df['Low'].iloc[i] < psar.iloc[i]:
                bullish = False
                psar.iloc[i] = ep
                ep = df['Low'].iloc[i]
                af = step
        else:
            psar.iloc[i] = previous_psar + af * (ep - previous_psar)
            psar.iloc[i] = max(psar.iloc[i], df['High'].iloc[i-1], df['High'].iloc[i])
            if df['High'].iloc[i] > psar.iloc[i]:
                bullish = True
                psar.iloc[i] = ep
                ep = df['High'].iloc[i]
                af = step

        if bullish:
            if df['High'].iloc[i] > ep:
                ep = df['High'].iloc[i]
                af = min(af + step, max_step)
        else:
            if df['Low'].iloc[i] < ep:
                ep = df['Low'].iloc[i]
                af = min(af + step, max_step)
    df = df.dropna()
    return psar

def add_volume_oscillator(df, window_fast=12, window_slow=26):
    df['MA_Fast_Volume'] = df['Volume'].rolling(window=window_fast).mean()
    df['MA_Slow_Volume'] = df['Volume'].rolling(window=window_slow).mean()
    df['VO'] = (df['MA_Fast_Volume'] - df['MA_Slow_Volume']) / df['MA_Slow_Volume'] * 100
    df.drop(['MA_Fast_Volume', 'MA_Slow_Volume'], axis=1, inplace=True)
    df = df.dropna()
    return df

def add_pivot_points(df):
    # Create a copy to avoid the warning
    df = df.copy()

    df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PP']) - df['Low']
    df['S1'] = (2 * df['PP']) - df['High']
    df['R2'] = df['PP'] + (df['High'] - df['Low'])
    df['S2'] = df['PP'] - (df['High'] - df['Low'])
    df['R3'] = df['High'] + 2 * (df['PP'] - df['Low'])
    df['S3'] = df['Low'] - 2 * (df['High'] - df['PP'])

    df = df.dropna()
    return df



def add_technical_indicators(
    df, 
    horizon=1, 
    indicator_types=None, 
    metrics_folder='metrics'
):
    # Default to all indicators if indicator_types is not provided
    if indicator_types is None:
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

    # Add indicators conditionally based on indicator_types
    

    # Trend Indicators
    if indicator_types.get("trend", True):
        colored_print("Add Trand Indicators", "green")
        df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
        df["MACD"] = MACD(close=df["Close"]).macd_diff()
        df["EMA"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df["CCI"] = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()
        df["ADX"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).adx()

    # Momentum Indicators
    if indicator_types.get("momentum", True):
        colored_print("Add Momentum Indicators", "green")
        df["ROC"] = ROCIndicator(close=df["Close"], window=20).roc()
        df["TSI"] = TSIIndicator(close=df["Close"]).tsi()
        df["UO"] = UltimateOscillator(high=df["High"], low=df["Low"], close=df["Close"]).ultimate_oscillator()

    # Volume Indicators
    if indicator_types.get("volume", True):
        colored_print("Add Volume Indicators", "green")
        df["CMF"] = ChaikinMoneyFlowIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=20
        ).chaikin_money_flow()

    # Volatility Indicators
    if indicator_types.get("volatility", True):
        colored_print("Add Volatility Indicators", "green")
        df["ATR"] = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        ).average_true_range()
        df["DC_H"] = ta.volatility.DonchianChannel(
            high=df["High"], low=df["Low"], close=df["Close"]
        ).donchian_channel_hband()
        df["DC_L"] = ta.volatility.DonchianChannel(
            high=df["High"], low=df["Low"], close=df["Close"]
        ).donchian_channel_lband()
        df["DC_M"] = ta.volatility.DonchianChannel(
            high=df["High"], low=df["Low"], close=df["Close"]
        ).donchian_channel_mband()
        df["DC_Width"] = df["DC_H"] - df["DC_L"]

    # Additional Indicators
    if indicator_types.get("additional", True):
        colored_print("Add Additional Indicators", "green")
        ichimoku = IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52)
        df["ICHIMOKU_A"] = ichimoku.ichimoku_a()
        df["ICHIMOKU_B"] = ichimoku.ichimoku_b()
        df["PSAR"] = calculate_psar(df)
        df["VWAP"] = VolumeWeightedAveragePrice(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
        ).volume_weighted_average_price()

    # Vol Oscillator
    if indicator_types.get("vol_oscillator", True):
        colored_print("Add VO Indicators", "green")
        df = add_volume_oscillator(df, window_fast=12, window_slow=26)

    # Pivot Points
    if indicator_types.get("pivot_point", True):
        colored_print("Add PP Indicators", "green")
        df = add_pivot_points(df)

    # Lag Features
    if indicator_types.get("lag", True):
        colored_print("Add Lag Indicators", "green")
        df = add_lag_features(df, 10)

    # Rolling Statistics
    if indicator_types.get("rolling", True):
        colored_print("Add Rolling Indicators", "green")
        df = add_rolling_statistics(df)

    # Polynomial Features
    features_to_expand = ['Close', 'Volume', 'RSI', 'MACD', "VO"]
    if indicator_types.get("polynomial", True):
        df = add_polynomial_and_interaction_features(
            df,
            features=features_to_expand,
            degree=2,
            interaction_only=False,
            include_bias=False
        )

    # Add the target column and clean up
    df['Target'] = df['Close'].shift(-horizon)
    df = df[:-horizon]
    df = df.dropna()

    # Print indicators (assuming print_indicators is defined)
    print_indicators(df)

    return df
