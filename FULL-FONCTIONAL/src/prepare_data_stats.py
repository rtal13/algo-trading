import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_descriptive_stats(df):
    """
    Computes descriptive statistics for each feature in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features.
    
    Returns:
    - stats (pd.DataFrame): A DataFrame containing the descriptive statistics.
    """
    stats = df.describe().transpose()
    return stats



def plot_correlation_matrix(df):
    """
    Plots the correlation matrix heatmap of the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and target.
    """
    corr = df.corr()
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
def plot_feature_distributions(df, features):
    """
    Plots the distribution of specified features.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features.
    - features (list): List of feature names to plot.
    """
    import math
    num_features = len(features)
    num_cols = 3
    num_rows = math.ceil(num_features / num_cols)
    
    plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_time_series(df, features, start_date=None, end_date=None):
    """
    Plots time series of specified features.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features.
    - features (list): List of feature names to plot.
    - start_date (str): Start date for the plot (optional).
    - end_date (str): End date for the plot (optional).
    """
    df_plot = df.copy()
    if start_date and end_date:
        df_plot = df_plot.loc[start_date:end_date]
    
    plt.figure(figsize=(14, 7))
    for feature in features:
        plt.plot(df_plot.index, df_plot[feature], label=feature)
    
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, title='Time Series'):
    """
    Performs the Augmented Dickey-Fuller test to check stationarity.
    
    Parameters:
    - timeseries (pd.Series): The time series to test.
    - title (str): Title for the plot.
    """
    # Rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(14, 7))
    plt.plot(timeseries, label='Original')
    plt.plot(rolmean, label='Rolling Mean')
    plt.plot(rolstd, label='Rolling Std')
    plt.title(f'Rolling Mean & Standard Deviation ({title})')
    plt.legend()
    plt.show()
    
    # Perform Augmented Dickey-Fuller test
    print(f'Results of Dickey-Fuller Test ({title}):')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(timeseries, lags=50, title='Time Series'):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
    
    Parameters:
    - timeseries (pd.Series): The time series data.
    - lags (int): Number of lags to display.
    - title (str): Title for the plots.
    """
    plt.figure(figsize=(14, 7))
    
    plt.subplot(121)
    plot_acf(timeseries.dropna(), lags=lags, ax=plt.gca())
    plt.title(f'Autocorrelation Function ({title})')
    
    plt.subplot(122)
    plot_pacf(timeseries.dropna(), lags=lags, ax=plt.gca())
    plt.title(f'Partial Autocorrelation Function ({title})')
    
    plt.tight_layout()
    plt.show()
    
def compute_feature_target_correlation(df, target_column='Target'):
    """
    Computes the correlation between each feature and the target variable.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing features and target.
    - target_column (str): Name of the target column.
    
    Returns:
    - corr_df (pd.DataFrame): DataFrame of feature-target correlations.
    """
    corr = df.corr()[target_column].drop(target_column)
    corr_df = corr.sort_values(ascending=False).to_frame('Correlation with Target')
    return corr_df

def check_missing_data(df):
    """
    Checks for missing data in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to check.
    
    Returns:
    - missing_data (pd.Series): Series indicating the count of missing values per column.
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    return missing_data

def detect_outliers_zscore(df, threshold=3.0):
    """
    Detects outliers in the DataFrame using the Z-score method.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing features.
    - threshold (float): Z-score threshold to consider a data point an outlier.
    
    Returns:
    - outliers (dict): Dictionary with feature names as keys and indices of outliers as values.
    """
    from scipy import stats
    outliers = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            if len(outlier_indices) > 0:
                outliers[col] = outlier_indices
    return outliers

from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_time_series(timeseries, model='additive', freq=None):
    """
    Decomposes the time series into trend, seasonal, and residual components.
    
    Parameters:
    - timeseries (pd.Series): The time series data.
    - model (str): 'additive' or 'multiplicative' model.
    - freq (int): Frequency of the series (e.g., 12 for monthly data).
    
    Returns:
    - decomposition: Decomposition object.
    """
    decomposition = seasonal_decompose(timeseries.dropna(), model=model, period=freq)
    decomposition.plot()
    plt.show()
    return decomposition

def analyse_prepared_data(df, essential_feature):
    """
    Performs exploratory data analysis on the prepared data.
    
    Parameters:
    - df (pd.DataFrame): The prepared DataFrame with features and target.
    """
    # 1. Descriptive Statistics
    stats = compute_descriptive_stats(df)
    print("Descriptive Statistics:")
    print(stats)
    
    # 2. Correlation Matrix
    plot_correlation_matrix(df)
    # essential_feature = [f for f in essential_feature if f not in ["CloseLag", "VolumeLag"]]

    # 3. Feature Distributions
    essential_features = ['Target', 'Close', "High", "Low", "Open", "Volume", "MACD", "RSI", "ROC", "VO", "ATR"]
    features_to_plot = [essential_features]
    # plot_feature_distributions(df, features_to_plot)
    
    # 4. Time Series Visualization
    # plot_time_series(df, ['Close', 'Target', 'Price_Diff'])
    
    # # 5. Stationarity Test
    # test_stationarity(df['Log_Returns'], title='Log Returns')
    
    # # 6. Autocorrelation and Partial Autocorrelation
    # plot_acf_pacf(df['Log_Returns'], lags=50, title='Log Returns')
    
    # # 7. Feature Correlation with Target
    # corr_with_target = compute_feature_target_correlation(df, target_column='Target')
    # print("Feature Correlation with Target:")
    # print(corr_with_target)
    
    # 8. Missing Data Analysis
    missing_data = check_missing_data(df)
    if missing_data.empty:
        print("No missing data.")
    else:
        print("Missing data in the following columns:")
        print(missing_data)
    
    # 9. Outlier Detection
    # outliers = detect_outliers_zscore(df)
    # if outliers:
    #     print("Outliers detected in the following features:")
    #     for feature, indices in outliers.items():
    #         print(f"{feature}: {len(indices)} outliers")
    # else:
    #     print("No significant outliers detected.")
    
    # 10. Seasonality Analysis
    # decomposition = decompose_time_series(df['Log_Returns'], model='additive', freq=60*24)
    # if decomposition is not None:
    #     print("Decomposition completed.")