import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest

# Data Summarization
def summarize_data(df):
    """Generates a summary of the dataset with key statistics."""
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'unique_values': df.nunique().to_dict(),
        'column_stats': df.describe(include='all').to_dict()
    }
    return summary

def get_top_10(column):
    """Returns the top 10 values in a column."""

    top_10 = column.value_counts().head(10)
    # top_10 = column.value_counts().reset_index().head(10)
    top_10.columns = [column.name.title(), 'Count']
    return top_10

def analyze_top_ten(df, column, column_to_analyze):
    """Analyzes the top 10 values in a column."""
    top_10 = get_top_10(df[column])
    for top in top_10.index:
        print(f"\nArticles by {top}:")
        print(df[df[column] == top][column_to_analyze].head(5))
        
def check_email_publishers(df, column="publisher"):
    """Checks if email addresses are used as publisher names."""
    email_publishers = df[df[column].str.contains('@', na=False)]
    return email_publishers[column].str.extract(r'@([\w\.-]+)')[0].value_counts()
    

def get_missing_dates(df, date_column='date'):
    """Gets the missing dates in the column."""

    date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max(), freq='D')
    missing_dates = date_range.difference(df[date_column])
    return missing_dates, df[date_column].min(), df[date_column].max()

def column_time_analysis(df, column_name, date_column='date'):
    """Analyzes the publication dates to see trends over time."""

    # Normalize timestamps
    df[date_column] = pd.to_datetime(df[date_column], format='mixed', utc=True).dt.tz_convert('UTC')

    # Analyze the publication dates to see trends over time 
    df[f'{column_name}_year'] = df[date_column].dt.year
    df[f'{column_name}_month'] = df[date_column].dt.month
    df[f'{column_name}_day'] = df[date_column].dt.day
    df[f'{column_name}_weekday'] = df[date_column].dt.dayofweek
    df[f'{column_name}_weekday_name'] = df[date_column].dt.day_name()
    df[f'{column_name}_hour'] = df[date_column].dt.hour
    df[f'{column_name}_year_month'] = df[date_column].dt.to_period('M') # Extract year and month from the date

    return df

def get_counts_and_spikes(df, title, column_name='date'):
    """Gets the counts and identifies spikes in the data."""

    if column_name not in df.columns:
        raise ValueError(f"{column_name} not found in the DataFrame.")

    # Group by the specified column to get the count
    counts = df.groupby(column_name).size()
    # counts_df = counts.sort_index().reset_index()
    # counts_df.columns = [column_name.replace('_', ' ').title(), f'Number of {title}']

    if counts.empty:
        raise ValueError("No data available to identify spikes.")

    # Identify spikes
    spikes = counts[counts > counts.mean() + 2 * counts.std()]

    return counts, spikes

def get_event_trends(df, start_date, end_date, date_column='date'):
    """Gets the trends of the column during an event."""

    df[date_column] = pd.to_datetime(df[date_column], format='mixed', utc=True).dt.tz_convert('UTC')
    event_data = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

    # Analyze counts during the event
    event_trends = event_data[date_column].dt.date.value_counts().sort_index()

    return event_trends

def get_column_counts(column, title):
    """Gets the counts of the column."""
    
    return column.value_counts().sort_index()
    # column_df = column.value_counts().sort_index().reset_index()
    # column_df.columns = [column.name.replace('_', ' ').title(), f'Number of {title}']

    # return column_df

def calculate_correlations(df, method='pearson'):
    """Calculates correlations between numeric columns."""

    return df.corr(method=method)

# Correlation Analysis
def calculate_correlation(df, col1, col2):
    """Calculates Pearson correlation coefficient and p-value."""

    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Columns not found in the dataset.")
    
    series1 = df[col1]
    series2 = df[col2]

    if series1.isnull().any() or series2.isnull().any():
        raise ValueError("Columns contain NaN values; handle them before correlation analysis.")

    corr, p_value = pearsonr(series1, series2)
    return corr, p_value

# Time-based Aggregation
def time_based_aggregation(df, time_column, agg_column, freq='M', agg_func='mean'):
    """Performs time-based aggregation."""

    if time_column not in df.columns:
        raise ValueError(f"{time_column} not found in the DataFrame.")

    df[time_column] = pd.to_datetime(df[time_column])
    aggregated = df.set_index(time_column).resample(freq)[agg_column].agg(agg_func).reset_index()
    return aggregated

# Time Series Analysis 
def analyze_time_series(df, time_column, value_column):
    """
    Performs time series analysis techniques.

    Techniques include:
    - Rolling Mean
    - Rolling Standard Deviation
    - Exponential Weighted Moving Average (EWMA)
    - Decomposition (Trend, Seasonality, Residuals)
    - Autocorrelation and Partial Autocorrelation Functions (ACF/PACF)

    Args:
        df (pd.DataFrame): The input DataFrame.
        time_column (str): The name of the time column.
        value_column (str): The name of the value column to analyze.

    Returns:
        pd.DataFrame: The DataFrame with added analysis columns.
    """

  
    df['rolling_mean'] = df[value_column].rolling(window=7).mean() # Rolling Mean
    df['rolling_std'] = df[value_column].rolling(window=7).std()# Rolling Standard Deviation
    df['ewma'] = df[value_column].ewm(span=7).mean() # Exponential Weighted Moving Average (EWMA)

    # Decomposition
    # decomposition = seasonal_decompose(df[value_column], model='additive', period=7)  
    decomposition = sm.tsa.seasonal_decompose(df[value_column], model='additive', period=7)
    df['trend'] = decomposition.trend
    df['seasonal'] = decomposition.seasonal
    df['residual'] = decomposition.resid

    # ACF and PACF (for plotting)
    plot_acf(df[value_column], lags=30) 
    plot_pacf(df[value_column], lags=30)

    return df

# Outlier Detection 
def detect_outliers(df, column, method='iqr'):
    """
    Detects outliers using various methods.

    Methods include:
    - IQR (Interquartile Range)
    - Z-score
    - Modified Z-score
    # - Isolation Forest (from scikit-learn)  

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze for outliers.
        method (str, optional): The outlier detection method to use. 
                                 Defaults to 'iqr'.

    Returns:
        pd.DataFrame: A DataFrame containing the outlier data points.
    """

    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[(z_scores > 3)]  # Threshold for outliers 
    elif method == 'modified_zscore':
        median = df[column].median()
        mad = np.median(np.abs(df[column] - median))  # Median Absolute Deviation
        modified_z_scores = 0.6745 * (df[column] - median) / mad
        outliers = df[(np.abs(modified_z_scores) > 3.5)]  # Threshold for outliers
    elif method == 'isolation_forest':
        model = IsolationForest(contamination='auto') 
        df['outlier'] = model.fit_predict(df[[column]])
        outliers = df[df['outlier'] == -1]

    return outliers