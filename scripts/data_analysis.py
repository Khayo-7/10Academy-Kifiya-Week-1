import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest

# Data Summarization
def summarize_data(dataframe):
    """Generates a summary of the dataset with key statistics."""
    summary = {
        'shape': dataframe.shape,
        'missing_values': dataframe.isnull().sum().to_dict(),
        'duplicates': dataframe.duplicated().sum(),
        'unique_values': dataframe.nunique().to_dict(),
        'column_stats': dataframe.describe(include='all').to_dict()
    }
    return summary

def get_top_10(column):
    """Returns the top 10 values in a column."""

    top_10 = column.value_counts().head(10)
    # top_10 = column.value_counts().reset_index().head(10)
    top_10.columns = [column.name.title(), 'Count']
    return top_10

def analyze_top_ten(dataframe, column, column_to_analyze):
    """Analyzes the top 10 values in a column."""
    top_10 = get_top_10(dataframe[column])
    for top in top_10.index:
        print(f"\nArticles by {top}:")
        print(dataframe[dataframe[column] == top][column_to_analyze].head(5))
        
def check_email_publishers(dataframe, column="publisher"):
    """Checks if email addresses are used as publisher names."""
    email_publishers = dataframe[dataframe[column].str.contains('@', na=False)]
    return email_publishers[column].str.extract(r'@([\w\.-]+)')[0].value_counts()
    

def get_missing_dates(dataframe, date_column='date'):
    """Gets the missing dates in the column."""

    date_range = pd.date_range(start=dataframe[date_column].min(), end=dataframe[date_column].max(), freq='D')
    missing_dates = date_range.difference(dataframe[date_column])
    return missing_dates, dataframe[date_column].min(), dataframe[date_column].max()

def column_time_analysis(dataframe, column_name, date_column='date'):
    """Analyzes the publication dates to see trends over time."""

    # Normalize timestamps
    dataframe[date_column] = pd.to_datetime(dataframe[date_column], format='mixed', utc=True).dt.tz_convert('UTC')

    # Analyze the publication dates to see trends over time 
    dataframe[f'{column_name}_year'] = dataframe[date_column].dt.year
    dataframe[f'{column_name}_month'] = dataframe[date_column].dt.month
    dataframe[f'{column_name}_day'] = dataframe[date_column].dt.day
    dataframe[f'{column_name}_weekday'] = dataframe[date_column].dt.dayofweek
    dataframe[f'{column_name}_weekday_name'] = dataframe[date_column].dt.day_name()
    dataframe[f'{column_name}_hour'] = dataframe[date_column].dt.hour
    dataframe[f'{column_name}_year_month'] = dataframe[date_column].dt.to_period('M') # Extract year and month from the date

    return dataframe

def get_counts_and_spikes(dataframe, title, column_name='date'):
    """Gets the counts and identifies spikes in the data."""

    if column_name not in dataframe.columns:
        raise ValueError(f"{column_name} not found in the DataFrame.")

    # Group by the specified column to get the count
    counts = dataframe.groupby(column_name).size()
    # counts_dataframe = counts.sort_index().reset_index()
    # counts_dataframe.columns = [column_name.replace('_', ' ').title(), f'Number of {title}']

    if counts.empty:
        raise ValueError("No data available to identify spikes.")

    # Identify spikes
    spikes = counts[counts > counts.mean() + 2 * counts.std()]

    return counts, spikes

def get_event_trends(dataframe, start_date, end_date, date_column='date'):
    """Gets the trends of the column during an event."""

    dataframe[date_column] = pd.to_datetime(dataframe[date_column], format='mixed', utc=True).dt.tz_convert('UTC')
    event_data = dataframe[(dataframe[date_column] >= start_date) & (dataframe[date_column] <= end_date)]

    # Analyze counts during the event
    event_trends = event_data[date_column].dt.date.value_counts().sort_index()

    return event_trends

def get_column_counts(column, title):
    """Gets the counts of the column."""
    
    return column.value_counts().sort_index()
    # column_dataframe = column.value_counts().sort_index().reset_index()
    # column_dataframe.columns = [column.name.replace('_', ' ').title(), f'Number of {title}']

    # return column_dataframe

def calculate_correlations(dataframe, method='pearson'):
    """Calculates correlations between numeric columns."""

    return dataframe.corr(method=method)

# Correlation Analysis
def calculate_correlation(dataframe, col1, col2):
    """Calculates Pearson correlation coefficient and p-value."""

    if col1 not in dataframe.columns or col2 not in dataframe.columns:
        raise ValueError("Columns not found in the dataset.")
    
    series1 = dataframe[col1]
    series2 = dataframe[col2]

    if series1.isnull().any() or series2.isnull().any():
        raise ValueError("Columns contain NaN values; handle them before correlation analysis.")

    corr, p_value = pearsonr(series1, series2)
    return corr, p_value

# Time-based Aggregation
def time_based_aggregation(dataframe, time_column, agg_column, freq='M', agg_func='mean'):
    """Performs time-based aggregation."""

    if time_column not in dataframe.columns:
        raise ValueError(f"{time_column} not found in the DataFrame.")

    dataframe[time_column] = pd.to_datetime(dataframe[time_column])
    aggregated = dataframe.set_index(time_column).resample(freq)[agg_column].agg(agg_func).reset_index()
    return aggregated

# Time Series Analysis 
def analyze_time_series(dataframe, time_column, value_column):
    """
    Performs time series analysis techniques.

    Techniques include:
    - Rolling Mean
    - Rolling Standard Deviation
    - Exponential Weighted Moving Average (EWMA)
    - Decomposition (Trend, Seasonality, Residuals)
    - Autocorrelation and Partial Autocorrelation Functions (ACF/PACF)

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        time_column (str): The name of the time column.
        value_column (str): The name of the value column to analyze.

    Returns:
        pd.DataFrame: The DataFrame with added analysis columns.
    """

  
    dataframe['rolling_mean'] = dataframe[value_column].rolling(window=7).mean() # Rolling Mean
    dataframe['rolling_std'] = dataframe[value_column].rolling(window=7).std()# Rolling Standard Deviation
    dataframe['ewma'] = dataframe[value_column].ewm(span=7).mean() # Exponential Weighted Moving Average (EWMA)

    # Decomposition
    # decomposition = seasonal_decompose(dataframe[value_column], model='additive', period=7)  
    decomposition = sm.tsa.seasonal_decompose(dataframe[value_column], model='additive', period=7)
    dataframe['trend'] = decomposition.trend
    dataframe['seasonal'] = decomposition.seasonal
    dataframe['residual'] = decomposition.resid

    # ACF and PACF (for plotting)
    plot_acf(dataframe[value_column], lags=30) 
    plot_pacf(dataframe[value_column], lags=30)

    return dataframe

# Outlier Detection 
def detect_outliers(dataframe, column, method='iqr'):
    """
    Detects outliers using various methods.

    Methods include:
    - IQR (Interquartile Range)
    - Z-score
    - Modified Z-score
    # - Isolation Forest (from scikit-learn)  

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze for outliers.
        method (str, optional): The outlier detection method to use. 
                                 Defaults to 'iqr'.

    Returns:
        pd.DataFrame: A DataFrame containing the outlier data points.
    """

    if method == 'iqr':
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((dataframe[column] - dataframe[column].mean()) / dataframe[column].std())
        outliers = dataframe[(z_scores > 3)]  # Threshold for outliers 
    elif method == 'modified_zscore':
        median = dataframe[column].median()
        mad = np.median(np.abs(dataframe[column] - median))  # Median Absolute Deviation
        modified_z_scores = 0.6745 * (dataframe[column] - median) / mad
        outliers = dataframe[(np.abs(modified_z_scores) > 3.5)]  # Threshold for outliers
    elif method == 'isolation_forest':
        model = IsolationForest(contamination='auto') 
        dataframe['outlier'] = model.fit_predict(dataframe[[column]])
        outliers = dataframe[dataframe['outlier'] == -1]

    return outliers