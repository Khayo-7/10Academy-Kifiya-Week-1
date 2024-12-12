import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
import streamlit as st
from scipy.stats import zscore

# Helper functions
def extract_datasets(input_data_path, output_data_path):
    """Extract datasets from a zip file."""
    with zipfile.ZipFile(input_data_path, 'r') as zip:    
        zip.extractall(output_data_path)
        print(f"Extracted {input_data_path} to {output_data_path}")

# Utility: Save dataset
def save_dataset(df, save_path):
    """Save cleaned dataset to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        df.to_csv(save_path, index=False)
        print(f"Cleaned dataset saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save data to {save_path}: {e}")

# Clean dataset
def clean_dataset(df, output_path=None):
    """
    Cleans the dataset by:
        - handling missing values
        - removing duplicates
        - detecting and capping outliers using Z-score.
    """
    if df.empty:
        print("Empty DataFrame provided for cleaning.")
        return df

    # Parse datetime columns
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)  # Drop invalid/missing timestamps

    # Handle missing values
    df.ffill(inplace=True)  # Forward fill
    # df.fillna(method='ffill', inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill as fallback
    # df.fillna(method='bfill', inplace=True)  # Backward fill as fallback
    df.dropna(inplace=True)  # Drop rows with remaining NaNs

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Detect and cap outliers using Z-score
    numeric_columns = df.select_dtypes(include=['number'])
    if not numeric_columns.empty:
        zscores = numeric_columns.apply(zscore)
        df = df[(zscores.abs() <= 3).all(axis=1)]

    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset if output path is provided
    if output_path:
        save_dataset(df, output_path)

    return df

def plot_correlation_heatmap(data):
    """Plot a correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

def plot_time_series(data):
    """Plot time series trends."""
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    data[['GHI', 'DNI', 'DHI', 'Tamb']].plot(ax=ax)
    plt.title("Time Series Trends: GHI, DNI, DHI, Tamb")
    st.pyplot(fig)

def plot_wind_rose(data):
    """Create a wind rose visualization."""
    ax = WindroseAxes.from_ax()
    ax.bar(data['WD'], data['WS'], normed=True, opening=0.8, edgecolor="black")
    ax.set_title("Wind Rose: Speed vs. Direction")
    ax.set_legend()
    st.pyplot(ax.figure)

def detect_outliers(data, columns):
    """Detect outliers using z-scores."""
    z_scores = data[columns].apply(zscore)
    outliers = z_scores.abs() > 3
    return outliers.sum(axis=1).sum()

def plot_wind_direction_distribution(data):
    """Plot wind direction distribution."""
    fig, ax = plt.subplots()
    sns.histplot(data['WD'], kde=True, bins=36, ax=ax)
    plt.title("Wind Direction Distribution")
    st.pyplot(fig)

def plot_temperature_vs_humidity(data):
    """Plot temperature vs. relative humidity."""
    fig, ax = plt.subplots()
    sns.scatterplot(x="Tamb", y="RH", data=data, alpha=0.7, ax=ax)
    plt.title("Temperature vs. Relative Humidity")
    st.pyplot(fig)

def plot_temperature_trends(data):
    """Plot temperature trends across modules."""
    module_columns = ['TModA', 'TModB', 'Tamb']
    fig, ax = plt.subplots(figsize=(10, 5))
    data[module_columns].plot(ax=ax)
    plt.title("Module Temperature Trends")
    st.pyplot(fig)

def plot_pair_plot(data):
    """Create a pair plot of key variables."""
    analysis_columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS']
    available_columns = [col for col in analysis_columns if col in data.columns]
    if available_columns:
        sns.pairplot(data[available_columns])
        st.pyplot()
