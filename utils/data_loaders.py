import os
import logging
import zipfile
import pandas as pd

# Configure logging for tracking operations
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper functions
def extract_datasets(input_data_path: str, extract_path: str) -> None:
    """
    Extract datasets from a zip file.
    Args:
        input_data_path (str): Path to the input zip file.
        extract_path (str): Path to extract the contents.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    try:
        with zipfile.ZipFile(input_data_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            logging.info(f"Extracted {input_data_path} to {extract_path}")
    except Exception as e:
        logging.error(f"Failed to extract {input_data_path}: {e}")
        raise

def save_dataset(df: pd.DataFrame, save_path: str) -> None:
    """
    Save cleaned dataset to a CSV file.
    Args:
        df (pd.DataFrame): DataFrame to save.
        save_path (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        df.to_csv(save_path, index=False)
        logging.info(f"Cleaned dataset saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {save_path}: {e}")
        raise


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        logging.info(f"Loading CSV file from {filepath}")
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error while loading CSV: {e}")
        raise

def load_excel(filepath: str, sheet_name=None) -> pd.DataFrame:
    """
    Load data from an Excel file.
    Args:
        filepath (str): Path to the Excel file.
        sheet_name (str, optional): Name of the sheet to load. Defaults to None (loads the first sheet).
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        logging.info(f"Loading Excel file from {filepath}")
        return pd.read_excel(filepath, sheet_name=sheet_name)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error while loading Excel: {e}")
        raise

def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate the loaded data to ensure it contains required columns.
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    Returns:
        bool: True if valid, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    logging.info("Data validation successful.")
    return True

def load_analyst_ratings(file_path: str) -> pd.DataFrame:
    """
    Load the analyst ratings dataset.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the analyst ratings data.
    """
    try:
        logging.info(f"Loading analyst ratings from {file_path}")
        df = pd.read_csv(file_path)
        logging.info("Analyst ratings loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error while loading analyst ratings: {e}")
        raise

def load_yfinance_data(zip_path: str, folder_path: str = "./datasets/raw/yfinance_data") -> dict:
    """
    Load stock data from a yfinance zip file containing multiple CSVs.
    Args:
        zip_path (str): Path to the zip file.
        folder_path (str): Path to the folder where CSVs will be extracted and loaded.
    Returns:
        dict: Dictionary with filenames as keys and DataFrames as values.
    """
    data_frames = {}
    
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            logging.info(f"Creating folder: {folder_path}")
            os.makedirs(folder_path)

        # Extract datasets if the folder is empty
        if not os.listdir(folder_path):
            logging.info(f"Extracting datasets from {zip_path} to {folder_path}")
            extract_datasets(zip_path, folder_path)

        # Load CSV files from the folder
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                stock_name = file.split('.')[0]
                logging.info(f"Loading {file}...")
                data_frames[stock_name] = pd.read_csv(os.path.join(folder_path, file))
                logging.info(f"{file} loaded successfully!")

        return data_frames

    except FileNotFoundError:
        logging.error(f"File not found: {zip_path}")
        raise
    except Exception as e:
        logging.error(f"Error while loading data from zip file: {e}")
        raise
