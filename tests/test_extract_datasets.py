import os
from utils.data_loaders import extract_datasets

def test_extract_analyst_ratings_datasets():
    zip_path = r"../datasets/raw_analyst_ratings.csv.zip"
    extract_path = r"./datasets/raw"
    extract_datasets(zip_path, extract_path)
    assert os.path.exists(extract_path), "Extracted dataset not found"

def test_extract_yfinance_data_datasets():
    zip_path = r"../datasets/yfinance_data.zip"
    extract_path = r"./datasets/raw/yfinance_data"
    extract_datasets(zip_path, extract_path)
    assert os.path.exists(extract_path), "Extracted datasets not found"

