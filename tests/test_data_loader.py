from utils.data_loaders import load_analyst_ratings, load_yfinance_data

def test_load_analyst_ratings():
    file_path = r"./datasets/raw/raw_analyst_ratings.csv"
    df = load_analyst_ratings(file_path)
    assert not df.empty, "Analyst Ratings file is empty"

def test_load_yfinance_data():
    folder_path = r"./datasets/raw/yfinance_data"
    datasets = load_yfinance_data(folder_path)
    assert len(datasets) == 7, "Expected 7 datasets, found fewer"
