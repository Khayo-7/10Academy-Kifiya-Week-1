import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def load_data(filepath):
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)

# Text Preprocessing
def preprocess_text(text):
    """Preprocesses text with lemmatization and stop word removal."""
    
    
    if pd.isna(text):
        return ''
    
    # Tokenize, convert to lowercase, and remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Numerical Data Normalization
def normalize_numeric_columns(dataframe, numeric_columns):
    """Normalizes numeric columns using StandardScaler."""
    scaler = StandardScaler()
    dataframe[numeric_columns] = scaler.fit_transform(dataframe[numeric_columns])
    return dataframe

# Categorical Data Encoding
def encode_categorical_columns(dataframe, categorical_columns):
    """Encodes categorical columns using LabelEncoder."""
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        dataframe[column] = le.fit_transform(dataframe[column].astype(str))
        label_encoders[column] = le
    return dataframe, label_encoders


# Handle Missing Values
def handle_missing_values(dataframe, strategy='mean', columns=None):
    """Fills missing values based on the given strategy."""
    if not columns:
        columns = dataframe.columns

    for col in columns:
        if dataframe[col].isnull().sum() > 0:
            if strategy == 'mean':
                dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
            elif strategy == 'median':
                dataframe[col] = dataframe[col].fillna(dataframe[col].median())
            elif strategy == 'mode':
                dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
    return dataframe

# Data Cleaning and Preprocessing
def clean_and_preprocess_data(dataframe, date_column='date', text_columns=[], numeric_columns=[], categorical_columns=[], delete_columns=[], drop_duplicates=True, drop_na=True):
    """
    Cleans and preprocesses the data.

    Combines data cleaning and transformation steps from previous scripts.
    """
    
    # --- Data Cleaning ---
    print("[INFO] Cleaning data...")
    if delete_columns:
        dataframe = dataframe.drop(columns=delete_columns)
    if drop_duplicates:
        dataframe = dataframe.drop_duplicates()  # Remove duplicates
    if drop_na:
        dataframe = dataframe.dropna(subset=["headline", date_column, "stock"])  # Drop rows with missing values
    elif numeric_columns:
        dataframe = handle_missing_values(dataframe, strategy='mean', columns=numeric_columns)
    dataframe[date_column] = pd.to_datetime(dataframe[date_column], format='mixed', utc=True).dt.tz_convert('UTC')  # Normalize timestamps
    
    print("[INFO] Data cleaning completed.")

    # --- Data Transformation ---
    print("[INFO] Transforming data...")
    for column in text_columns:
        dataframe[column] = dataframe[column].astype(str).apply(preprocess_text)
    if numeric_columns:
        dataframe = normalize_numeric_columns(dataframe, numeric_columns)
    label_encoders = {}
    if categorical_columns:
        dataframe, label_encoders = encode_categorical_columns(dataframe, categorical_columns)
        
    print("[INFO] Data transformation completed.")

    return dataframe, label_encoders

def merge_dataframes(dataframes, tickers, column, start_date=None, end_date=None):
    
    # Set default start and end dates if not provided
    if start_date is None:
        start_date = max(dataframe['Date'].min() for dataframe in dataframes.values())
    if end_date is None:
        end_date = min(dataframe['Date'].max() for dataframe in dataframes.values())
    
    return pd.concat(
        [
            dataframes[ticker][(dataframes[ticker]['Date'] >= start_date) & (dataframes[ticker]['Date'] <= end_date)][['Date', column]]
            .rename(columns={column: ticker})
            .set_index('Date')
            for ticker in tickers
        ],
        axis=1, join='inner'
    )