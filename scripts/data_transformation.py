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
def normalize_numeric_columns(df, numeric_columns):
    """Normalizes numeric columns using StandardScaler."""
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Categorical Data Encoding
def encode_categorical_columns(df, categorical_columns):
    """Encodes categorical columns using LabelEncoder."""
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    return df, label_encoders


# Handle Missing Values
def handle_missing_values(df, strategy='mean', columns=None):
    """Fills missing values based on the given strategy."""
    if not columns:
        columns = df.columns

    for col in columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Data Cleaning and Preprocessing
def clean_and_preprocess_data(df, date_column='date', text_columns=[], numeric_columns=[], categorical_columns=[], delete_columns=[], drop_duplicates=True, drop_na=True):
    """
    Cleans and preprocesses the data.

    Combines data cleaning and transformation steps from previous scripts.
    """
    
    # --- Data Cleaning ---
    print("[INFO] Cleaning data...")
    if delete_columns:
        df = df.drop(columns=delete_columns)
    if drop_duplicates:
        df = df.drop_duplicates()  # Remove duplicates
    if drop_na:
        df = df.dropna(subset=["headline", date_column, "stock"])  # Drop rows with missing values
    elif numeric_columns:
        df = handle_missing_values(df, strategy='mean', columns=numeric_columns)
    df[date_column] = pd.to_datetime(df[date_column], format='mixed', utc=True).dt.tz_convert('UTC')  # Normalize timestamps
    
    print("[INFO] Data cleaning completed.")

    # --- Data Transformation ---
    print("[INFO] Transforming data...")
    for column in text_columns:
        df[column] = df[column].astype(str).apply(preprocess_text)
    if numeric_columns:
        df = normalize_numeric_columns(df, numeric_columns)
    label_encoders = {}
    if categorical_columns:
        df, label_encoders = encode_categorical_columns(df, categorical_columns)
        
    print("[INFO] Data transformation completed.")

    return df, label_encoders

def merge_dataframes(dataframes, tickers, column, start_date=None, end_date=None):
    
    # Set default start and end dates if not provided
    if start_date is None:
        start_date = max(df['Date'].min() for df in dataframes.values())
    if end_date is None:
        end_date = min(df['Date'].max() for df in dataframes.values())
    
    return pd.concat(
        [
            dataframes[ticker][(dataframes[ticker]['Date'] >= start_date) & (dataframes[ticker]['Date'] <= end_date)][['Date', column]]
            .rename(columns={column: ticker})
            .set_index('Date')
            for ticker in tickers
        ],
        axis=1, join='inner'
    )