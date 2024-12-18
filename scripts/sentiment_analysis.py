import pandas as pd
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .data_analysis import calculate_correlation

nltk.download('vader_lexicon')

def get_sentiment_score(phrase):
    """Performs sentiment analysis on a single phrase/text."""
    
    # Safely handle non-string input
    if not isinstance(phrase, str):
        phrase = str(phrase) if phrase else ""
    analysis = TextBlob(phrase)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    
def map_sentiment(polarity):
    """Maps sentiment analysis to a single phrase/text."""
    
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def analyze_sentiment(column):
    """Performs sentiment analysis on phrases/texts in a DataFrame."""
    
    sentiment = column.fillna("").astype(str).apply(lambda x: get_sentiment_score(x)[0])
    # sentiment = column.apply(lambda x: get_sentiment_analysis(str(x))[0] if pd.notnull(x) else 0)
    sentiment_dataframe = pd.DataFrame({'sentiment': sentiment, 'sentiment_category': sentiment.apply(map_sentiment)})
    return sentiment_dataframe

# Sentiment Aggregation
def aggregate_sentiment(dataframe, text_column, sentiment_column, date_column, method='textblob'):
    """
    Aggregates sentiment scores over time and computes average sentiment per date.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing sentiment and date data.
        sentiment_column (str): The column name for sentiment scores. If it does not exist, it will be generated using text_column.
        text_column (str): The column name for text data. Must be present in the DataFrame.
        date_column (str): The column name for dates.
        method (str): The sentiment analysis method to use ('textblob' or 'vader').

    Returns:
        pd.DataFrame: A DataFrame with average sentiment aggregated by date.
    """

    # Ensure the required columns exist
    if text_column not in dataframe.columns:
        raise ValueError(f"The text column '{text_column}' must exist in the DataFrame.")
    if date_column not in dataframe.columns:
        raise ValueError(f"The date column '{date_column}' must exist in the DataFrame.")

    # Check if sentiment_column exists
    if sentiment_column not in dataframe.columns:
        # Generate sentiment scores if sentiment_column does not exist
        if method == 'textblob':
            dataframe[sentiment_column] = dataframe[text_column].apply(lambda x: get_sentiment_score(x)[0])
        elif method == 'vader':
            dataframe[sentiment_column] = dataframe[text_column].apply(get_vader_sentiment)
        else:
            raise ValueError("Invalid method. Choose 'textblob' or 'vader'.")

    # Aggregate by date and calculate average sentiment
    sentiment_summary = dataframe.groupby(dataframe[date_column].dt.date)[sentiment_column].mean().reset_index().rename(columns={sentiment_column: 'average_sentiment'})
    sentiment_summary['average_sentiment_category'] = sentiment_summary['average_sentiment'].apply(map_sentiment)

    return sentiment_summary

# Aspect-Based Sentiment Analysis 
def analyze_aspect_sentiment(text, aspects):
    """
    Performs aspect-based sentiment analysis using TextBlob.

    Args:
        text: The text to analyze.
        aspects: A list of aspects to consider.

    Returns:
        A dictionary mapping aspects to sentiment scores.
    """

    blob = TextBlob(text)
    aspect_sentiments = {}
    for aspect in aspects:
        for sentence in blob.sentences:
            if aspect in sentence:
                aspect_sentiments[aspect] = sentence.sentiment.polarity
                break 

    return aspect_sentiments

# Stock and Sentiment Comparison
def analyze_sentiment_stock(dataframe_sentiment, dataframe_stock, date_column, stock_metric):
    """Merges sentiment and stock data and performs comparative analysis."""
    if date_column not in dataframe_sentiment.columns or date_column not in dataframe_stock.columns:
        raise ValueError("Date column not found in one or both datasets.")
    if stock_metric not in dataframe_stock.columns:
        raise ValueError("Stock metric column not found in stock dataset.")

    merged_dataframe = pd.merge(dataframe_sentiment, dataframe_stock, on=date_column, how='inner')
    corr, p_value = calculate_correlation(merged_dataframe, 'average_sentiment', stock_metric)

    return merged_dataframe, {'correlation': corr, 'p_value': p_value}

# Function to calculate VADER sentiment
def get_vader_sentiment(text):
    """Get sentiment score using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound'] # Compound score summarizes polarity (-1 to 1)

def lda_topic_modeling(text_data, n_components=5, max_features=1000):
    """Perform topic modeling on news headlines."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    term_matrix = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(term_matrix)
    return lda, vectorizer.get_feature_names_out()

def cluster_sentiment(dataframe, sentiment_column, n_clusters=3):
    """Cluster sentiment scores."""
    sentiment_array = np.array(dataframe[sentiment_column]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(sentiment_array)
    dataframe['Sentiment_Cluster'] = clusters
    return dataframe
