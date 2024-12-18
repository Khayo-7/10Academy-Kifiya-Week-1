import pandas as pd
from textblob import TextBlob
from .data_analysis import calculate_correlation

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
    sentiment_df = pd.DataFrame({'sentiment': sentiment, 'sentiment_category': sentiment.apply(map_sentiment)})
    return sentiment_df

# Sentiment Aggregation
def aggregate_sentiment(df, sentiment_column, date_column):
    """
    Aggregates sentiment scores over time and computes average sentiment per date.

    Parameters:
        df (pd.DataFrame): DataFrame containing sentiment and date data.
        sentiment_column (str): The column name for sentiment scores.
        date_column (str): The column name for dates.

    Returns:
        pd.DataFrame: A DataFrame with average sentiment aggregated by date.
    """

    # Ensure the required columns exist
    if sentiment_column not in df.columns or date_column not in df.columns:
        raise ValueError(f"Columns '{sentiment_column}' and '{date_column}' must exist in the DataFrame.")

    # Aggregate by date and calculate average sentiment
    sentiment_summary = df.groupby(date_column)[sentiment_column].mean().reset_index().rename(columns={sentiment_column: 'average_sentiment'})
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
def analyze_sentiment_stock(df_sentiment, df_stock, date_column, stock_metric):
    """Merges sentiment and stock data and performs comparative analysis."""
    if date_column not in df_sentiment.columns or date_column not in df_stock.columns:
        raise ValueError("Date column not found in one or both datasets.")
    if stock_metric not in df_stock.columns:
        raise ValueError("Stock metric column not found in stock dataset.")

    merged_df = pd.merge(df_sentiment, df_stock, on=date_column, how='inner')
    corr, p_value = calculate_correlation(merged_df, 'average_sentiment', stock_metric)

    return merged_df, {'correlation': corr, 'p_value': p_value}

