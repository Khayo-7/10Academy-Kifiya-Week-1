import pandas as pd

def correlate_sentiment_and_prices(sentiment_df, stock_df):
    """Compute correlation between sentiment scores and stock prices."""
    combined = pd.merge(sentiment_df, stock_df, on="Date")
    correlation = combined["sentiment"].corr(combined["Close"])
    return correlation
