from textblob import TextBlob

def add_sentiment_scores(df, column="headline"):
    """Add sentiment scores and categories to a DataFrame."""
    df["sentiment"] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment_category"] = pd.cut(df["sentiment"], bins=[-1, -0.1, 0.1, 1],
                                      labels=["negative", "neutral", "positive"])
    return df

def visualize_sentiment_distribution(df, sentiment_col="sentiment_category"):
    """Plot the distribution of sentiment categories."""
    sns.countplot(data=df, x=sentiment_col, palette="coolwarm")
    plt.title("Sentiment Distribution")
    plt.show()
