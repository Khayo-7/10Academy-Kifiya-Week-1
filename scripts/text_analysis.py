import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams

def get_all_keywords(column):
    """Generates all keywords from a column in a DataFrame."""
    
    # Remove punctuation and split into words
    return re.findall(r'\w+', column.str.cat(sep=" ").lower())

def extract_keywords_tfidf(column, max_features=20, ngram_range=(1, 2)):
    """
    Extracts top keywords using TF-IDF from a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        column_name (str): The name of the column to extract keywords from.
        max_features (int): The maximum number of features (keywords) to extract.
        ngram_range (tuple): The range of n-grams to consider.

    Returns:
        list: A list of tuples containing the top keywords and their scores.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(column)

    # Get feature names (keywords)
    keywords = vectorizer.get_feature_names_out()
    keyword_scores = X.sum(axis=0).A1 # Sum the scores of each feature (keyword)
    top_keywords = sorted(zip(keywords, keyword_scores), key=lambda x: x[1], reverse=True)

    return keywords, top_keywords

def extract_phrases_ngrams(column, n=2):
    """Extracts n-grams from a specified column."""

    return ngrams(" ".join(column).split(), n)