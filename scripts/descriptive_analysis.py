import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_descriptive_stats(df):
    """Generate basic descriptive statistics."""
    return df.describe()

def visualize_column_distribution(df, column):
    """Visualize distribution of a specific column."""
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.show()
