import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants for default configuration
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_COLORMAP = "coolwarm" # "YlGnBu"
DEFAULT_COLOR_MAP = 'tab20'
DEFAULT_MARKER = 'o'
DEFAULT_COLOR = 'blue'

# Generic Plotting Utilities

def plot_line(data, x_column, y_column, title, xlabel="", ylabel="", 
              color=DEFAULT_COLOR, marker=DEFAULT_MARKER, linestyle="-", grid=True):
    """
    Plots a line chart with customizable options.

    Parameters:
        data (pd.DataFrame): Data for plotting
        x_column (str): Name of the column to use for the X-axis
        y_column (str): Name of the column to use for the Y-axis
        title (str): Title of the plot
        xlabel (str): Label for X-axis
        ylabel (str): Label for Y-axis
        color (str): Line color
        marker (str): Marker style
        linestyle (str): Line style
        grid (bool): Whether to show gridlines
    """
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(data[x_column], data[y_column], color=color, marker=marker, linestyle=linestyle)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if grid:
        plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_bar(column, title, xlabel="", ylabel="", color=DEFAULT_COLOR, figsize=DEFAULT_FIGSIZE):
    """
    Plots a bar chart for the specified column.

    Parameters:
        data (pd.DataFrame): Data for plotting
        column (str): Column name to count occurrences
        title (str): Title of the plot
        xlabel (str): Label for X-axis
        ylabel (str): Label for Y-axis
        color (str): Bar color
        figsize (tuple): Figure size
    """
    counts = column.value_counts()
    counts.plot(kind="bar", color=color, alpha=0.7, figsize=figsize)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Legend")
    plt.tight_layout()
    plt.show()

def plot_dataframe(dataframe, title, xlabel='', ylabel='', mode='line', marker=DEFAULT_MARKER, color=DEFAULT_COLOR, figsize=DEFAULT_FIGSIZE, grid=True):
    """Plots the trend of the column during an event."""
    
    plt.figure(figsize=figsize)
    if mode == 'line':
        dataframe.plot(kind=mode, marker=marker, color=color)
    else:
        dataframe.plot(kind=mode, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title="Legend")
    plt.grid(visible=grid)
    plt.tight_layout()
    plt.show()

def plot_dataframes(data, title, xlabel='', ylabel='', mode='line', marker=DEFAULT_MARKER, color=DEFAULT_COLOR, figsize=DEFAULT_FIGSIZE, grid=True):   
    """Plots the trends of the column."""
    
    plt.figure(figsize=figsize)

    for i, (name, dataframe) in enumerate(data.items()):
        plt.subplot(len(data), 1, i+1)

        if mode == 'line':
            dataframe.plot(kind=mode, title=title, label=name, color=color, marker=marker)
        else:
            dataframe.plot(kind=mode, title=title, label=name, color=color)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.legend(title="Legend")
        plt.grid(visible=grid)
    plt.tight_layout()
    plt.show()

def plot_sns(data, title, xlabel, ylabel, mode=None, figsize=DEFAULT_FIGSIZE, cmap=DEFAULT_COLORMAP, colormap=DEFAULT_COLOR_MAP, grid=True): 
    """ Plots a sns plot for the given data. """

    plt.figure(figsize=figsize)
    plot_func = {
        'bar': sns.barplot,
        'line': sns.lineplot,
        'scatter': sns.scatterplot,
        'hist': sns.histplot,
        'box': sns.boxplot,
        'violin': sns.violinplot,
        'kde': sns.kdeplot,
        'heatmap': sns.heatmap
    }.get(mode, None)        
    
    if plot_func:        
        if mode == 'heatmap':
            plot_func(data, annot=True, fmt="d", cmap=cmap)# , square=True, cbar=True, linewidths=0.5)
        else:
            plot_func(data) 
    else:
        data.plot(kind='bar', stacked=True, colormap=colormap)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title='Legend')
    if not mode or mode == 'line':
        plt.grid(visible=grid)
    plt.tight_layout()
    plt.show()

def generate_wordcloud(words, title="Word Cloud", figsize=(12, 6), background_color="white"):
    """
    Generates and displays a word cloud.

    Parameters:
        words (list): List of words to visualize
        title (str): Title of the word cloud
        figsize (tuple): Figure size
        background_color (str): Background color
    """
    wordcloud = WordCloud(width=800, height=400, background_color=background_color).generate(" ".join(words))
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.legend(title="Legend")
    plt.show()

# Specialized Plotting Functions
def plot_sentiment_distribution(column, title):
    """Plots the sentiment distribution."""
    
    plt.figure(figsize=(8, 5))
    # plt.hist(column, bins=15, color='blue', alpha=0.7)
    counts = column.value_counts()
    counts.plot(kind='bar', color=['Yellow', 'green', 'red'])
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel(f"Frequency of {title}")
    plt.xticks(rotation=45)
    plt.legend(title="Legend")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()    
    
def plot_sentiment_over_time(data, date_column, sentiment_column, title, xlabel="", ylabel="", color=DEFAULT_COLOR, figsize=DEFAULT_FIGSIZE, grid=True):
    """
    Plots average sentiment scores over time.

    Parameters:
        sentiment_dataframe (pd.DataFrame): Dataframe containing sentiment data
        date_column (str): Name of the date column
        sentiment_column (str): Name of the sentiment score column
        title (str): Title of the plot
    """
    plt.figure(figsize=figsize)
    sns.lineplot(x=date_column, y=sentiment_column, data=data, color=color)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(visible=grid)
    plt.tight_layout()
    plt.legend(title="Legend")
    plt.show()

def plot_correlation_matrix(data, method="pearson", title="Correlation Matrix"):
    """
    Plots a correlation matrix heatmap.

    Parameters:
        data (pd.DataFrame): Dataframe for correlation calculation
        method (str): Method to compute correlation (e.g., "pearson", "spearman")
        title (str): Title of the heatmap
    """
    corr_matrix = data.corr(method=method)
    plot_sns(corr_matrix, title=title, xlabel="", ylabel="", mode='heatmap')

def plot_correlation_heatmap(dataframe, columns):
    """Plots a heatmap of correlations between specified columns."""
    
    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Some specified columns are not in the dataset.")

    corr_matrix = dataframe[columns].corr()
    plot_sns(corr_matrix, title='Correlation Heatmap', xlabel="", ylabel="", mode='heatmap')

def plot_stock_data(hist):
    """Plots stock data (Open, High, Low, Close)."""
    
    fig = make_subplots(rows=1, cols=4, subplot_titles=['Close', 'Open', 'High', 'Low'])
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Open']), row=1, col=2)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['High']), row=1, col=3)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Low']), row=1, col=4)
    fig.update_layout(height=400, width=1200, title_text='Stock Analysis')
    
    return fig

def plot_stock_data_2(hist):
    """Plots stock data (Open, High, Low, Close) on the same figure."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['High'], mode='lines', name='High'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Low'], mode='lines', name='Low'))
    
    fig.update_layout(height=400, width=800, title_text='Stock Analysis', xaxis_title='Date', yaxis_title='Price')
    
    return fig

def plot_indicators(data, indicators, title):
    """Plots the specified indicators using Plotly."""

    fig = px.line(data, x=data.index, y=indicators, title=title)
    fig.show()
 
def plot_interactive_stock_chart(dataframe, stock_column="Close"):
    """
    Plots an interactive stock chart using Plotly.

    Parameters:
        dataframe (pd.DataFrame): Dataframe containing stock data
        stock_column (str): Name of the stock price column
    """
    fig = go.Figure(data=[
        go.Candlestick(
            x=dataframe.index,
            open=dataframe['Open'],
            high=dataframe['High'],
            low=dataframe['Low'],
            close=dataframe['Close']
        )
    ])
    fig.update_layout(title_text="Interactive Stock Chart", xaxis_title="Date", yaxis_title=stock_column)
    fig.show()

def plot_stock_vs_sentiment(merged_dataframe, date_column, sentiment_column, stock_metric):
    """Plots sentiment and stock metric over time."""
    
    if date_column not in merged_dataframe.columns or sentiment_column not in merged_dataframe.columns or stock_metric not in merged_dataframe.columns:
        raise ValueError("Required columns not found in the dataset.")

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    sns.lineplot(x=date_column, y=sentiment_column, data=merged_dataframe, color='blue', ax=ax1, label='Sentiment')
    sns.lineplot(x=date_column, y=stock_metric, data=merged_dataframe, color='green', ax=ax2, label='Stock Metric')

    ax1.set_ylabel('Average Sentiment', color='blue')
    ax2.set_ylabel(stock_metric, color='green')
    ax1.set_xlabel('Date')
    plt.title('Stock Metric vs. Sentiment Over Time')
    plt.xticks(rotation=45)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
def plot_sentiment_trends(sentiment_summary, date_column='date', sentiment_column='average_sentiment'):
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_summary[date_column], sentiment_summary[sentiment_column], marker='o', linestyle='-')
    plt.title('Sentiment Trends Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Sentiment', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Sentiment vs. Events Overlay
def plot_sentiment_with_events(sentiment_summary, events, date_column='date', sentiment_column='average_sentiment'):
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_summary[date_column], sentiment_summary[sentiment_column], marker='o', linestyle='-', label='Sentiment')
    
    for event_date, event_desc in events.items():
        plt.axvline(x=pd.to_datetime(event_date), color='red', linestyle='--', alpha=0.7)
        plt.text(pd.to_datetime(event_date), 0.1, event_desc, rotation=45, fontsize=10, color='red')
    
    plt.title('Sentiment Trends with Events', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Sentiment', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_distributions(data):
    """
    Plots the distributions of specified columns in the dataframe.

    Parameters:
    - data: DataFrame containing the data to plot.
    """
    columns = ["Close", "Volume"]
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

def plot_scatter(x, y, title, xlabel, ylabel, color='blue', alpha=0.7, figsize=(8, 6)):
    """Plots a scatter plot."""

    # data = pd.DataFrame({xlabel: x, ylabel: y})
    # plot_sns(data, title, xlabel, ylabel, mode='scatter', figsize=figsize, grid=True)
    plt.figure(figsize=figsize)
    plt.scatter(x, y, color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    

def plot_correlation_matrix(dataframe, title="Correlation Matrix"):
    """Plot correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

def plot_scatter(x, y, title, xlabel, ylabel, **kwargs):
    """Plot scatter plot for comparisons."""
    plt.scatter(x, y, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_portfolio_returns(dataframe, columns, title="Portfolio Returns"):
    """Plot cumulative returns."""
    dataframe[columns].plot(figsize=(10, 6))
    plt.title(title)
    plt.ylabel("Cumulative Returns")
    plt.show()
