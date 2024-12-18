import pandas as pd
import numpy as np
import talib as ta
import pyfolio as pf
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models
# from pypfopt.efficient_frontier import EfficientFrontier

def clean_stock_data(data):
    """Cleans the stock data and sets 'Date' as index."""
    
    cleaned_data = data.copy()
    cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
    cleaned_data.index = cleaned_data['Date']
    return cleaned_data

def generate_signals(data):
    """Generates trading signals based on RSI."""
    
    signals = pd.Series(0, index=data.index)  # Neutral
    signals[data['RSI'] < 30] = 1  # Buy
    signals[data['RSI'] > 70] = -1  # Sell
    return signals

def calculate_RSI(data, price_column='Close'):
    """Calculate Relative Strength Index (RSI)."""

    return ta.RSI(data[price_column], timeperiod=14)

def calculate_volatility(high, low, open_price):
    """Calculate daily stock volatility for given high, low, and open prices."""

    return (high - low) / open_price

def calculate_daily_return(data, price_column='Close'):
    """Calculates daily returns."""
    
    return data[price_column].pct_change()

def calculate_strategy_return(data):
    """Calculates strategy returns based on signals."""
    
    strategy_returns = data['Signal'].shift(1) * data['Daily_Return']
    return strategy_returns.dropna()

def get_strategy_returns(dataframes, tickers):
    """
    Calculates strategy returns for multiple tickers, ensuring that the final DataFrame
    has 'Date' as the index and tickers as the columns.
    """
    
    def process_strategy_returns(dataframe):
        dataframe = dataframe.copy()
        dataframe['Signal'] = generate_signals(dataframe)
        dataframe['Daily_Return'] = calculate_daily_return(dataframe)
        dataframe['Strategy_Return'] = calculate_strategy_return(dataframe)
        return dataframe[['Date', 'Strategy_Return']]  # Preserve Date explicitly

    strategy_returns = []

    for ticker in tickers:
        ticker_data = process_strategy_returns(dataframes[ticker])        
        ticker_data.set_index('Date', inplace=True)# Ensure 'Date' is used as index for consistency
        ticker_data.rename(columns={'Strategy_Return': ticker}, inplace=True) 
        strategy_returns.append(ticker_data)

    return pd.concat(strategy_returns, axis=1)

def create_performance_tear_sheet(strategy_returns):
    """Creates a performance tear sheet using pyfolio."""
    
    return pf.create_simple_tear_sheet(strategy_returns)

def create_performance_tear_sheet_multiple(strategy_returns, tickers=None):
    """
    Creates a performance tear sheet using pyfolio.
    
    Parameters:
    - strategy_returns: DataFrame containing strategy returns for multiple tickers.
    - tickers: List of tickers to include in the tear sheet. If None, includes all tickers.
    
    Returns:
    - Pyfolio tear sheet.
    """
    
    if tickers is not None:
        strategy_returns = strategy_returns[tickers]
    
    return pf.create_simple_tear_sheet(strategy_returns.mean(axis=1).dropna())

def calculate_technical_indicators(data, price_column='Close'):
    """ 
    Calculate various technical indicators based on a specified price column and return them as a dictionary.
    
    Parameters:
    - data: DataFrame containing the stock data.
    - price_column: The column name to use for price-based calculations (default is 'Close').
    
    Returns:
    - indicators: Dictionary of calculated technical indicators.
    """
    
    indicators = {}
    indicators['SMA'] = ta.SMA(data[price_column], timeperiod=20)
    indicators['RSI'] = ta.RSI(data[price_column], timeperiod=14)
    indicators['EMA'] = ta.EMA(data[price_column], timeperiod=20)
    macd, macd_signal, macd_hist = ta.MACD(data[price_column])
    indicators['MACD'] = macd
    indicators['MACD_Signal'] = macd_signal
    indicators['MACD_Hist'] = macd_hist
    indicators['WMA'] = ta.WMA(data[price_column], timeperiod=20)
    indicators['ROC'] = ta.ROC(data[price_column], timeperiod=10)
    indicators['MACD_hist'] = indicators['MACD'] - indicators['MACD_Signal']

    return indicators

def calculate_multi_value_indicators(data):
    """ 
    Calculate technical indicators that require multiple data columns and return them as a dictionary.
    
    Parameters:
    - data: DataFrame containing the stock data.
    
    Returns:
    - indicators: Dictionary of calculated technical indicators that require multiple columns.
    """
    
    indicators = {}
    bb_upper, bb_middle, bb_lower = ta.BBANDS(data['Close'], timeperiod=20)
    indicators['BB_upper'] = bb_upper
    indicators['BB_middle'] = bb_middle
    indicators['BB_lower'] = bb_lower
    indicators['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    indicators['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    indicators['CCI'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=20)

    return indicators

def calculate_portfolio_indicators(data):
    """
    Calculate expected returns and covariance matrix for the given price data.

    Parameters:
    - data: DataFrame of historical prices (columns = tickers).

    Returns:
    - (mu, cov): Tuple of mean returns and covariance matrix.
    """
    mu = expected_returns.mean_historical_return(data)
    cov = risk_models.sample_cov(data)
    
    return mu, cov
    # return efficient_frontier.EfficientFrontier(mu, cov)

def calculate_portfolio_weights(data, tickers):
    """
    Calculate optimal portfolio weights.

    Parameters:
    - data: DataFrame of historical prices.
    - tickers: List of tickers corresponding to the data columns.

    Returns:
    - weights: Dictionary of portfolio weights.
    """
    mu, cov = calculate_portfolio_indicators(data)
    ef = EfficientFrontier(mu, cov)
    weights = ef.max_sharpe()
    return dict(zip(tickers, weights.values()))

def analyze_portfolio_performance(returns, weights=None):
    """
    Analyze the performance of a portfolio using pyfolio.
    You can optionally provide `weights` for weighted performance analysis.
    If no weights are provided, an equal-weighted portfolio will be used.
    """
    if weights is None:
        # If no weights provided, use equal weights
        weights = [1.0 / len(returns.columns)] * len(returns.columns)
    
    # Calculate the portfolio's weighted returns if weights are provided
    weighted_returns = (returns * weights).sum(axis=1)
    
    # Use PyFolio to analyze the portfolio's performance
    pf.create_simple_tear_sheet(weighted_returns)
    pf.create_turnover_tear_sheet(weighted_returns)
    
    return weighted_returns

def calculate_portfolio_performance(data):
    """
    Calculate portfolio performance indicators (return, volatility, Sharpe ratio).

    Parameters:
    - data: DataFrame of historical prices.

    Returns:
    - (portfolio_return, portfolio_volatility, sharpe_ratio): Performance metrics.
    """
    mu, cov = calculate_portfolio_indicators(data)
    ef = EfficientFrontier(mu, cov)

    # portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    # return portfolio_return, portfolio_volatility, sharpe_ratio
    return ef.portfolio_performance()

# Portfolio Optimization 
def optimize_portfolio(returns, risk_free_rate=0.0, method='sharpe'):
    """
    Performs portfolio optimization using various methods.

    Methods include:
    - 'sharpe': Maximize Sharpe Ratio (custom implementation or via PyPortfolioOpt).
    - 'min_volatility': Minimize portfolio volatility.
    - 'efficient_frontier': Maximize Sharpe Ratio via mean-variance optimization (PyPortfolioOpt).
    - 'risk_parity': Allocate weights using hierarchical risk parity (PyPortfolioOpt).
    - 'equal_weight': Assign equal weights to all assets.

    Args:
        returns (pd.DataFrame): DataFrame of historical asset returns.
        risk_free_rate (float, optional): Annualized risk-free rate of return. Defaults to 0.0.
        method (str, optional): Optimization method. Choose from 
                                ['sharpe', 'min_volatility', 'efficient_frontier', 'risk_parity', 'equal_weight'].
                                Defaults to 'sharpe'.

    Returns:
        dict: A dictionary containing:
              - 'weights': Optimized portfolio weights (list of floats).
              - 'performance': Performance metrics (return, volatility, Sharpe ratio).
    """
    # Number of assets
    num_assets = returns.shape[1]

    # Define objective functions
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualized return
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Minimize negative Sharpe ratio

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility

    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
    bounds = [(0, 1)] * num_assets  # Weights between 0 and 1
    initial_weights = np.array([1 / num_assets] * num_assets)  # Equal weight initialization

    if method == 'sharpe':
        # Maximize Sharpe ratio using custom optimization
        result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x

    elif method == 'min_volatility':
        # Minimize portfolio volatility
        result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x

    elif method == 'efficient_frontier':
        # Efficient frontier using PyPortfolioOpt
        mu = expected_returns.mean_historical_return(returns)
        S = risk_models.sample_covariance(returns)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return {'weights': cleaned_weights, 'performance': {'return': performance[0], 'volatility': performance[1], 'sharpe_ratio': performance[2]}}

    elif method == 'risk_parity':
        # Risk parity using PyPortfolioOpt
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        performance = hrp.portfolio_performance(verbose=False)
        return {'weights': weights, 'performance': {'return': performance[0], 'volatility': performance[1], 'sharpe_ratio': performance[2]}}

    elif method == 'equal_weight':
        # Equal weights strategy
        weights = np.array([1 / num_assets] * num_assets)
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return {'weights': weights.tolist(), 'performance': {'return': portfolio_return, 'volatility': portfolio_volatility, 'sharpe_ratio': sharpe_ratio}}

    else:
        raise ValueError(f"Invalid method '{method}'. Choose from ['sharpe', 'min_volatility', 'efficient_frontier', 'risk_parity', 'equal_weight'].")

    # Performance metrics (for custom Sharpe and Min-Volatility methods)
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return {'weights': weights.tolist(), 'performance': {'return': portfolio_return, 'volatility': portfolio_volatility, 'sharpe_ratio': sharpe_ratio}}


# def calculate_RSI_2(series, period=14):
#     """Calculate Relative Strength Index (RSI)."""
#     delta = series.diff()
#     gain = np.where(delta > 0, delta, 0)
#     loss = np.where(delta < 0, -delta, 0)
#     avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
#     avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
#     rs = avg_gain / avg_loss
#     return 100 - (100 / (1 + rs))

# def calculate_portfolio_performance_2(dataframe, weights):
#     """Calculate portfolio performance metrics."""

#     daily_returns = dataframe.pct_change()
#     portfolio_return = (daily_returns * weights).sum(axis=1).mean()
#     portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov(), weights)))
#     sharpe_ratio = portfolio_return / portfolio_risk

#     return portfolio_return, portfolio_risk, sharpe_ratio

def calculate_portfolio_indicators_2(dataframe):
    """Calculate and print portfolio indicators like mean return and risk."""
    
    return_mean = dataframe.pct_change().mean() * 252
    return_std = dataframe.pct_change().std() * np.sqrt(252)
    return return_mean, return_std

def combine_stock_data(dataframes, tickers):
    """ Combine closing prices of multiple tickers into a single DataFrame. """
    
    stock_prices = pd.DataFrame({ticker: dataframes[ticker]['Close'] for ticker in tickers})
    stock_prices = stock_prices.align(dataframes[tickers[0]][['Date']], join='inner', axis=0)
    return stock_prices

def calculate_returns_for_tickers(stock_prices):
    """ Calculate daily returns for the combined stock data. """

    return stock_prices.pct_change().dropna()

