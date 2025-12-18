"""
Data Loader Module
Downloads historical data from Yahoo Finance and performs data cleaning.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def download_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 10
) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        years: Number of years of historical data if dates not specified

    Returns:
        DataFrame with adjusted close prices
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=years * 365)
        start_date = start.strftime('%Y-%m-%d')

    print(f"Downloading data from {start_date} to {end_date}")
    print(f"Tickers: {tickers}")

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # Extract Close prices
    if 'Close' in data.columns.get_level_values(0):
        prices = data['Close']
    else:
        prices = data

    # Ensure all tickers are present
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices


def remove_missing_values(prices: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Remove columns with too many missing values and forward fill remaining NAs.

    Args:
        prices: DataFrame with price data
        threshold: Maximum allowed proportion of missing values per column

    Returns:
        Cleaned DataFrame
    """
    # Calculate proportion of missing values per column
    missing_ratio = prices.isnull().sum() / len(prices)

    # Remove columns with too many missing values
    valid_columns = missing_ratio[missing_ratio < threshold].index.tolist()
    removed_columns = missing_ratio[missing_ratio >= threshold].index.tolist()

    if removed_columns:
        print(f"Removed tickers due to missing data (>{threshold*100}%): {removed_columns}")

    prices = prices[valid_columns]

    # Forward fill then backward fill remaining NAs
    prices = prices.ffill().bfill()

    # Drop any remaining rows with NAs
    prices = prices.dropna()

    return prices


def remove_outliers(
    returns: pd.DataFrame,
    method: str = 'zscore',
    threshold: float = 4.0
) -> pd.DataFrame:
    """
    Remove outliers from returns data.

    Args:
        returns: DataFrame with return data
        method: 'zscore' or 'iqr'
        threshold: Z-score threshold or IQR multiplier

    Returns:
        DataFrame with outliers replaced by NaN and forward filled
    """
    cleaned = returns.copy()

    if method == 'zscore':
        # Calculate z-scores
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        # Replace outliers with NaN
        cleaned = returns.where(z_scores < threshold, np.nan)

    elif method == 'iqr':
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        for col in returns.columns:
            mask = (returns[col] < lower_bound[col]) | (returns[col] > upper_bound[col])
            cleaned.loc[mask, col] = np.nan

    # Forward fill outliers
    cleaned = cleaned.ffill().bfill()

    return cleaned


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily log returns from prices.

    Args:
        prices: DataFrame with price data

    Returns:
        DataFrame with log returns
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns


def load_and_clean_data(
    tickers: List[str],
    years: int = 10,
    outlier_method: str = 'zscore',
    outlier_threshold: float = 4.0,
    missing_threshold: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Main function to load and clean data.

    Args:
        tickers: List of ticker symbols
        years: Number of years of historical data
        outlier_method: Method for outlier detection
        outlier_threshold: Threshold for outlier detection
        missing_threshold: Maximum allowed missing value ratio

    Returns:
        Tuple of (prices DataFrame, returns DataFrame, valid tickers list)
    """
    # Download data
    prices = download_data(tickers, years=years)

    # Remove columns with too many missing values
    prices = remove_missing_values(prices, threshold=missing_threshold)

    # Calculate returns
    returns = calculate_returns(prices)

    # Remove outliers
    returns = remove_outliers(returns, method=outlier_method, threshold=outlier_threshold)

    # Get valid tickers
    valid_tickers = prices.columns.tolist()

    print(f"\nData loaded successfully:")
    print(f"  - Valid tickers: {len(valid_tickers)}")
    print(f"  - Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  - Number of observations: {len(prices)}")

    return prices, returns, valid_tickers


if __name__ == "__main__":
    # Test with default tickers
    tickers = [
        # Thai Export
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        # Thai Domestic
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        # Global
        'WDC', 'THD',
        # Fixed Income
        'LEMB', 'VWOB', 'EMLC',
        # FX
        'THB=X'
    ]

    prices, returns, valid_tickers = load_and_clean_data(tickers, years=10)
    print(f"\nPrices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
