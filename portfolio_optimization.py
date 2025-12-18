"""
Portfolio Optimization Module
Mean-variance portfolio optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List


def calculate_portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252
) -> Tuple[float, float]:
    """
    Calculate annualized portfolio return and volatility.

    Args:
        weights: Portfolio weights
        mean_returns: Mean daily returns
        cov_matrix: Covariance matrix of returns
        trading_days: Number of trading days per year

    Returns:
        Tuple of (annualized return, annualized volatility)
    """
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix * trading_days, weights))
    )
    return portfolio_return, portfolio_volatility


def negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> float:
    """
    Calculate negative Sharpe ratio (for minimization).

    Args:
        weights: Portfolio weights
        mean_returns: Mean daily returns
        cov_matrix: Covariance matrix of returns
        risk_free_rate: Annual risk-free rate
        trading_days: Number of trading days per year

    Returns:
        Negative Sharpe ratio
    """
    p_return, p_volatility = calculate_portfolio_performance(
        weights, mean_returns, cov_matrix, trading_days
    )

    sharpe = (p_return - risk_free_rate) / p_volatility if p_volatility > 0 else 0
    return -sharpe


def portfolio_volatility(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252
) -> float:
    """
    Calculate portfolio volatility (for minimum variance optimization).
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'sharpe',
    risk_free_rate: float = 0.02,
    target_return: Optional[float] = None
) -> Dict:
    """
    Perform mean-variance portfolio optimization.

    Args:
        returns: DataFrame with asset returns
        objective: 'sharpe' for max Sharpe ratio, 'min_variance' for minimum variance
        risk_free_rate: Annual risk-free rate
        target_return: Target annual return (for efficient frontier point)

    Returns:
        Dictionary with optimization results
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)
    tickers = returns.columns.tolist()

    # Initial guess: equal weights
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    # Add target return constraint if specified
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(mean_returns * x) * 252 - target_return
        })

    # Bounds: no short selling (weights between 0 and 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Choose objective function
    if objective == 'sharpe':
        obj_func = lambda w: negative_sharpe_ratio(
            w, mean_returns, cov_matrix, risk_free_rate
        )
    elif objective == 'min_variance':
        obj_func = lambda w: portfolio_volatility(w, mean_returns, cov_matrix)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Optimize
    result = minimize(
        obj_func,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    if not result.success:
        print(f"Warning: Optimization may not have converged: {result.message}")

    optimal_weights = result.x
    p_return, p_volatility = calculate_portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )
    sharpe = (p_return - risk_free_rate) / p_volatility if p_volatility > 0 else 0

    return {
        'weights': dict(zip(tickers, optimal_weights.tolist())),
        'expected_return': p_return,
        'volatility': p_volatility,
        'sharpe_ratio': sharpe,
        'success': result.success,
        'message': result.message
    }


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate the efficient frontier.

    Args:
        returns: DataFrame with asset returns
        n_points: Number of points on the frontier
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame with efficient frontier points
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    # Find min and max returns
    min_var_result = optimize_portfolio(returns, objective='min_variance')
    min_return = min_var_result['expected_return']

    # Max return is the return of the best single asset
    max_return = np.max(mean_returns) * 252 * 0.95

    target_returns = np.linspace(min_return, max_return, n_points)
    frontier_volatilities = []
    frontier_returns = []

    for target in target_returns:
        try:
            result = optimize_portfolio(
                returns,
                objective='min_variance',
                target_return=target
            )
            if result['success']:
                frontier_returns.append(result['expected_return'])
                frontier_volatilities.append(result['volatility'])
        except Exception:
            continue

    return pd.DataFrame({
        'return': frontier_returns,
        'volatility': frontier_volatilities
    })


def backtest_portfolio(
    weights: Dict[str, float],
    returns: pd.DataFrame,
    initial_value: float = 100.0
) -> Dict:
    """
    Backtest portfolio performance.

    Args:
        weights: Dictionary of asset weights
        returns: DataFrame with asset returns
        initial_value: Starting portfolio value

    Returns:
        Dictionary with backtest results
    """
    # Align weights with returns columns
    weight_array = np.array([weights.get(col, 0) for col in returns.columns])

    # Calculate portfolio daily returns
    portfolio_returns = (returns.values * weight_array).sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = np.exp(np.cumsum(portfolio_returns))
    portfolio_values = initial_value * cumulative_returns

    # Calculate performance metrics
    total_return = (portfolio_values[-1] / initial_value - 1) * 100
    annualized_return = (np.mean(portfolio_returns) * 252) * 100
    annualized_volatility = (np.std(portfolio_returns) * np.sqrt(252)) * 100
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    # Maximum drawdown
    rolling_max = pd.Series(portfolio_values).cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = np.min(drawdowns) * 100

    return {
        'portfolio_values': portfolio_values.tolist(),
        'dates': returns.index.strftime('%Y-%m-%d').tolist(),
        'daily_returns': portfolio_returns.tolist(),
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'annualized_volatility_pct': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'final_value': portfolio_values[-1]
    }


if __name__ == "__main__":
    # Test with sample data
    from data_loader import load_and_clean_data

    tickers = [
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        'WDC', 'THD', 'LEMB', 'VWOB', 'EMLC', 'THB=X'
    ]

    prices, returns, valid_tickers = load_and_clean_data(tickers, years=10)

    # Optimize portfolio
    result = optimize_portfolio(returns, objective='sharpe')
    print("\nOptimal Portfolio (Max Sharpe):")
    print(f"  Expected Return: {result['expected_return']*100:.2f}%")
    print(f"  Volatility: {result['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for ticker, weight in result['weights'].items():
        if weight > 0.01:
            print(f"  {ticker}: {weight*100:.2f}%")

    # Backtest
    backtest = backtest_portfolio(result['weights'], returns)
    print(f"\nBacktest Results:")
    print(f"  Total Return: {backtest['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {backtest['max_drawdown_pct']:.2f}%")
