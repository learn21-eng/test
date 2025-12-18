"""
Scenario Modeling Module
Stress testing and forward-looking simulation using Geometric Brownian Motion (GBM).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StressScenario:
    """
    Configuration for a stress scenario.

    Attributes:
        name: Scenario name for identification
        start_date: When the stress scenario begins
        duration_days: How long the stress event lasts (trading days)
        affected_assets: List of tickers affected by the stress
        expected_drop_pct: Expected percentage drop for affected assets (e.g., -10 for 10% drop)
        drop_std_pct: Standard deviation of the drop (e.g., 2 for ±2%)
        volatility_increase_pct: Percentage increase in volatility (e.g., 10 for 10% increase)
    """
    name: str
    start_date: Optional[str] = None  # 'YYYY-MM-DD' format, None means start immediately
    duration_days: int = 252  # Default 1 year
    affected_assets: List[str] = field(default_factory=list)
    expected_drop_pct: float = -10.0  # 10% drop
    drop_std_pct: float = 2.0  # ±2% variation
    volatility_increase_pct: float = 10.0  # 10% increase in volatility


def create_trump_tariff_scenario(
    thai_assets: List[str],
    start_date: Optional[str] = None,
    duration_days: int = 252,
    expected_drop_pct: float = -10.0,
    drop_std_pct: float = 2.0,
    volatility_increase_pct: float = 10.0
) -> StressScenario:
    """
    Create a Trump tariff stress scenario affecting Thai assets.

    Args:
        thai_assets: List of Thai-related asset tickers
        start_date: When the stress begins
        duration_days: Duration in trading days
        expected_drop_pct: Expected percentage drop
        drop_std_pct: Standard deviation of drop
        volatility_increase_pct: Volatility increase percentage

    Returns:
        Configured StressScenario
    """
    return StressScenario(
        name="Trump Tariffs Impact",
        start_date=start_date,
        duration_days=duration_days,
        affected_assets=thai_assets,
        expected_drop_pct=expected_drop_pct,
        drop_std_pct=drop_std_pct,
        volatility_increase_pct=volatility_increase_pct
    )


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_simulations: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.

    dS = mu * S * dt + sigma * S * dW

    Args:
        S0: Initial price
        mu: Drift (annualized)
        sigma: Volatility (annualized)
        T: Time horizon in years
        dt: Time step in years (e.g., 1/252 for daily)
        n_simulations: Number of simulation paths
        seed: Random seed for reproducibility

    Returns:
        Array of simulated price paths (n_simulations x n_steps)
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)

    # Generate random shocks
    dW = np.random.normal(0, np.sqrt(dt), (n_simulations, n_steps))

    # Calculate price paths using GBM formula
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dW

    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=1)

    # Prepend initial price
    prices = S0 * np.exp(np.hstack([np.zeros((n_simulations, 1)), log_prices]))

    return prices


def apply_stress_to_parameters(
    base_mu: float,
    base_sigma: float,
    scenario: StressScenario,
    is_affected: bool
) -> Tuple[float, float]:
    """
    Apply stress scenario adjustments to drift and volatility.

    Args:
        base_mu: Base annualized drift
        base_sigma: Base annualized volatility
        scenario: Stress scenario configuration
        is_affected: Whether this asset is affected by the stress

    Returns:
        Tuple of (stressed_mu, stressed_sigma)
    """
    if not is_affected:
        return base_mu, base_sigma

    # Apply the expected drop as a one-time shock distributed over the period
    # Convert percentage to decimal and annualize based on duration
    drop_decimal = scenario.expected_drop_pct / 100.0
    drop_std_decimal = scenario.drop_std_pct / 100.0

    # Random drop within the specified range
    actual_drop = np.random.normal(drop_decimal, drop_std_decimal)

    # Adjust drift to incorporate the expected drop over the simulation period
    duration_years = scenario.duration_days / 252.0
    stressed_mu = base_mu + (actual_drop / duration_years)

    # Increase volatility
    vol_multiplier = 1 + (scenario.volatility_increase_pct / 100.0)
    stressed_sigma = base_sigma * vol_multiplier

    return stressed_mu, stressed_sigma


def simulate_forward_returns(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    weights: Dict[str, float],
    scenario: StressScenario,
    forward_days: int = 252,
    n_simulations: int = 1000,
    seed: Optional[int] = 42
) -> Dict:
    """
    Simulate forward-looking portfolio returns under a stress scenario.

    Args:
        returns: Historical returns DataFrame
        prices: Historical prices DataFrame
        weights: Portfolio weights
        scenario: Stress scenario to apply
        forward_days: Number of days to simulate forward
        n_simulations: Number of Monte Carlo simulations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    if seed is not None:
        np.random.seed(seed)

    tickers = returns.columns.tolist()
    n_assets = len(tickers)

    # Calculate historical parameters
    historical_mu = returns.mean().values * 252  # Annualized drift
    historical_sigma = returns.std().values * np.sqrt(252)  # Annualized volatility

    # Get last prices as starting points
    last_prices = prices.iloc[-1].values

    # Time parameters
    T = forward_days / 252.0  # Time in years
    dt = 1 / 252.0  # Daily steps

    # Store simulated portfolio values
    portfolio_simulations = np.zeros((n_simulations, forward_days + 1))

    # Calculate initial portfolio value
    weight_array = np.array([weights.get(t, 0) for t in tickers])
    initial_portfolio_value = 100.0
    portfolio_simulations[:, 0] = initial_portfolio_value

    # Simulate each asset and aggregate into portfolio
    for sim in range(n_simulations):
        # Set seed for this simulation
        sim_seed = seed + sim if seed is not None else None
        if sim_seed is not None:
            np.random.seed(sim_seed)

        # Simulate each asset
        asset_paths = np.zeros((n_assets, forward_days + 1))

        for i, ticker in enumerate(tickers):
            is_affected = ticker in scenario.affected_assets

            # Apply stress to parameters
            mu, sigma = apply_stress_to_parameters(
                historical_mu[i],
                historical_sigma[i],
                scenario,
                is_affected
            )

            # Generate path for this asset
            n_steps = forward_days
            dW = np.random.normal(0, np.sqrt(dt), n_steps)

            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * dW

            log_returns = drift + diffusion
            log_prices = np.cumsum(log_returns)

            asset_paths[i, 0] = last_prices[i]
            asset_paths[i, 1:] = last_prices[i] * np.exp(log_prices)

        # Calculate portfolio value over time
        for t in range(forward_days + 1):
            if t == 0:
                portfolio_simulations[sim, t] = initial_portfolio_value
            else:
                # Calculate returns from t-1 to t for each asset
                asset_returns = asset_paths[:, t] / asset_paths[:, t-1] - 1
                portfolio_return = np.sum(weight_array * asset_returns)
                portfolio_simulations[sim, t] = portfolio_simulations[sim, t-1] * (1 + portfolio_return)

    # Calculate summary statistics
    final_values = portfolio_simulations[:, -1]
    mean_final = np.mean(final_values)
    median_final = np.median(final_values)
    std_final = np.std(final_values)
    var_5 = np.percentile(final_values, 5)
    var_1 = np.percentile(final_values, 1)
    cvar_5 = np.mean(final_values[final_values <= var_5])

    # Calculate returns
    total_returns = (final_values / initial_portfolio_value - 1) * 100

    # Generate date range
    last_date = prices.index[-1]
    forward_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forward_days + 1,
        freq='B'
    )

    # Mean path
    mean_path = np.mean(portfolio_simulations, axis=0)
    percentile_5 = np.percentile(portfolio_simulations, 5, axis=0)
    percentile_95 = np.percentile(portfolio_simulations, 95, axis=0)

    return {
        'scenario_name': scenario.name,
        'affected_assets': scenario.affected_assets,
        'stress_parameters': {
            'expected_drop_pct': scenario.expected_drop_pct,
            'drop_std_pct': scenario.drop_std_pct,
            'volatility_increase_pct': scenario.volatility_increase_pct
        },
        'simulation_parameters': {
            'n_simulations': n_simulations,
            'forward_days': forward_days,
            'seed': seed
        },
        'dates': forward_dates.strftime('%Y-%m-%d').tolist(),
        'mean_path': mean_path.tolist(),
        'percentile_5_path': percentile_5.tolist(),
        'percentile_95_path': percentile_95.tolist(),
        'final_value_statistics': {
            'mean': mean_final,
            'median': median_final,
            'std': std_final,
            'var_5pct': var_5,
            'var_1pct': var_1,
            'cvar_5pct': cvar_5
        },
        'return_statistics': {
            'mean_return_pct': np.mean(total_returns),
            'median_return_pct': np.median(total_returns),
            'std_return_pct': np.std(total_returns),
            'min_return_pct': np.min(total_returns),
            'max_return_pct': np.max(total_returns),
            'var_5pct_return_pct': np.percentile(total_returns, 5),
            'var_1pct_return_pct': np.percentile(total_returns, 1)
        },
        'all_final_values': final_values.tolist()
    }


def simulate_multiple_scenarios(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    weights: Dict[str, float],
    scenarios: List[StressScenario],
    forward_days: int = 252,
    n_simulations: int = 1000,
    seed: Optional[int] = 42
) -> List[Dict]:
    """
    Run multiple stress scenarios.

    Args:
        returns: Historical returns DataFrame
        prices: Historical prices DataFrame
        weights: Portfolio weights
        scenarios: List of stress scenarios
        forward_days: Number of days to simulate
        n_simulations: Number of simulations per scenario
        seed: Random seed

    Returns:
        List of simulation results for each scenario
    """
    results = []

    for i, scenario in enumerate(scenarios):
        scenario_seed = seed + i * 10000 if seed is not None else None
        result = simulate_forward_returns(
            returns=returns,
            prices=prices,
            weights=weights,
            scenario=scenario,
            forward_days=forward_days,
            n_simulations=n_simulations,
            seed=scenario_seed
        )
        results.append(result)

    return results


def create_baseline_scenario() -> StressScenario:
    """Create a baseline (no stress) scenario."""
    return StressScenario(
        name="Baseline (No Stress)",
        affected_assets=[],
        expected_drop_pct=0.0,
        drop_std_pct=0.0,
        volatility_increase_pct=0.0
    )


if __name__ == "__main__":
    from data_loader import load_and_clean_data
    from portfolio_optimization import optimize_portfolio

    # Define tickers
    thai_export = ['DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK']
    thai_domestic = ['CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK']
    global_assets = ['WDC', 'THD']
    fixed_income = ['LEMB', 'VWOB', 'EMLC']
    fx = ['THB=X']

    all_tickers = thai_export + thai_domestic + global_assets + fixed_income + fx
    thai_assets = thai_export + thai_domestic + ['THD', 'THB=X']

    # Load data
    prices, returns, valid_tickers = load_and_clean_data(all_tickers, years=10)

    # Optimize portfolio
    opt_result = optimize_portfolio(returns, objective='sharpe')
    weights = opt_result['weights']

    # Update thai_assets to only include valid ones
    thai_assets = [t for t in thai_assets if t in valid_tickers]

    # Create stress scenario
    scenario = create_trump_tariff_scenario(
        thai_assets=thai_assets,
        expected_drop_pct=-10.0,
        drop_std_pct=2.0,
        volatility_increase_pct=10.0
    )

    # Run simulation
    result = simulate_forward_returns(
        returns=returns,
        prices=prices,
        weights=weights,
        scenario=scenario,
        forward_days=252,
        n_simulations=1000,
        seed=42
    )

    print(f"\nStress Scenario: {result['scenario_name']}")
    print(f"Affected Assets: {result['affected_assets']}")
    print(f"\nReturn Statistics:")
    for key, value in result['return_statistics'].items():
        print(f"  {key}: {value:.2f}%")
