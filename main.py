"""
Main Module
Runs the complete portfolio optimization and stress testing pipeline.
Generates JSON output with all results.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

from data_loader import load_and_clean_data
from portfolio_optimization import optimize_portfolio, backtest_portfolio
from scenario_modeling import (
    create_trump_tariff_scenario,
    create_baseline_scenario,
    simulate_forward_returns,
    simulate_multiple_scenarios,
    StressScenario
)


# Default asset universe
DEFAULT_TICKERS = {
    'thai_export': ['DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK'],
    'thai_domestic': ['CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK'],
    'global': ['WDC', 'THD'],
    'fixed_income': ['LEMB', 'VWOB', 'EMLC'],
    'fx': ['THB=X']
}


def get_all_tickers(ticker_groups: Dict[str, List[str]]) -> List[str]:
    """Flatten ticker groups into a single list."""
    all_tickers = []
    for group in ticker_groups.values():
        all_tickers.extend(group)
    return all_tickers


def get_thai_related_tickers(ticker_groups: Dict[str, List[str]]) -> List[str]:
    """Get all Thai-related tickers including THD ETF and THB currency."""
    thai_tickers = []
    thai_tickers.extend(ticker_groups.get('thai_export', []))
    thai_tickers.extend(ticker_groups.get('thai_domestic', []))

    # Also include Thailand ETF and currency
    if 'THD' in ticker_groups.get('global', []):
        thai_tickers.append('THD')
    thai_tickers.extend(ticker_groups.get('fx', []))

    return thai_tickers


def run_pipeline(
    ticker_groups: Optional[Dict[str, List[str]]] = None,
    historical_years: int = 10,
    forward_days: int = 252,
    n_simulations: int = 1000,
    stress_drop_pct: float = -10.0,
    stress_drop_std_pct: float = 2.0,
    stress_vol_increase_pct: float = 10.0,
    risk_free_rate: float = 0.02,
    seed: int = 42,
    output_file: Optional[str] = None
) -> Dict:
    """
    Run the complete portfolio optimization and stress testing pipeline.

    Args:
        ticker_groups: Dictionary of ticker groups
        historical_years: Years of historical data to use
        forward_days: Number of days to simulate forward
        n_simulations: Number of Monte Carlo simulations
        stress_drop_pct: Expected percentage drop for stressed assets
        stress_drop_std_pct: Standard deviation of the drop
        stress_vol_increase_pct: Percentage increase in volatility
        risk_free_rate: Annual risk-free rate
        seed: Random seed for reproducibility
        output_file: Path to save JSON output (optional)

    Returns:
        Dictionary with all results
    """
    if ticker_groups is None:
        ticker_groups = DEFAULT_TICKERS

    all_tickers = get_all_tickers(ticker_groups)
    thai_related = get_thai_related_tickers(ticker_groups)

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)

    # Step 1: Load and clean data
    print("\n[1/4] Loading and cleaning data...")
    prices, returns, valid_tickers = load_and_clean_data(
        tickers=all_tickers,
        years=historical_years
    )

    # Update Thai-related tickers to only include valid ones
    thai_related_valid = [t for t in thai_related if t in valid_tickers]

    # Step 2: Portfolio optimization
    print("\n[2/4] Optimizing portfolio (Max Sharpe Ratio)...")
    optimization_result = optimize_portfolio(
        returns=returns,
        objective='sharpe',
        risk_free_rate=risk_free_rate
    )

    print(f"  Expected Return: {optimization_result['expected_return']*100:.2f}%")
    print(f"  Volatility: {optimization_result['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {optimization_result['sharpe_ratio']:.2f}")

    # Show significant weights
    print("\n  Significant weights (>1%):")
    for ticker, weight in sorted(
        optimization_result['weights'].items(),
        key=lambda x: -x[1]
    ):
        if weight > 0.01:
            print(f"    {ticker}: {weight*100:.2f}%")

    # Step 3: Historical backtest
    print("\n[3/4] Running historical backtest...")
    backtest_result = backtest_portfolio(
        weights=optimization_result['weights'],
        returns=returns,
        initial_value=100.0
    )

    print(f"  Total Return: {backtest_result['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {backtest_result['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {backtest_result['annualized_volatility_pct']:.2f}%")
    print(f"  Max Drawdown: {backtest_result['max_drawdown_pct']:.2f}%")

    # Step 4: Stress scenario simulation
    print("\n[4/4] Running stress scenario simulations...")

    # Create scenarios
    baseline_scenario = create_baseline_scenario()

    stress_scenario = create_trump_tariff_scenario(
        thai_assets=thai_related_valid,
        expected_drop_pct=stress_drop_pct,
        drop_std_pct=stress_drop_std_pct,
        volatility_increase_pct=stress_vol_increase_pct,
        duration_days=forward_days
    )

    # Run simulations
    scenarios = [baseline_scenario, stress_scenario]
    simulation_results = simulate_multiple_scenarios(
        returns=returns,
        prices=prices,
        weights=optimization_result['weights'],
        scenarios=scenarios,
        forward_days=forward_days,
        n_simulations=n_simulations,
        seed=seed
    )

    # Print simulation results
    for result in simulation_results:
        print(f"\n  Scenario: {result['scenario_name']}")
        print(f"    Mean Return: {result['return_statistics']['mean_return_pct']:.2f}%")
        print(f"    Std Dev: {result['return_statistics']['std_return_pct']:.2f}%")
        print(f"    VaR (5%): {result['return_statistics']['var_5pct_return_pct']:.2f}%")

    # Compile final output
    output = {
        'metadata': {
            'run_timestamp': datetime.now().isoformat(),
            'historical_years': historical_years,
            'forward_days': forward_days,
            'n_simulations': n_simulations,
            'risk_free_rate': risk_free_rate,
            'seed': seed,
            'valid_tickers': valid_tickers,
            'thai_related_tickers': thai_related_valid
        },
        'optimization': {
            'objective': 'maximize_sharpe_ratio',
            'constraints': {
                'no_short_selling': True,
                'weights_sum_to_one': True
            },
            'optimal_weights': optimization_result['weights'],
            'expected_annual_return': optimization_result['expected_return'],
            'expected_annual_volatility': optimization_result['volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio']
        },
        'historical_backtest': {
            'dates': backtest_result['dates'],
            'portfolio_values': backtest_result['portfolio_values'],
            'daily_returns': backtest_result['daily_returns'],
            'metrics': {
                'total_return_pct': backtest_result['total_return_pct'],
                'annualized_return_pct': backtest_result['annualized_return_pct'],
                'annualized_volatility_pct': backtest_result['annualized_volatility_pct'],
                'sharpe_ratio': backtest_result['sharpe_ratio'],
                'max_drawdown_pct': backtest_result['max_drawdown_pct'],
                'final_value': backtest_result['final_value']
            }
        },
        'stress_scenarios': []
    }

    # Add simulation results for each scenario
    for result in simulation_results:
        scenario_output = {
            'scenario_name': result['scenario_name'],
            'affected_assets': result['affected_assets'],
            'stress_parameters': result['stress_parameters'],
            'dates': result['dates'],
            'mean_path': result['mean_path'],
            'percentile_5_path': result['percentile_5_path'],
            'percentile_95_path': result['percentile_95_path'],
            'return_statistics': result['return_statistics'],
            'final_value_statistics': result['final_value_statistics'],
            'all_final_values': result['all_final_values']
        }
        output['stress_scenarios'].append(scenario_output)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return output


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Stress Testing'
    )

    parser.add_argument(
        '--historical-years', type=int, default=10,
        help='Years of historical data (default: 10)'
    )
    parser.add_argument(
        '--forward-days', type=int, default=252,
        help='Forward simulation days (default: 252, ~1 year)'
    )
    parser.add_argument(
        '--simulations', type=int, default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    parser.add_argument(
        '--stress-drop', type=float, default=-10.0,
        help='Expected drop %% for Thai assets (default: -10)'
    )
    parser.add_argument(
        '--stress-drop-std', type=float, default=2.0,
        help='Std dev of drop %% (default: 2)'
    )
    parser.add_argument(
        '--stress-vol-increase', type=float, default=10.0,
        help='Volatility increase %% for Thai assets (default: 10)'
    )
    parser.add_argument(
        '--risk-free-rate', type=float, default=0.02,
        help='Risk-free rate (default: 0.02)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output', type=str, default='portfolio_results.json',
        help='Output JSON file path (default: portfolio_results.json)'
    )

    args = parser.parse_args()

    result = run_pipeline(
        historical_years=args.historical_years,
        forward_days=args.forward_days,
        n_simulations=args.simulations,
        stress_drop_pct=args.stress_drop,
        stress_drop_std_pct=args.stress_drop_std,
        stress_vol_increase_pct=args.stress_vol_increase,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
        output_file=args.output
    )

    return result


if __name__ == "__main__":
    main()
