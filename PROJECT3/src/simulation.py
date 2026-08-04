import numpy as np

from src.models.geometric_brownian_motion import simulate_gbm
from src.models.vasicek import simulate_vasicek


def make_time_grid(options, swaps, steps_per_year):
    max_option_maturity = options["maturity_years"].max()
    max_swap_maturity = swaps["maturity_years"].max()
    max_maturity = max(max_option_maturity, max_swap_maturity)
    n_steps = int(round(max_maturity * steps_per_year))
    return np.linspace(0, max_maturity, n_steps + 1)


def simulate_market(inputs, time_grid, n_paths, seed):
    rates_data = inputs["rates"].iloc[0]
    equities = inputs["equities"]
    correlations = inputs["correlations"]

    rng = np.random.default_rng(seed)
    n_steps = len(time_grid) - 1
    n_factors = len(correlations)
    independent_shocks = rng.standard_normal((n_paths, n_steps, n_factors))
    cholesky = np.linalg.cholesky(correlations.values)
    shocks = independent_shocks @ cholesky.T

    market_shocks = {}
    for factor_number, factor in enumerate(correlations.columns):
        market_shocks[factor] = shocks[:, :, factor_number]

    rate_factor = rates_data["risk_factor"]

    rate_paths = simulate_vasicek(
        rates_data["initial_rate"],
        rates_data["long_run_mean"],
        rates_data["mean_reversion"],
        rates_data["volatility"],
        time_grid,
        market_shocks[rate_factor],
    )

    paths = {rate_factor: rate_paths}

    for equity in equities.itertuples(index=False):
        factor_shocks = market_shocks[equity.risk_factor]
        paths[equity.risk_factor] = simulate_gbm(
            equity.spot,
            equity.volatility,
            equity.dividend_yield,
            rate_paths,
            time_grid,
            factor_shocks,
        )

    return paths, market_shocks
