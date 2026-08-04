import numpy as np

from src.models.cir import simulate_cir


def scenario_correlations(credit, scenario, rate_factor):
    if scenario == "independent":
        equity_correlation = 0.0
        rate_correlation = 0.0
    elif scenario == "moderate":
        equity_correlation = credit.wwr_correlation
        rate_correlation = credit.wwr_rate_correlation
    elif scenario == "severe":
        equity_correlation = credit.severe_wwr_correlation
        rate_correlation = credit.severe_wwr_rate_correlation
    else:
        raise ValueError("Unknown WWR scenario: " + scenario)

    return {
        credit.wwr_market_factor: equity_correlation,
        rate_factor: rate_correlation,
    }


def make_credit_shocks(market_shocks, market_correlations,
                       target_correlations, independent_shocks):
    factors = list(target_correlations)
    targets = np.array([target_correlations[factor] for factor in factors])

    if np.allclose(targets, 0):
        return independent_shocks

    factor_correlations = market_correlations.loc[factors, factors].values
    loadings = np.linalg.solve(factor_correlations, targets)
    residual_variance = 1 - targets @ loadings

    if residual_variance < -1e-10:
        raise ValueError("The requested WWR correlations are not valid together.")

    systematic_shock = np.zeros_like(independent_shocks)
    for factor_number, factor in enumerate(factors):
        systematic_shock += loadings[factor_number] * market_shocks[factor]

    residual_variance = max(residual_variance, 0)
    return systematic_shock + np.sqrt(residual_variance) * independent_shocks


def simulate_credit_path(credit, time_grid, market_shocks,
                         market_correlations, independent_shocks,
                         target_correlations):
    credit_shocks = make_credit_shocks(
        market_shocks,
        market_correlations,
        target_correlations,
        independent_shocks,
    )

    return simulate_cir(
        credit.initial_hazard,
        credit.long_run_hazard,
        credit.mean_reversion,
        credit.credit_volatility,
        time_grid,
        credit_shocks,
    )


def survival_and_default_probabilities(hazard, time_grid):
    survival = np.ones_like(hazard)
    default_probability = np.zeros_like(hazard)

    for step in range(1, len(time_grid)):
        dt = time_grid[step] - time_grid[step - 1]
        average_hazard = 0.5 * (hazard[:, step - 1] + hazard[:, step])
        survival[:, step] = survival[:, step - 1] * np.exp(-average_hazard * dt)
        default_probability[:, step] = survival[:, step - 1] - survival[:, step]

    return survival, default_probability
