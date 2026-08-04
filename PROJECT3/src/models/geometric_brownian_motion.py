import numpy as np


def simulate_gbm(spot, volatility, dividend_yield, rates, time_grid, shocks):
    n_paths = shocks.shape[0]
    prices = np.zeros((n_paths, len(time_grid)))
    prices[:, 0] = spot

    for step in range(1, len(time_grid)):
        dt = time_grid[step] - time_grid[step - 1]
        drift = rates[:, step - 1] - dividend_yield - 0.5 * volatility ** 2
        diffusion = volatility * np.sqrt(dt) * shocks[:, step - 1]
        prices[:, step] = prices[:, step - 1] * np.exp(drift * dt + diffusion)

    return prices
