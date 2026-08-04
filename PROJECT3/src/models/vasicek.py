import numpy as np


def simulate_vasicek(r0, long_run_mean, mean_reversion, volatility,
                      time_grid, shocks):
    n_paths = shocks.shape[0]
    rates = np.zeros((n_paths, len(time_grid)))
    rates[:, 0] = r0

    for step in range(1, len(time_grid)):
        dt = time_grid[step] - time_grid[step - 1]
        decay = np.exp(-mean_reversion * dt)
        mean = long_run_mean + (rates[:, step - 1] - long_run_mean) * decay
        variance = volatility ** 2 * (1 - decay ** 2) / (2 * mean_reversion)
        rates[:, step] = mean + np.sqrt(variance) * shocks[:, step - 1]

    return rates


def zero_coupon_bond(rate, time_to_maturity, long_run_mean,
                     mean_reversion, volatility):
    if time_to_maturity <= 0:
        return np.ones_like(rate)

    kappa = mean_reversion
    sigma = volatility
    b = (1 - np.exp(-kappa * time_to_maturity)) / kappa
    a = (long_run_mean - sigma ** 2 / (2 * kappa ** 2))
    a = a * (b - time_to_maturity) - sigma ** 2 * b ** 2 / (4 * kappa)

    return np.exp(a - b * rate)
