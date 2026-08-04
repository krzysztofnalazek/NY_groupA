import numpy as np


def simulate_cir(initial_hazard, long_run_hazard, mean_reversion,
                 volatility, time_grid, shocks):
    n_paths = shocks.shape[0]
    hazard = np.zeros((n_paths, len(time_grid)))
    hazard[:, 0] = initial_hazard

    for step in range(1, len(time_grid)):
        dt = time_grid[step] - time_grid[step - 1]
        previous = np.maximum(hazard[:, step - 1], 0)
        drift = mean_reversion * (long_run_hazard - previous) * dt
        diffusion = volatility * np.sqrt(previous) * np.sqrt(dt)
        next_hazard = previous + drift + diffusion * shocks[:, step - 1]
        hazard[:, step] = np.maximum(next_hazard, 0)

    return hazard
