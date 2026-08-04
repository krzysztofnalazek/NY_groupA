import numpy as np


def payoff(spot, strike, option_type):
    if option_type == "call":
        return np.maximum(spot - strike, 0)
    if option_type == "put":
        return np.maximum(strike - spot, 0)
    raise ValueError("Option type must be call or put.")


def polynomial_basis(x, degree):
    columns = [np.ones_like(x)]
    for power in range(1, degree + 1):
        columns.append(x ** power)
    return np.column_stack(columns)


def discount_between(rate_paths, time_grid, start_index, end_indices):
    discounts = np.ones(rate_paths.shape[0])

    for end_index in np.unique(end_indices):
        if end_index <= start_index:
            continue

        paths = end_indices == end_index
        year_steps = np.diff(time_grid[start_index:end_index + 1])
        short_rates = rate_paths[paths, start_index:end_index]
        discounts[paths] = np.exp(-np.sum(short_rates * year_steps, axis=1))

    return discounts


def make_exercise_indices(time_grid, maturity, exercise_dates_per_year):
    maturity_index = np.where(time_grid <= maturity + 1e-12)[0][-1]
    exercise_indices = [0, maturity_index]

    for time_index, time in enumerate(time_grid[:maturity_index + 1]):
        scaled_time = time * exercise_dates_per_year
        is_exercise_date = abs(scaled_time - round(scaled_time)) < 1e-12
        if is_exercise_date:
            exercise_indices.append(time_index)

    return set(exercise_indices), maturity_index


def american_option_value(
    spot_paths,
    rate_paths,
    time_grid,
    strike,
    maturity,
    option_type,
    exercise_dates_per_year,
    regression_degree=2,
):
    """
    Longstaff-Schwartz valuation for an American option on existing paths.

    This follows the reference repo's LSM idea: work backwards, regress
    discounted future cashflows on polynomial spot functions, then compare
    continuation value with immediate exercise value.
    """
    n_paths, n_dates = spot_paths.shape
    values = np.zeros((n_paths, n_dates))

    if maturity <= 0:
        return values

    exercise_indices, maturity_index = make_exercise_indices(
        time_grid, maturity, exercise_dates_per_year
    )
    cashflow = payoff(spot_paths[:, maturity_index], strike, option_type)
    cashflow_date = np.full(n_paths, maturity_index)
    first_exercise_date = np.full(n_paths, maturity_index)

    for time_index in range(maturity_index - 1, -1, -1):
        exercise_value = payoff(spot_paths[:, time_index], strike, option_type)
        discount = discount_between(rate_paths, time_grid, time_index, cashflow_date)
        discounted_cashflow = cashflow * discount

        continuation = np.zeros(n_paths)
        exercise_allowed = time_index in exercise_indices
        regression_paths = np.ones(n_paths, dtype=bool)
        if exercise_allowed:
            regression_paths = exercise_value > 0

        if regression_paths.sum() > regression_degree + 1:
            x = spot_paths[regression_paths, time_index] / strike
            y = discounted_cashflow[regression_paths]
            basis = polynomial_basis(x, regression_degree)
            coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]
            continuation = polynomial_basis(
                spot_paths[:, time_index] / strike, regression_degree
            ) @ coefficients
        else:
            continuation[:] = discounted_cashflow.mean()

        values[:, time_index] = continuation

        if exercise_allowed:
            values[:, time_index] = np.maximum(exercise_value, continuation)
            exercise_now = (exercise_value > 0) & (exercise_value > continuation)
            cashflow[exercise_now] = exercise_value[exercise_now]
            cashflow_date[exercise_now] = time_index
            first_exercise_date[exercise_now] = time_index

    for path_number in range(n_paths):
        exercise_date = first_exercise_date[path_number]
        values[path_number, exercise_date + 1:] = 0
        values[path_number, maturity_index] = 0

    return np.maximum(values, 0)
