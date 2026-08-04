import numpy as np
from scipy.stats import norm


def option_price(spot, strike, time_to_maturity, volatility,
                 dividend_yield, discount_factor, option_type):
    spot = np.asarray(spot)

    if time_to_maturity <= 0:
        if option_type == "call":
            return np.maximum(spot - strike, 0)
        if option_type == "put":
            return np.maximum(strike - spot, 0)
        raise ValueError("Option type must be call or put.")

    root_time = np.sqrt(time_to_maturity)
    rate = -np.log(discount_factor) / time_to_maturity
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time

    discounted_spot = spot * np.exp(-dividend_yield * time_to_maturity)

    if option_type == "call":
        return discounted_spot * norm.cdf(d1) - strike * discount_factor * norm.cdf(d2)
    if option_type == "put":
        return strike * discount_factor * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)

    raise ValueError("Option type must be call or put.")
