import numpy as np
import pandas as pd

from src.models.vasicek import zero_coupon_bond
from src.pricing.black_scholes import option_price
from src.pricing.interest_rate_swap import swap_value


def value_portfolio(inputs, paths, time_grid):
    options = inputs["options"]
    swaps = inputs["swaps"]
    equity_parameters = inputs["equities"].set_index("risk_factor")
    rate_parameters = inputs["rates"].iloc[0]
    rate_paths = paths[rate_parameters["risk_factor"]]

    trade_values = {}

    for trade in options.itertuples(index=False):
        values = np.zeros_like(rate_paths)
        equity = equity_parameters.loc[trade.underlying]
        spot_paths = paths[trade.underlying]

        for time_number, time in enumerate(time_grid):
            time_left = trade.maturity_years - time
            if time_left <= 0:
                continue

            discount_factor = zero_coupon_bond(
                rate_paths[:, time_number],
                time_left,
                rate_parameters["long_run_mean"],
                rate_parameters["mean_reversion"],
                rate_parameters["volatility"],
            )
            price = option_price(
                spot_paths[:, time_number],
                trade.strike,
                time_left,
                equity["volatility"],
                equity["dividend_yield"],
                discount_factor,
                trade.option_type,
            )

            sign = 1 if trade.position == "long" else -1
            values[:, time_number] = sign * trade.quantity * price

        trade_values[trade.trade_id] = values

    for trade in swaps.itertuples(index=False):
        values = np.zeros_like(rate_paths)

        for time_number, time in enumerate(time_grid):
            values[:, time_number] = swap_value(
                rate_paths[:, time_number],
                time,
                trade.maturity_years,
                trade.notional,
                trade.fixed_rate,
                trade.payments_per_year,
                trade.position,
                rate_parameters,
            )

        trade_values[trade.trade_id] = values

    option_details = options[["trade_id", "netting_set"]].copy()
    option_details["product"] = "European option"
    swap_details = swaps[["trade_id", "netting_set"]].copy()
    swap_details["product"] = "Interest rate swap"
    trade_details = pd.concat([option_details, swap_details], ignore_index=True)

    return trade_values, trade_details
