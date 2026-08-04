import numpy as np

from src.models.vasicek import zero_coupon_bond


def swap_value(rate, valuation_time, maturity, notional, fixed_rate,
               payments_per_year, position, rate_parameters):
    if valuation_time >= maturity:
        return np.zeros_like(rate)

    payment_interval = 1 / payments_per_year
    payment_dates = np.arange(
        payment_interval,
        maturity + payment_interval / 2,
        payment_interval,
    )
    payment_dates = payment_dates[payment_dates > valuation_time + 1e-10]

    discount_factors = []
    for payment_date in payment_dates:
        time_to_payment = payment_date - valuation_time
        discount_factor = zero_coupon_bond(
            rate,
            time_to_payment,
            rate_parameters["long_run_mean"],
            rate_parameters["mean_reversion"],
            rate_parameters["volatility"],
        )
        discount_factors.append(discount_factor)

    discount_factors = np.column_stack(discount_factors)
    annuity = payment_interval * discount_factors.sum(axis=1)
    floating_leg = 1 - discount_factors[:, -1]
    payer_value = notional * (floating_leg - fixed_rate * annuity)

    if position == "payer":
        return payer_value
    if position == "receiver":
        return -payer_value

    raise ValueError("Swap position must be payer or receiver.")
