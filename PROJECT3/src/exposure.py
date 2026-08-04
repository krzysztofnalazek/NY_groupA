import numpy as np
import pandas as pd


def calculate_exposures(trade_values, trade_details, netting_sets):
    first_trade = next(iter(trade_values.values()))
    gross_exposure = np.zeros_like(first_trade)
    netting_values = {}

    for netting_set in netting_sets["netting_set"]:
        netting_values[netting_set] = np.zeros_like(first_trade)

    for trade in trade_details.itertuples(index=False):
        values = trade_values[trade.trade_id]
        gross_exposure += np.maximum(values, 0)
        netting_values[trade.netting_set] += values

    netting_exposures = {}
    portfolio_exposure = np.zeros_like(first_trade)

    for netting_set, values in netting_values.items():
        exposure = np.maximum(values, 0)
        netting_exposures[netting_set] = exposure
        portfolio_exposure += exposure

    return {
        "gross": gross_exposure,
        "portfolio": portfolio_exposure,
        "netting_values": netting_values,
        "netting_exposures": netting_exposures,
    }


def make_exposure_profile(exposures, time_grid, pfe_levels):
    portfolio = exposures["portfolio"]
    profile = pd.DataFrame({"time_years": time_grid})
    profile["ee"] = portfolio.mean(axis=0)

    for level in pfe_levels:
        column = "pfe_" + str(int(level * 100))
        profile[column] = np.quantile(portfolio, level, axis=0)

    profile["gross_ee"] = exposures["gross"].mean(axis=0)
    profile["netting_benefit"] = profile["gross_ee"] - profile["ee"]
    return profile


def make_netting_set_profile(exposures, time_grid, pfe_levels):
    rows = []

    for netting_set, values in exposures["netting_exposures"].items():
        for time_number, time in enumerate(time_grid):
            row = {
                "netting_set": netting_set,
                "time_years": time,
                "ee": values[:, time_number].mean(),
            }
            for level in pfe_levels:
                column = "pfe_" + str(int(level * 100))
                row[column] = np.quantile(values[:, time_number], level)
            rows.append(row)

    return pd.DataFrame(rows)


def make_trade_summary(trade_values, trade_details):
    rows = []

    for trade in trade_details.itertuples(index=False):
        values = trade_values[trade.trade_id]
        positive_values = np.maximum(values, 0)
        rows.append(
            {
                "trade_id": trade.trade_id,
                "netting_set": trade.netting_set,
                "product": trade.product,
                "initial_mtm": values[0, 0],
                "max_standalone_ee": positive_values.mean(axis=0).max(),
                "max_absolute_mtm": np.abs(values).max(),
            }
        )

    return pd.DataFrame(rows)
