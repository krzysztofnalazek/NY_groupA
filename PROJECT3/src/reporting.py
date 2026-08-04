import numpy as np
import pandas as pd


def make_portfolio_table(inputs):
    options = inputs["options"].copy()
    options.insert(2, "product", "European option")

    swaps = inputs["swaps"].copy()
    swaps.insert(2, "product", "Interest rate swap")

    return pd.concat([options, swaps], ignore_index=True, sort=False)


def make_paths_table(paths, time_grid, number_of_paths):
    rows = []

    for risk_factor, values in paths.items():
        paths_to_use = min(number_of_paths, values.shape[0])
        for path_number in range(paths_to_use):
            for time_number, time in enumerate(time_grid):
                rows.append(
                    {
                        "risk_factor": risk_factor,
                        "path": path_number,
                        "time_years": time,
                        "value": values[path_number, time_number],
                    }
                )

    return pd.DataFrame(rows)


def make_distribution_table(portfolio_exposure, time_grid):
    date_numbers = [
        len(time_grid) // 4,
        len(time_grid) // 2,
        3 * len(time_grid) // 4,
    ]
    rows = []

    for date_number in date_numbers:
        for path_number in range(portfolio_exposure.shape[0]):
            rows.append(
                {
                    "time_years": time_grid[date_number],
                    "path": path_number,
                    "exposure": portfolio_exposure[path_number, date_number],
                }
            )

    return pd.DataFrame(rows)


def make_netting_summary(netting_profile, netting_sets):
    summary = netting_profile.drop(columns="time_years")
    summary = summary.groupby("netting_set", as_index=False).max()

    new_names = {"ee": "max_ee"}
    for column in summary.columns:
        if column.startswith("pfe_"):
            new_names[column] = "max_" + column
    summary = summary.rename(columns=new_names)

    return summary.merge(netting_sets, on="netting_set", how="left")


def make_model_table(inputs):
    equity_models = inputs["equities"].copy()
    equity_models.insert(1, "model", "GBM")

    rate_models = inputs["rates"].copy()
    rate_models.insert(1, "model", "Vasicek")

    return pd.concat([equity_models, rate_models], ignore_index=True, sort=False)
