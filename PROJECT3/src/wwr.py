import numpy as np
import pandas as pd

from config import (
    WWR_CORRELATION_SWEEP,
    WWR_FOCUS_COUNTERPARTY,
    WWR_PATHS_TO_SAVE,
    WWR_PFE_LEVEL,
    WWR_SCATTER_PATHS,
    WWR_SCENARIOS,
)
from src.credit import (
    scenario_correlations,
    simulate_credit_path,
    survival_and_default_probabilities,
)


def make_discount_factors(rate_paths, time_grid):
    discount_factors = np.ones_like(rate_paths)

    for step in range(1, len(time_grid)):
        dt = time_grid[step] - time_grid[step - 1]
        average_rate = 0.5 * (rate_paths[:, step - 1] + rate_paths[:, step])
        discount_factors[:, step] = discount_factors[:, step - 1]
        discount_factors[:, step] *= np.exp(-average_rate * dt)

    return discount_factors


def weighted_quantile(values, weights, level):
    if weights.sum() <= 0:
        return np.quantile(values, level)

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_weights = np.cumsum(sorted_weights)
    cutoff = level * cumulative_weights[-1]
    position = np.searchsorted(cumulative_weights, cutoff)
    return sorted_values[min(position, len(sorted_values) - 1)]


def calculate_credit_metrics(exposure, hazard, default_probability,
                             discount_factors, recovery_rate,
                             time_grid, pfe_level):
    rows = []

    for time_number, time in enumerate(time_grid):
        values = exposure[:, time_number]
        weights = default_probability[:, time_number]

        if weights.sum() > 0:
            default_weighted_ee = np.average(values, weights=weights)
        else:
            default_weighted_ee = values.mean()

        rows.append(
            {
                "time_years": time,
                "ee": values.mean(),
                "pfe_95": np.quantile(values, pfe_level),
                "default_weighted_ee": default_weighted_ee,
                "default_weighted_pfe_95": weighted_quantile(
                    values, weights, pfe_level
                ),
                "mean_hazard": hazard[:, time_number].mean(),
                "mean_default_probability": weights.mean(),
            }
        )

    discounted_loss = discount_factors * exposure * default_probability
    cva = (1 - recovery_rate) * discounted_loss.mean(axis=0).sum()
    return pd.DataFrame(rows), cva


def run_wwr_analysis(inputs, paths, market_shocks, exposures,
                     time_grid, seed):
    credit_data = inputs["credit"]
    market_correlations = inputs["correlations"]
    rate_factor = inputs["rates"].iloc[0]["risk_factor"]
    rate_paths = paths[rate_factor]
    discount_factors = make_discount_factors(rate_paths, time_grid)

    rng = np.random.default_rng(seed + 1000)
    n_paths = rate_paths.shape[0]
    n_steps = len(time_grid) - 1
    independent_credit_shocks = {}

    for credit in credit_data.itertuples(index=False):
        independent_credit_shocks[credit.counterparty] = rng.standard_normal(
            (n_paths, n_steps)
        )

    profile_frames = []
    summary_rows = []
    intensity_rows = []
    scenario_results = {}

    for scenario in WWR_SCENARIOS:
        for credit in credit_data.itertuples(index=False):
            correlations = scenario_correlations(credit, scenario, rate_factor)
            hazard = simulate_credit_path(
                credit,
                time_grid,
                market_shocks,
                market_correlations,
                independent_credit_shocks[credit.counterparty],
                correlations,
            )
            survival, default_probability = survival_and_default_probabilities(
                hazard, time_grid
            )
            exposure = exposures["netting_exposures"][credit.netting_set]
            profile, cva = calculate_credit_metrics(
                exposure,
                hazard,
                default_probability,
                discount_factors,
                credit.recovery_rate,
                time_grid,
                WWR_PFE_LEVEL,
            )
            profile.insert(0, "netting_set", credit.netting_set)
            profile.insert(0, "counterparty", credit.counterparty)
            profile.insert(0, "scenario", scenario)
            profile_frames.append(profile)

            summary_rows.append(
                {
                    "scenario": scenario,
                    "counterparty": credit.counterparty,
                    "netting_set": credit.netting_set,
                    "cva": cva,
                    "recovery_rate": credit.recovery_rate,
                    "market_factor": credit.wwr_market_factor,
                    "market_correlation": correlations[credit.wwr_market_factor],
                    "rate_correlation": correlations[rate_factor],
                }
            )

            for path_number in range(min(WWR_PATHS_TO_SAVE, n_paths)):
                for time_number, time in enumerate(time_grid):
                    intensity_rows.append(
                        {
                            "scenario": scenario,
                            "counterparty": credit.counterparty,
                            "path": path_number,
                            "time_years": time,
                            "hazard_rate": hazard[path_number, time_number],
                            "survival_probability": survival[path_number, time_number],
                        }
                    )

            scenario_results[(scenario, credit.counterparty)] = {
                "hazard": hazard,
                "survival": survival,
                "default_probability": default_probability,
                "exposure": exposure,
            }

    wwr_profile = pd.concat(profile_frames, ignore_index=True)
    wwr_summary = pd.DataFrame(summary_rows)
    independent_cva = wwr_summary[wwr_summary["scenario"] == "independent"]
    independent_cva = independent_cva.set_index("counterparty")["cva"]
    wwr_summary["independent_cva"] = wwr_summary["counterparty"].map(independent_cva)
    wwr_summary["cva_uplift"] = wwr_summary["cva"] - wwr_summary["independent_cva"]
    wwr_summary["cva_uplift_pct"] = 100 * wwr_summary["cva_uplift"]
    wwr_summary["cva_uplift_pct"] /= wwr_summary["independent_cva"]

    sweep_rows = make_correlation_sweep(
        credit_data,
        market_shocks,
        market_correlations,
        independent_credit_shocks,
        exposures,
        discount_factors,
        time_grid,
        rate_factor,
    )
    correlation_sweep = pd.DataFrame(sweep_rows)

    scatter_sample, distribution_sample = make_focus_samples(
        wwr_profile, scenario_results, WWR_FOCUS_COUNTERPARTY
    )

    outputs = {
        "wwr_profile": wwr_profile,
        "wwr_summary": wwr_summary,
        "wwr_correlation_sweep": correlation_sweep,
        "credit_intensity_paths": pd.DataFrame(intensity_rows),
        "wwr_scatter_sample": scatter_sample,
        "wwr_distribution_sample": distribution_sample,
    }
    return outputs, scenario_results


def make_correlation_sweep(credit_data, market_shocks, market_correlations,
                           independent_credit_shocks, exposures,
                           discount_factors, time_grid, rate_factor):
    rows = []

    for credit in credit_data.itertuples(index=False):
        counterparty_results = []
        rate_ratio = credit.wwr_rate_correlation / credit.wwr_correlation

        for correlation in WWR_CORRELATION_SWEEP:
            target_correlations = {
                credit.wwr_market_factor: correlation,
                rate_factor: correlation * rate_ratio,
            }
            hazard = simulate_credit_path(
                credit,
                time_grid,
                market_shocks,
                market_correlations,
                independent_credit_shocks[credit.counterparty],
                target_correlations,
            )
            _, default_probability = survival_and_default_probabilities(
                hazard, time_grid
            )
            exposure = exposures["netting_exposures"][credit.netting_set]
            _, cva = calculate_credit_metrics(
                exposure,
                hazard,
                default_probability,
                discount_factors,
                credit.recovery_rate,
                time_grid,
                WWR_PFE_LEVEL,
            )
            counterparty_results.append(
                {
                    "counterparty": credit.counterparty,
                    "netting_set": credit.netting_set,
                    "correlation": correlation,
                    "rate_correlation": correlation * rate_ratio,
                    "cva": cva,
                }
            )

        base_cva = next(
            row["cva"] for row in counterparty_results
            if np.isclose(row["correlation"], 0)
        )
        for row in counterparty_results:
            row["cva_uplift_pct"] = 100 * (row["cva"] / base_cva - 1)
            rows.append(row)

    return rows


def make_focus_samples(wwr_profile, scenario_results, focus_counterparty):
    focus_profile = wwr_profile[
        (wwr_profile["scenario"] == "moderate")
        & (wwr_profile["counterparty"] == focus_counterparty)
    ]
    peak_row = focus_profile.loc[focus_profile["default_weighted_ee"].idxmax()]
    time_number = focus_profile.index.get_loc(peak_row.name)

    result = scenario_results[("moderate", focus_counterparty)]
    n_paths = min(WWR_SCATTER_PATHS, result["exposure"].shape[0])
    exposure = result["exposure"][:n_paths, time_number]
    hazard = result["hazard"][:n_paths, time_number]
    default_weight = result["default_probability"][:n_paths, time_number]

    scatter_sample = pd.DataFrame(
        {
            "counterparty": focus_counterparty,
            "time_years": peak_row["time_years"],
            "path": np.arange(n_paths),
            "exposure": exposure,
            "hazard_rate": hazard,
            "default_weight": default_weight,
        }
    )

    distribution_sample = scatter_sample[
        ["counterparty", "time_years", "path", "exposure", "default_weight"]
    ].copy()
    distribution_sample["ordinary_weight"] = 1 / n_paths
    total_weight = distribution_sample["default_weight"].sum()
    distribution_sample["default_weight"] /= total_weight

    return scatter_sample, distribution_sample
