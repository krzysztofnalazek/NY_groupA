from config import (
    N_PATHS,
    OUTPUT_DIR,
    PATHS_TO_SAVE,
    PFE_LEVELS,
    SEED,
    STEPS_PER_YEAR,
)
from src.charts import make_all_charts
from src.exposure import (
    calculate_exposures,
    make_exposure_profile,
    make_netting_set_profile,
    make_trade_summary,
)
from src.load_data import load_inputs
from src.reporting import (
    make_distribution_table,
    make_model_table,
    make_netting_summary,
    make_paths_table,
    make_portfolio_table,
)
from src.simulation import make_time_grid, simulate_market
from src.valuation import value_portfolio
from src.wwr import run_wwr_analysis


def run(n_paths=N_PATHS, seed=SEED, save_outputs=True, make_charts=True):
    inputs = load_inputs()
    time_grid = make_time_grid(
        inputs["options"], inputs["swaps"], STEPS_PER_YEAR
    )
    paths, market_shocks = simulate_market(inputs, time_grid, n_paths, seed)
    trade_values, trade_details = value_portfolio(inputs, paths, time_grid)
    exposures = calculate_exposures(
        trade_values, trade_details, inputs["netting_sets"]
    )

    exposure_profile = make_exposure_profile(
        exposures, time_grid, PFE_LEVELS
    )
    netting_set_profile = make_netting_set_profile(
        exposures, time_grid, PFE_LEVELS
    )

    outputs = {
        "portfolio": make_portfolio_table(inputs),
        "model_parameters": make_model_table(inputs),
        "correlations": inputs["correlations"].reset_index(),
        "risk_factor_paths": make_paths_table(paths, time_grid, PATHS_TO_SAVE),
        "exposure_profile": exposure_profile,
        "exposure_distributions": make_distribution_table(
            exposures["portfolio"], time_grid
        ),
        "netting_set_profile": netting_set_profile,
        "netting_set_summary": make_netting_summary(
            netting_set_profile, inputs["netting_sets"]
        ),
        "trade_summary": make_trade_summary(trade_values, trade_details),
    }

    wwr_outputs, credit_results = run_wwr_analysis(
        inputs,
        paths,
        market_shocks,
        exposures,
        time_grid,
        seed,
    )
    outputs.update(wwr_outputs)
    outputs["counterparty_credit"] = inputs["credit"]

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            data.to_csv(OUTPUT_DIR / (name + ".csv"), index=False)

    if make_charts:
        make_all_charts(paths, time_grid, outputs, inputs["correlations"])

    results = {
        "inputs": inputs,
        "time_grid": time_grid,
        "paths": paths,
        "market_shocks": market_shocks,
        "trade_values": trade_values,
        "exposures": exposures,
        "credit_results": credit_results,
        "outputs": outputs,
    }
    return results
