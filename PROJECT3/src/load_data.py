import numpy as np
import pandas as pd

from config import DATA_DIR


def check_columns(data, required_columns, file_name):
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(file_name + " is missing columns: " + str(sorted(missing)))


def load_inputs():
    equities = pd.read_csv(DATA_DIR / "equity_parameters.csv")
    rates = pd.read_csv(DATA_DIR / "rate_parameters.csv")
    options = pd.read_csv(DATA_DIR / "equity_options.csv")
    swaps = pd.read_csv(DATA_DIR / "interest_rate_swaps.csv")
    netting_sets = pd.read_csv(DATA_DIR / "netting_sets.csv")
    credit = pd.read_csv(DATA_DIR / "counterparty_credit.csv")
    correlations = pd.read_csv(
        DATA_DIR / "correlations.csv", index_col="risk_factor"
    )

    check_columns(
        equities,
        ["risk_factor", "spot", "volatility", "dividend_yield"],
        "equity_parameters.csv",
    )
    check_columns(
        rates,
        [
            "risk_factor",
            "initial_rate",
            "long_run_mean",
            "mean_reversion",
            "volatility",
        ],
        "rate_parameters.csv",
    )
    check_columns(
        options,
        [
            "trade_id",
            "netting_set",
            "underlying",
            "option_type",
            "position",
            "quantity",
            "strike",
            "maturity_years",
            "exercise_style",
            "exercise_dates_per_year",
        ],
        "equity_options.csv",
    )
    check_columns(
        swaps,
        [
            "trade_id",
            "netting_set",
            "position",
            "notional",
            "fixed_rate",
            "maturity_years",
            "payments_per_year",
        ],
        "interest_rate_swaps.csv",
    )
    check_columns(
        credit,
        [
            "counterparty",
            "initial_hazard",
            "long_run_hazard",
            "mean_reversion",
            "credit_volatility",
            "recovery_rate",
            "wwr_market_factor",
            "wwr_correlation",
            "severe_wwr_correlation",
            "wwr_rate_correlation",
            "severe_wwr_rate_correlation",
        ],
        "counterparty_credit.csv",
    )

    trade_ids = pd.concat([options["trade_id"], swaps["trade_id"]])
    if trade_ids.duplicated().any():
        raise ValueError("Trade IDs must be unique.")

    if len(rates) != 1:
        raise ValueError("This project expects one USD short-rate model.")
    if not set(options["option_type"]).issubset({"call", "put"}):
        raise ValueError("Option type must be call or put.")
    if not set(options["position"]).issubset({"long", "short"}):
        raise ValueError("Option position must be long or short.")
    if not set(options["exercise_style"]).issubset({"european", "american"}):
        raise ValueError("Exercise style must be european or american.")
    if not set(swaps["position"]).issubset({"payer", "receiver"}):
        raise ValueError("Swap position must be payer or receiver.")
    if (options[["quantity", "strike", "maturity_years"]] <= 0).any().any():
        raise ValueError("Option quantities, strikes and maturities must be positive.")
    american_options = options["exercise_style"] == "american"
    if (options.loc[american_options, "exercise_dates_per_year"] <= 0).any():
        raise ValueError("American options need positive exercise_dates_per_year.")
    if (swaps[["notional", "maturity_years", "payments_per_year"]] <= 0).any().any():
        raise ValueError("Swap notionals, maturities and payment frequencies must be positive.")

    known_factors = set(equities["risk_factor"])
    unknown_factors = set(options["underlying"]) - known_factors
    if unknown_factors:
        raise ValueError("Unknown option underlyings: " + str(sorted(unknown_factors)))

    known_sets = set(netting_sets["netting_set"])
    used_sets = set(options["netting_set"]) | set(swaps["netting_set"])
    if used_sets - known_sets:
        raise ValueError("Unknown netting sets: " + str(sorted(used_sets - known_sets)))

    known_counterparties = set(netting_sets["counterparty"])
    if set(credit["counterparty"]) != known_counterparties:
        raise ValueError("Credit parameters must cover every counterparty once.")
    if credit["counterparty"].duplicated().any():
        raise ValueError("Counterparty credit rows must be unique.")
    if not set(credit["wwr_market_factor"]).issubset(known_factors):
        raise ValueError("A WWR market factor is missing from equity parameters.")
    if (credit["recovery_rate"] < 0).any() or (credit["recovery_rate"] >= 1).any():
        raise ValueError("Recovery rates must be between zero and one.")
    correlation_columns = [
        "wwr_correlation",
        "severe_wwr_correlation",
        "wwr_rate_correlation",
        "severe_wwr_rate_correlation",
    ]
    if (credit[correlation_columns].abs() > 1).any().any():
        raise ValueError("WWR correlations must be between -1 and 1.")

    credit = credit.merge(
        netting_sets, on="counterparty", how="left", validate="one_to_one"
    )

    factors = list(rates["risk_factor"]) + list(equities["risk_factor"])
    missing_factors = set(factors) - set(correlations.index)
    missing_factors |= set(factors) - set(correlations.columns)
    if missing_factors:
        raise ValueError("Missing correlations for: " + str(sorted(missing_factors)))
    correlations = correlations.loc[factors, factors].astype(float)
    if not np.allclose(correlations, correlations.T):
        raise ValueError("The correlation matrix must be symmetric.")
    if np.linalg.eigvalsh(correlations).min() <= 0:
        raise ValueError("The correlation matrix must be positive definite.")

    return {
        "equities": equities,
        "rates": rates,
        "options": options,
        "swaps": swaps,
        "netting_sets": netting_sets,
        "credit": credit,
        "correlations": correlations,
    }
