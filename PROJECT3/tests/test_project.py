import unittest

import numpy as np
import pandas as pd

from src.engine import run
from src.credit import make_credit_shocks, survival_and_default_probabilities
from src.exposure import calculate_exposures
from src.models.cir import simulate_cir
from src.models.vasicek import simulate_vasicek, zero_coupon_bond
from src.pricing.black_scholes import option_price
from src.pricing.least_squares_monte_carlo import american_option_value
from src.wwr import weighted_quantile


class ModelTests(unittest.TestCase):
    def test_black_scholes_put_call_parity(self):
        spot = np.array([100.0])
        strike = 100.0
        maturity = 2.0
        dividend_yield = 0.01
        discount_factor = np.array([np.exp(-0.04 * maturity)])

        call = option_price(
            spot, strike, maturity, 0.20, dividend_yield,
            discount_factor, "call"
        )
        put = option_price(
            spot, strike, maturity, 0.20, dividend_yield,
            discount_factor, "put"
        )

        expected = spot * np.exp(-dividend_yield * maturity)
        expected -= strike * discount_factor
        self.assertAlmostEqual(call[0] - put[0], expected[0], places=10)

    def test_zero_coupon_bond_at_maturity(self):
        rates = np.array([0.02, 0.04, 0.06])
        prices = zero_coupon_bond(rates, 0, 0.035, 0.60, 0.012)
        np.testing.assert_array_equal(prices, np.ones(3))

    def test_vasicek_zero_shock_follows_the_conditional_mean(self):
        time_grid = np.array([0.0, 1.0])
        shocks = np.zeros((1, 1))
        paths = simulate_vasicek(0.04, 0.03, 0.50, 0.01, time_grid, shocks)
        expected = 0.03 + (0.04 - 0.03) * np.exp(-0.50)
        self.assertAlmostEqual(paths[0, 1], expected)

    def test_lsm_american_put_can_exercise_early(self):
        time_grid = np.linspace(0, 1, 5)
        spot_paths = np.full((20, 5), 50.0)
        rate_paths = np.full((20, 5), 0.10)

        values = american_option_value(
            spot_paths,
            rate_paths,
            time_grid,
            strike=100.0,
            maturity=1.0,
            option_type="put",
            exercise_dates_per_year=4,
        )

        self.assertAlmostEqual(values[0, 0], 50.0)
        self.assertEqual(values[0, -1], 0.0)


class ExposureTests(unittest.TestCase):
    def test_trades_are_netted_before_the_zero_floor(self):
        trade_values = {
            "LONG": np.array([[10.0]]),
            "SHORT": np.array([[-8.0]]),
        }
        trade_details = pd.DataFrame(
            {
                "trade_id": ["LONG", "SHORT"],
                "netting_set": ["NS1", "NS1"],
                "product": ["Test", "Test"],
            }
        )
        netting_sets = pd.DataFrame({"netting_set": ["NS1"]})

        exposures = calculate_exposures(
            trade_values, trade_details, netting_sets
        )

        self.assertEqual(exposures["gross"][0, 0], 10.0)
        self.assertEqual(exposures["portfolio"][0, 0], 2.0)

    def test_small_end_to_end_run(self):
        results = run(
            n_paths=500,
            seed=7,
            save_outputs=False,
            make_charts=False,
        )
        profile = results["outputs"]["exposure_profile"]

        self.assertTrue(np.isfinite(profile.select_dtypes("number")).all().all())
        self.assertTrue((profile["ee"] >= 0).all())
        self.assertEqual(profile["ee"].iloc[-1], 0)

        summary = results["outputs"]["wwr_summary"]
        bank_b = summary[summary["counterparty"] == "Bank B"]
        independent = bank_b[bank_b["scenario"] == "independent"]["cva"].iloc[0]
        severe = bank_b[bank_b["scenario"] == "severe"]["cva"].iloc[0]
        self.assertGreater(severe, independent)


class WrongWayRiskTests(unittest.TestCase):
    def test_cir_hazard_rates_are_non_negative(self):
        time_grid = np.linspace(0, 2, 9)
        shocks = np.full((3, 8), -10.0)
        hazard = simulate_cir(0.02, 0.025, 0.7, 0.08, time_grid, shocks)
        self.assertTrue((hazard >= 0).all())

    def test_survival_falls_and_default_probabilities_are_positive(self):
        time_grid = np.array([0.0, 0.5, 1.0])
        hazard = np.full((2, 3), 0.02)
        survival, default_probability = survival_and_default_probabilities(
            hazard, time_grid
        )
        self.assertTrue((np.diff(survival, axis=1) <= 0).all())
        self.assertTrue((default_probability >= 0).all())

    def test_credit_shocks_match_requested_correlations(self):
        rng = np.random.default_rng(10)
        n_paths = 50000
        factor_a = rng.standard_normal((n_paths, 1))
        factor_b = rng.standard_normal((n_paths, 1))
        independent = rng.standard_normal((n_paths, 1))
        market_shocks = {"A": factor_a, "B": factor_b}
        market_correlations = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"]
        )
        targets = {"A": -0.4, "B": -0.2}

        credit_shocks = make_credit_shocks(
            market_shocks, market_correlations, targets, independent
        )
        correlation_a = np.corrcoef(credit_shocks[:, 0], factor_a[:, 0])[0, 1]
        correlation_b = np.corrcoef(credit_shocks[:, 0], factor_b[:, 0])[0, 1]

        self.assertAlmostEqual(correlation_a, -0.4, delta=0.02)
        self.assertAlmostEqual(correlation_b, -0.2, delta=0.02)

    def test_default_weighted_quantile(self):
        values = np.array([0.0, 10.0, 20.0])
        weights = np.array([0.1, 0.2, 0.7])
        self.assertEqual(weighted_quantile(values, weights, 0.5), 20.0)


if __name__ == "__main__":
    unittest.main()
