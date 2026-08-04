# PROJECT3 - Synthetic PFE

This is a clean teaching implementation of Potential Future Exposure (PFE).
It follows the main ideas in the local reference repository at
`../REPO/montecarlo-risk-engine`, while using the simpler NumPy/Pandas style of
the course notebooks.

The project is standalone. The reference repository is useful for comparison,
but it is not required to run `PROJECT3` or submit the folder.

## What the engine does

1. Reads a synthetic USD portfolio and model assumptions from CSV files.
2. Simulates correlated equity indexes with geometric Brownian motion.
3. Simulates the USD short rate with the exact Vasicek transition.
4. Revalues European options with Black-Scholes.
5. Revalues fixed-for-floating swaps using Vasicek zero-coupon bond prices.
6. Nets signed trade values inside each netting set.
7. Applies the positive exposure floor after netting.
8. Calculates EE, 95% PFE and 99% PFE at each future date.
9. Simulates counterparty hazard rates with a CIR model.
10. Compares independent, moderate and severe Wrong-Way Risk scenarios.
11. Calculates default-weighted EE/PFE and CVA.

## Run the project

From the main `Python for Quants` folder:

```powershell
.\.venv\Scripts\python.exe PROJECT3\run_project.py
```

Or, after installing `PROJECT3/requirements.txt`, run:

```powershell
cd PROJECT3
python run_project.py
```

CSV results are written to `outputs/` and charts are written to `charts/`.

## Folder structure

```text
PROJECT3/
|-- config.py
|-- run_project.py
|-- data/
|   |-- equity_parameters.csv
|   |-- rate_parameters.csv
|   |-- correlations.csv
|   |-- equity_options.csv
|   |-- interest_rate_swaps.csv
|   |-- counterparty_credit.csv
|   `-- netting_sets.csv
|-- src/
|   |-- models/
|   |   |-- geometric_brownian_motion.py
|   |   |-- vasicek.py
|   |   `-- cir.py
|   |-- pricing/
|   |   |-- black_scholes.py
|   |   `-- interest_rate_swap.py
|   |-- simulation.py
|   |-- valuation.py
|   |-- exposure.py
|   |-- credit.py
|   |-- wwr.py
|   |-- reporting.py
|   |-- charts.py
|   `-- engine.py
|-- notebooks/
|-- tests/
|-- outputs/
`-- charts/
```

## Inputs and hard-coding

Portfolio terms, spots, volatilities, dividends, Vasicek parameters and
correlations all live in `data/`. Counterparty hazard, recovery and WWR
correlations are stored in `data/counterparty_credit.csv`. Simulation controls
such as the number of paths, random seed and PFE confidence levels are grouped
in `config.py`.

## Wrong-Way Risk extension

The market paths are simulated once. Those same paths are reused for all three
credit scenarios:

- `independent` uses zero market-credit correlation.
- `moderate` uses the correlations in `wwr_correlation` and
  `wwr_rate_correlation`.
- `severe` uses the correlations in `severe_wwr_correlation` and
  `severe_wwr_rate_correlation`.

Each counterparty hazard rate follows a CIR process. Survival and interval
default probabilities are calculated from the integrated hazard rate.

```text
survival(t) = exp(-integrated hazard rate)
default probability(t) = survival(t-1) - survival(t)
```

The WWR exposure measures give greater weight to paths where default is more
likely:

```text
default-weighted EE = sum(exposure * default probability)
                      / sum(default probability)

CVA = (1 - recovery) *
      sum(expected discounted exposure * default probability)
```

Ordinary EE and PFE do not change between the scenarios because the marginal
market paths are unchanged. The effect appears in default-weighted exposure and
CVA, which depend on the joint market-credit distribution.

The correlation sweep from `-0.8` to `0.8` shows both wrong-way and right-way
risk. The Bank B scenario uses negative NASDAQ and USD-rate correlations because
its receiver swap and long put tend to gain value as those market factors fall.

The numbers inside the model functions are mathematical parts of the published
formulas. There are no instrument-name checks, arbitrary time-value additions,
or product multipliers.

## Exposure convention

For each path and date, the engine calculates:

```text
netting set MTM = sum of signed trade MTMs in the netting set
netting set exposure = max(netting set MTM, 0)
portfolio exposure = sum of netting set exposures
EE = mean exposure across paths
PFE = exposure quantile across paths
```

This is different from flooring every trade at zero before adding trades. That
approach reports gross exposure and loses the benefit of legally enforceable
netting.

## Scope and limitations

- All trades are in USD, so no FX conversion is hidden in the aggregation.
- The portfolio contains only European options and vanilla interest-rate swaps.
- Exposure dates and swap payment dates are quarterly and aligned.
- The option revaluation uses a pathwise Vasicek discount factor in the
  Black-Scholes formula. This is a transparent approximation, not a full joint
  stochastic-rate option model.
- Collateral, margin period of risk, credit migration and Monte Carlo confidence
  intervals are outside the current scope.
- The CIR process is synthetic and is not calibrated to market credit spreads.
- WWR uses market-credit correlation and does not include jump-to-default or
  specific WWR from a trade referencing the counterparty itself.
- Trades are set to zero at maturity because the final cashflow is treated as
  settled on that date.

These boundaries are deliberate: the project uses fewer products, but every
implemented product has a recognizable model and pricing formula.

## Tests

```powershell
cd PROJECT3
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover Black-Scholes put-call parity, Vasicek and CIR behaviour,
market-credit correlations, survival probabilities, weighted quantiles,
netting order, and a small end-to-end simulation.

## Audit 4/08

This audit reviewed the source code, inputs, notebooks, tests and existing
outputs. The engine and notebooks were not rerun during the audit because they
write output files.

### High-severity findings

1. **CVA is illustrative, not market or regulatory CVA.** The implementation in
   `src/wwr.py` is a reasonable unilateral CVA approximation, but the Vasicek,
   CIR, volatility and hazard parameters are synthetic and uncalibrated. Their
   pricing measure is not explicitly documented. The results should therefore
   be described as synthetic CVA estimates rather than production or regulatory
   CVA. The [Basel CVA framework](https://www.bis.org/basel_framework/chapter/MAR/50.htm)
   requires market-implied default probabilities and simulated discounted
   exposure under consistent assumptions.

2. **Swap valuation is only valid on floating-rate reset dates.**
   `src/pricing/interest_rate_swap.py` uses `1 - P(t,T)` for the floating leg.
   Between reset dates this omits the already-fixed next floating coupon and
   accrued interest. The current quarterly swaps align with the quarterly
   simulation grid, but `src/load_data.py` does not enforce that alignment.

3. **The correlation sweep can divide by zero.** `src/wwr.py` calculates
   `wwr_rate_correlation / wwr_correlation`. A valid counterparty configuration
   with zero equity correlation can fail or produce an infinite value, while
   the input validation currently allows zero.

### Medium-severity findings

4. **Default-weighted PFE is a custom diagnostic.** The weighted quantile in
   `src/wwr.py` is useful for demonstrating WWR, but it is not standard PFE,
   Effective EPE or a Basel-defined measure. It should be presented as a bespoke
   conditional or default-weighted exposure quantile.

5. **Option pricing is approximate under stochastic interest rates.** A
   pathwise Vasicek discount factor is passed into the Black-Scholes formula.
   This does not capture all joint equity-rate covariance effects inside the
   option price. It is suitable as a documented teaching approximation, but it
   is not exact hybrid-model pricing.

6. **WWR correlations are shock correlations.** `src/credit.py` correlates
   credit Brownian shocks with selected market shocks. The resulting
   correlations between exposure, hazard levels and default probabilities will
   differ because the CIR model and portfolio valuation are nonlinear. These
   realised correlations are not currently reported.

7. **Input validation is incomplete.** The loader does not fully validate
   positive model parameters, missing values, duplicate factors, swap/grid
   alignment, joint WWR feasibility, focus-counterparty existence, or all
   correlation bounds and diagonal values.

8. **Only one interest-rate factor is supported.** Portfolio valuation selects
   the first rate-parameter row. This prevents multiple currencies or rate
   curves despite the otherwise factor-based structure.

9. **Counterparty credit-risk realism is limited.** The project does not model
   collateral, CSA thresholds, variation margin, initial margin, margin period
   of risk, closeout, recovery uncertainty, DVA or settlement exposure.
   Counterparty and netting-set PFE should remain the primary measures rather
   than aggregate portfolio PFE. The
   [Basel WWR framework](https://www.bis.org/basel_framework/chapter/CRE/53.htm)
   also expects stressed relationships between exposure and creditworthiness
   to be considered.

### Lower-severity findings

10. Output names such as `pfe_95` are fixed even though `WWR_PFE_LEVEL` is
    configurable. A different percentile would therefore be mislabelled.

11. The WWR notebook fixes its analysis to Bank B and named scenarios, while
    the exposure notebook assumes exactly four risk factors in a 2 by 2 layout.

12. `run_project.py` executes when imported, scenario names are fixed,
    dependencies are not pinned, and complete scenario arrays are retained in
    memory.

13. Tests do not currently cover swap benchmark values, par-swap behaviour,
    zero WWR correlation inputs, infeasible correlation combinations,
    stochastic-rate option validation, grid alignment, simulation convergence,
    or independent-CVA factorisation.

### Validation checks passed

- The market correlation matrix is symmetric and positive definite. Its minimum
  eigenvalue is `0.24459459`.
- All current numeric inputs are finite.
- All current WWR correlation combinations are mathematically feasible.
- Current swap frequencies and maturities align with the quarterly grid.
- Ordinary EE and PFE are exactly unchanged across the credit scenarios.
- Correlation-sweep CVA at zero correlation exactly matches independent-scenario
  CVA.
- Existing results show plausible WWR directionality, including increased Bank
  B CVA in the moderate and severe WWR scenarios.

Overall, PROJECT3 is coherent as a clean educational synthetic PFE and WWR
engine. Its main limitations are assumptions and terminology that can make the
results appear more production-ready than they are, together with the genuine
zero-correlation sweep defect.

## Final critical analysis for senior management

### Executive assessment

This project is a credible educational demonstration of how Monte Carlo market
simulation, trade revaluation, netting, potential future exposure (PFE), credit
intensity and wrong-way risk (WWR) can be connected in one transparent Python
workflow. It is useful because the calculation can be followed from source
inputs to final charts without relying on a hidden pricing library or a black-box
risk platform.

It is not, however, a production counterparty-credit-risk engine, a regulatory
capital calculator, or evidence that the reported exposure and CVA amounts are
accurate representations of a real portfolio. The numerical outputs are driven
by synthetic trades and assumed model parameters rather than independently
validated market data. They demonstrate model behaviour, not the actual level
of risk that a bank should manage, hedge, reserve or report.

The most important management conclusion is therefore that the project has
successfully established a **working analytical prototype**, but has not
established a **decision-grade risk measurement process**. That distinction
should remain explicit in every presentation of the results.

### What the project demonstrates well

The project has several genuine strengths:

- It separates market simulation, pricing, exposure, credit and reporting into
  understandable modules. This makes assumptions visible and allows individual
  components to be challenged.
- It revalues trades across simulated market paths instead of applying a simple
  fixed percentage to notional. Exposure therefore responds to market movements,
  remaining maturity, trade direction and netting.
- It preserves signed trade values until netting-set aggregation and applies the
  positive-exposure floor afterwards. This is the correct conceptual order for
  unsecured exposure before collateral.
- It uses correlated market shocks, a Vasicek short-rate model, geometric
  Brownian motion for equities and a CIR credit-intensity model. These are
  recognisable quantitative models rather than arbitrary exposure multipliers.
- It demonstrates the central WWR result clearly: ordinary EE and PFE can remain
  unchanged while default-weighted exposure and CVA increase because adverse
  exposure becomes more likely when the counterparty is weaker.
- It uses common market paths across WWR scenarios. This reduces simulation
  noise when comparing independent, moderate and severe assumptions.
- It produces trade, netting-set and counterparty views together with charts and
  reusable CSV outputs. The results are therefore inspectable rather than being
  confined to a notebook.

These strengths make the project suitable for explaining methodology, testing
ideas and demonstrating the direction of model effects.

### Material weaknesses in the risk methodology

The most material weakness is calibration. Equity volatility, interest-rate
parameters, hazard rates, recovery rates and market-credit correlations are all
assumed. No market data, historical estimation window, calibration objective,
fit error or independent data source supports them. A model can be internally
consistent while still producing economically unreliable numbers if its inputs
are not representative. The current CVA and PFE amounts should consequently be
read as scenario outputs in arbitrary synthetic conditions, not point estimates
of financial loss.

The project also does not state clearly whether every parameter is intended to
be under the real-world probability measure or the risk-neutral pricing measure.
This matters because PFE, accounting CVA and regulatory CVA do not necessarily
use identical probability assumptions. Combining parameters from different
measures without adjustment can produce a result that is mathematically
calculable but economically inconsistent.

The portfolio is too narrow to establish broad engine validity. It includes
European equity options and vanilla interest-rate swaps, all in USD. It does not
test foreign exchange, inflation, commodities, credit derivatives, optional
rates products, path-dependent products, cross-currency discounting or large
multi-currency netting sets. A successful run on the current portfolio therefore
does not demonstrate that the design generalises safely to a bank-wide trading
book.

Exposure is calculated without collateral or margin. There are no CSA
thresholds, minimum transfer amounts, independent amounts, variation margin,
initial margin, margin call frequency, margin disputes, collateral currency,
haircuts, closeout delay or margin period of risk. For collateralised trading
relationships, these terms can dominate the exposure profile. Their absence is
not a minor refinement; it prevents direct comparison with most real counterparty
exposure and capital numbers.

The project calculates unilateral CVA only. It does not include own-credit
effects, funding costs, capital valuation adjustment, margin valuation
adjustment, hedging or accounting treatment. It also assumes deterministic
recovery and a simplified intensity-based default process. The CVA output is
therefore only one component of a wider valuation-adjustment framework.

### Pricing and simulation limitations

The swap calculation is appropriate only when exposure dates coincide with
floating-rate reset dates. It omits accrued interest and the known next floating
coupon between resets. The current data happens to align quarterly payment and
simulation dates, but the software does not enforce this requirement. A future
user could change the schedule and receive plausible-looking but incorrect
values without an error.

European options are valued by inserting a pathwise Vasicek discount factor into
the Black-Scholes formula. This is a useful approximation, but it is not a full
equity-interest-rate hybrid model and does not capture all effects of stochastic
rates and equity-rate correlation on option value. Volatility is also constant,
so volatility smiles, surfaces and forward term structures are absent.

The Vasicek model allows negative rates and is driven by a single short-rate
factor. The CIR credit model is simulated using a time-discretisation method and
has not been tested for convergence at the selected quarterly step size. A
quarterly exposure grid may also miss peaks between reporting dates, especially
for shorter-dated or highly nonlinear positions.

The model has no initial market-curve construction. A real rates implementation
would normally calibrate discount and projection curves to observable
instruments and distinguish curves by currency and index. Selecting the first
rate row as the portfolio rate factor is a material architectural restriction,
not merely a missing data enhancement.

### Wrong-way-risk interpretation

The WWR extension is directionally useful but should not be over-interpreted.
The configured values are correlations between Brownian shocks. They are not
directly the correlations between exposure and default, exposure and hazard
rate, or counterparty asset value and market prices. Portfolio nonlinearities
and the CIR transformation mean the realised relationship may differ
substantially from the input coefficient.

The current moderate and severe scenarios are management assumptions rather
than empirically estimated stresses. There is no historical crisis calibration,
sector mapping, sovereign linkage, legal-entity analysis or specific WWR where
the transaction directly references the counterparty. The scenarios demonstrate
sensitivity to assumed dependence, but they do not establish the appropriate
stress level for risk appetite or regulatory capital.

Default-weighted EE and default-weighted PFE are useful analytical indicators,
but the latter is a project-specific weighted quantile rather than an established
regulatory PFE measure. It should not be presented beside ordinary PFE without
explaining the different conditioning and interpretation.

### Software, controls and model governance

The code is intentionally simple and readable, which is valuable for review.
That simplicity also leaves control gaps. Input validation does not cover all
economically invalid parameters, schedule inconsistencies, duplicate factors or
jointly infeasible WWR assumptions. One valid zero-correlation input can cause a
division by zero in the correlation sweep. Configurable percentiles can also be
written under fixed `pfe_95` labels, creating a risk of misreported output.

The tests establish useful basic properties but do not amount to model
validation. Missing controls include independent benchmark portfolios, par-swap
tests, comparison with trusted pricing libraries, Monte Carlo convergence,
confidence intervals, sensitivity testing, stress testing, backtesting,
performance testing and reconciliation of CVA to an independent implementation.
Passing the current tests means that selected coding relationships hold; it does
not prove that the methodology is complete or that reported values are accurate.

There is no formal model inventory entry, versioned calibration dataset,
parameter approval process, change control, model-owner sign-off, independent
validation report or controlled production run record. Outputs are ordinary CSV
and image files and can be overwritten by rerunning the project. For management
or regulatory use, the process would need reproducible run identifiers, data
lineage, immutable results, exception logs and documented approvals.

The dependency versions are not pinned, and `run_project.py` executes when
imported. These are manageable prototype issues, but they weaken reproducibility
and operational control. Memory usage also grows with paths, dates,
counterparties and scenarios because complete path arrays are retained.

### Interpretation of the current results

The current outputs show the intended qualitative behaviour. Market exposure is
unchanged across credit scenarios, while Bank B's CVA rises as the assumed
adverse market-credit dependence becomes stronger. This is evidence that the
WWR mechanism is connected correctly at a high level.

It is not evidence that Bank B is genuinely riskier by the reported percentage,
that the severe scenario has the correct probability, or that the resulting CVA
would reconcile to accounting or regulatory systems. The reported uplift is a
conditional result of the selected synthetic portfolio, correlations, hazard
process, recovery assumption, simulation grid and valuation approximations.
Changing any of these may materially change both the size and direction of the
result.

Senior management should therefore focus on the comparative message rather than
the absolute currency values: dependence between exposure and counterparty
credit quality can materially increase expected credit loss even when the
standalone exposure distribution appears unchanged.

### Readiness assessment

| Intended use | Assessment | Reason |
| --- | --- | --- |
| Education and methodology demonstration | Suitable | The workflow is transparent and the principal PFE and WWR concepts are visible. |
| Exploratory scenario analysis | Suitable with clear caveats | Relative effects are useful, but inputs and stresses are assumed. |
| Internal prototype development | Suitable starting point | The modular structure can support controlled extensions. |
| Trading-limit or credit-approval decisions | Not suitable | Calibration, collateral, product coverage and validation are insufficient. |
| Accounting CVA or financial reporting | Not suitable | Market calibration, controls, governance and reconciliation are absent. |
| Regulatory capital or regulatory submission | Not suitable | The methodology and governance do not meet production or regulatory standards. |

### Recommended management priorities

1. **Correct known defects and lock down definitions.** Resolve the
   zero-correlation failure, make output labels follow configured percentiles,
   enforce schedule alignment and define every reported metric precisely.

2. **Establish a controlled market-data and calibration process.** Separate
   real-world and risk-neutral parameters, construct market curves, calibrate
   volatility and credit term structures, and record calibration quality.

3. **Add collateral and closeout mechanics.** Model CSA terms, margin flows and
   margin period of risk before using exposure results for real counterparties.

4. **Strengthen pricing validation.** Benchmark each product against an
   independent implementation and add accrued-interest, reset-date and
   stochastic-rate tests.

5. **Validate simulation stability.** Report Monte Carlo standard errors,
   percentile confidence intervals and convergence across path counts and time
   steps.

6. **Develop a defensible WWR framework.** Report realised exposure-credit
   dependence, calibrate or justify stress assumptions, add sector and
   counterparty-specific scenarios, and distinguish general from specific WWR.

7. **Introduce production controls only if production use is intended.** Add
   pinned dependencies, run identifiers, immutable inputs and outputs, data
   lineage, logging, access controls, performance monitoring and formal model
   governance.

### Final conclusion

The project should be judged positively as a transparent quantitative prototype
that explains PFE and WWR more convincingly than a purely formula-based example.
Its clean structure provides a good foundation for further development and its
current outputs support the expected qualitative risk narrative.

The project should be judged critically if presented as a measure of actual bank
risk. The largest gap is not code sophistication; it is the absence of calibrated
data, collateral mechanics, independent model validation and production
governance. Until those gaps are addressed, the appropriate management use is
education, challenge and exploratory analysis. It should not be used for limits,
pricing, reserves, capital or regulatory reporting.
