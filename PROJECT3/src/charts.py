import os

import numpy as np

from config import (
    CHART_DIR,
    PROJECT_DIR,
    WWR_FOCUS_COUNTERPARTY,
    WWR_SCENARIOS,
)


os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def plot_market_paths(paths, time_grid):
    n_columns = 2
    n_rows = int(np.ceil(len(paths) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(13, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for chart_number, (risk_factor, values) in enumerate(paths.items()):
        axes[chart_number].plot(time_grid, values[:30].T, alpha=0.35)
        axes[chart_number].set_title(risk_factor)
        axes[chart_number].set_xlabel("Years")
        axes[chart_number].set_ylabel("Rate" if "RATE" in risk_factor else "Index level")

    for chart_number in range(len(paths), len(axes)):
        axes[chart_number].set_visible(False)

    fig.suptitle("Simulated market paths")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_market_paths.png", dpi=150)
    plt.close(fig)


def plot_exposure_profile(profile):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(profile["time_years"], profile["ee"], label="EE", linewidth=2)

    pfe_columns = [column for column in profile.columns if column.startswith("pfe_")]
    for column in pfe_columns:
        confidence = column.replace("pfe_", "")
        ax.plot(profile["time_years"], profile[column], label=confidence + "% PFE")
    ax.set_title("Portfolio exposure profile")
    ax.set_xlabel("Years")
    ax.set_ylabel("Exposure (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_exposure_profile.png", dpi=150)
    plt.close(fig)


def plot_exposure_distributions(distributions, profile):
    dates = distributions["time_years"].unique()
    pfe_columns = [column for column in profile.columns if column.startswith("pfe_")]
    pfe_column = pfe_columns[0]
    confidence = pfe_column.replace("pfe_", "")
    fig, axes = plt.subplots(1, len(dates), figsize=(15, 4), sharey=True)

    for chart_number, time in enumerate(dates):
        values = distributions.loc[
            distributions["time_years"] == time, "exposure"
        ]
        profile_row = profile.iloc[(profile["time_years"] - time).abs().argmin()]
        axes[chart_number].hist(values, bins=40, color="#4C78A8", alpha=0.8)
        axes[chart_number].axvline(profile_row["ee"], color="#E45756", label="EE")
        axes[chart_number].axvline(
            profile_row[pfe_column],
            color="#54A24B",
            linestyle="--",
            label=confidence + "% PFE",
        )
        axes[chart_number].set_title("Year " + str(round(time, 2)))
        axes[chart_number].set_xlabel("Exposure (USD)")

    axes[0].set_ylabel("Number of paths")
    axes[-1].legend()
    fig.suptitle("Exposure distributions")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_exposure_distributions.png", dpi=150)
    plt.close(fig)


def plot_netting_benefit(profile):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(profile["time_years"], profile["gross_ee"], label="Before netting")
    ax.plot(profile["time_years"], profile["ee"], label="After netting")
    ax.fill_between(
        profile["time_years"],
        profile["ee"],
        profile["gross_ee"],
        alpha=0.2,
        label="Netting benefit",
    )
    ax.set_title("Expected exposure before and after netting")
    ax.set_xlabel("Years")
    ax.set_ylabel("Expected exposure (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_netting_benefit.png", dpi=150)
    plt.close(fig)


def plot_netting_sets(netting_profile):
    fig, ax = plt.subplots(figsize=(10, 6))

    for netting_set, data in netting_profile.groupby("netting_set"):
        ax.plot(data["time_years"], data["ee"], label=netting_set)

    ax.set_title("Expected exposure by netting set")
    ax.set_xlabel("Years")
    ax.set_ylabel("Expected exposure (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "05_netting_sets.png", dpi=150)
    plt.close(fig)


def plot_correlations(correlations):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlations.values, vmin=-1, vmax=1, cmap="coolwarm")
    labels = correlations.columns
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)

    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, f"{correlations.iloc[row, column]:.2f}",
                    ha="center", va="center")

    ax.set_title("Risk-factor correlations")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "06_correlations.png", dpi=150)
    plt.close(fig)


def plot_trade_summary(trade_summary):
    data = trade_summary.sort_values("max_standalone_ee")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["trade_id"], data["max_standalone_ee"], color="#72B7B2")
    ax.set_title("Maximum standalone EE by trade")
    ax.set_xlabel("Exposure (USD)")
    ax.set_ylabel("Trade")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "07_trade_exposure.png", dpi=150)
    plt.close(fig)


def plot_default_weighted_ee(wwr_profile):
    data = wwr_profile[wwr_profile["counterparty"] == WWR_FOCUS_COUNTERPARTY]
    ordinary = data[data["scenario"] == "independent"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        ordinary["time_years"], ordinary["ee"],
        label="Ordinary EE", linewidth=2, color="black"
    )

    for scenario in WWR_SCENARIOS:
        if scenario == "independent":
            continue
        scenario_data = data[data["scenario"] == scenario]
        ax.plot(
            scenario_data["time_years"],
            scenario_data["default_weighted_ee"],
            label=scenario.title() + " WWR EE",
        )

    ax.set_title(WWR_FOCUS_COUNTERPARTY + ": ordinary and default-weighted EE")
    ax.set_xlabel("Years")
    ax.set_ylabel("Exposure (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "08_default_weighted_ee.png", dpi=150)
    plt.close(fig)


def plot_cva_scenarios(wwr_summary):
    data = wwr_summary.pivot(
        index="counterparty", columns="scenario", values="cva"
    )
    data = data[WWR_SCENARIOS]

    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="bar", ax=ax)
    ax.set_title("Independent and WWR CVA")
    ax.set_xlabel("Counterparty")
    ax.set_ylabel("CVA (USD)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "09_cva_scenarios.png", dpi=150)
    plt.close(fig)


def plot_correlation_sweep(correlation_sweep):
    fig, ax = plt.subplots(figsize=(10, 6))

    for counterparty, data in correlation_sweep.groupby("counterparty"):
        ax.plot(
            data["correlation"], data["cva_uplift_pct"],
            marker="o", label=counterparty
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("CVA uplift across market-credit correlations")
    ax.set_xlabel("Correlation with primary market factor")
    ax.set_ylabel("CVA uplift (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "10_wwr_correlation_sweep.png", dpi=150)
    plt.close(fig)


def plot_exposure_hazard_scatter(scatter_sample):
    fig, ax = plt.subplots(figsize=(9, 6))
    points = ax.scatter(
        scatter_sample["exposure"],
        scatter_sample["hazard_rate"],
        c=scatter_sample["default_weight"],
        cmap="viridis",
        alpha=0.55,
        s=18,
    )
    time = scatter_sample["time_years"].iloc[0]
    ax.set_title(
        WWR_FOCUS_COUNTERPARTY + ": exposure and hazard at year " + str(round(time, 2))
    )
    ax.set_xlabel("Exposure (USD)")
    ax.set_ylabel("Hazard rate")
    fig.colorbar(points, ax=ax, label="Default probability weight")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "11_exposure_hazard_scatter.png", dpi=150)
    plt.close(fig)


def plot_default_weighted_distribution(distribution_sample):
    values = distribution_sample["exposure"]
    bins = np.linspace(values.min(), values.max(), 40)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values,
        bins=bins,
        weights=distribution_sample["ordinary_weight"],
        alpha=0.55,
        label="Ordinary exposure distribution",
    )
    ax.hist(
        values,
        bins=bins,
        weights=distribution_sample["default_weight"],
        histtype="step",
        linewidth=2,
        label="Default-weighted distribution",
    )
    time = distribution_sample["time_years"].iloc[0]
    ax.set_title(
        WWR_FOCUS_COUNTERPARTY + ": default-weighted exposure at year "
        + str(round(time, 2))
    )
    ax.set_xlabel("Exposure (USD)")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "12_default_weighted_distribution.png", dpi=150)
    plt.close(fig)


def plot_wwr_heatmap(correlation_sweep):
    heatmap = correlation_sweep.pivot(
        index="counterparty", columns="correlation", values="cva_uplift_pct"
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    limit = np.abs(heatmap.values).max()
    image = ax.imshow(heatmap.values, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(heatmap.columns)), heatmap.columns)
    ax.set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    ax.set_xlabel("Correlation with primary market factor")
    ax.set_title("CVA uplift from market-credit dependence")

    for row in range(len(heatmap.index)):
        for column in range(len(heatmap.columns)):
            ax.text(
                column, row, f"{heatmap.iloc[row, column]:.1f}%",
                ha="center", va="center", fontsize=8
            )

    fig.colorbar(image, ax=ax, label="CVA uplift (%)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "13_wwr_heatmap.png", dpi=150)
    plt.close(fig)


def make_all_charts(paths, time_grid, outputs, correlations):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_market_paths(paths, time_grid)
    plot_exposure_profile(outputs["exposure_profile"])
    plot_exposure_distributions(
        outputs["exposure_distributions"], outputs["exposure_profile"]
    )
    plot_netting_benefit(outputs["exposure_profile"])
    plot_netting_sets(outputs["netting_set_profile"])
    plot_correlations(correlations)
    plot_trade_summary(outputs["trade_summary"])
    plot_default_weighted_ee(outputs["wwr_profile"])
    plot_cva_scenarios(outputs["wwr_summary"])
    plot_correlation_sweep(outputs["wwr_correlation_sweep"])
    plot_exposure_hazard_scatter(outputs["wwr_scatter_sample"])
    plot_default_weighted_distribution(outputs["wwr_distribution_sample"])
    plot_wwr_heatmap(outputs["wwr_correlation_sweep"])
