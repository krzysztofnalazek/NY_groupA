from config import OUTPUT_DIR, WWR_FOCUS_COUNTERPARTY
from src.engine import run


results = run()
profile = results["outputs"]["exposure_profile"]
wwr_summary = results["outputs"]["wwr_summary"]
focus = wwr_summary[wwr_summary["counterparty"] == WWR_FOCUS_COUNTERPARTY]
focus_severe = focus[focus["scenario"] == "severe"].iloc[0]

print("PFE simulation complete")
print("Outputs:", OUTPUT_DIR)
print("Maximum EE:      {:,.0f}".format(profile["ee"].max()))
print("Maximum 95% PFE: {:,.0f}".format(profile["pfe_95"].max()))
print("Maximum 99% PFE: {:,.0f}".format(profile["pfe_99"].max()))
print(
    WWR_FOCUS_COUNTERPARTY + " independent CVA: {:,.0f}".format(
        focus_severe["independent_cva"]
    )
)
print(
    WWR_FOCUS_COUNTERPARTY + " severe WWR CVA:  {:,.0f}".format(
        focus_severe["cva"]
    )
)
