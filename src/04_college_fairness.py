"""Step 4: error and fairness analysis.

Three questions:
  1. Is the model equally accurate for every kind of college? Error and bias by minority-serving
     type, sector, share of low-income (Pell) students, and region. Cross-fitted predictions cover all
     4,194 colleges (each predicted by models that never saw it); the held-out test set is reported
     too, with the simple Pell + sector baseline beside the model.
  2. Who gets labeled "low value" by raw earnings vs by value-added? Shares of each group in the
     bottom 20% under each measure.
  3. How much do value-added scores depend on including race/ethnicity as inputs? (Including it
     compares colleges with similar student bodies; excluding it blames colleges for gaps in the labor
     market they did not create.)

Outputs: results/errors_by_group.csv, results/bottom20_by_measure.csv, results/race_sensitivity.csv,
         results/worst_test_predictions.csv, results/fairness_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import MSI_TYPES, RESULTS  # noqa: E402

REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, states in REGIONS.items() for s in states.split()}
SECTOR_GROUP = {"Public 4-year": "Public 4-year", "Private nonprofit 4-year": "Private nonprofit 4-year",
                "For-profit 4-year": "For-profit 4-year", "Public 2-year": "Public 2-year",
                "Private nonprofit 2-year": "Nonprofit/for-profit 2-year", "For-profit 2-year": "Nonprofit/for-profit 2-year",
                "Public <2-year": "Less-than-2-year", "Private nonprofit <2-year": "Less-than-2-year",
                "For-profit <2-year": "Less-than-2-year"}


def add_groups(df):
    df = df.copy()
    df["msi_type"] = "None of these"
    for t in MSI_TYPES:  # a college can hold several designations; report each separately below
        df.loc[df[t] == 1, "msi_type"] = t
    df["sector_group"] = df["sector_name"].map(SECTOR_GROUP)
    df["pell_tercile"] = pd.qcut(df["pell_share"], 3, labels=["Low Pell share", "Middle Pell share", "High Pell share"])
    df["region"] = df["state"].map(STATE_REGION).fillna("Territories")
    return df


def error_table(df, pred_col, label):
    rows = []
    def summarize(name, value, d):
        err = d[pred_col] - d["earnings_dep_10yr"]
        rows.append({"predictions": label, "dimension": name, "group": value, "n": int(len(d)),
                     "mae_usd": float(err.abs().mean()),
                     "median_abs_pct_error": float(100 * (err.abs() / d["earnings_dep_10yr"]).median()),
                     "mean_signed_error_usd": float(err.mean()),
                     "mean_signed_error_pct": float(100 * (err / d["earnings_dep_10yr"]).mean())})
    summarize("All colleges", "All", df)
    for t in MSI_TYPES:  # designations overlap, so each one is compared with everyone else
        summarize("Minority-serving designation", t, df[df[t] == 1])
    summarize("Minority-serving designation", "None of the five", df[df[MSI_TYPES].sum(axis=1) == 0])
    for dim, col in [("Sector", "sector_group"), ("Low-income share", "pell_tercile"), ("Region", "region")]:
        for value, d in df.groupby(col, observed=True):
            summarize(dim, str(value), d)
    return pd.DataFrame(rows)


def bottom20(df):
    """Share of each group among colleges in the bottom 20% by raw earnings vs by value-added."""
    # Rank only colleges with 500+ students (stable medians) and their OWN earnings figure: a figure shared by
    # several campuses describes the whole reporting unit, so it cannot rank one campus.
    d = df[(df["enrollment"] >= 500) & (df["report_group_size"] == 1)].copy()
    raw_cut = d["earnings_dep_10yr"].quantile(0.2)
    va_cut = d["value_added_pct"].quantile(0.2)
    va_nr_cut = d["value_added_no_race_pct"].quantile(0.2)
    d["bottom_raw"] = d["earnings_dep_10yr"] <= raw_cut
    d["bottom_va"] = d["value_added_pct"] <= va_cut
    d["bottom_va_nr"] = d["value_added_no_race_pct"] <= va_nr_cut
    rows = []
    groups = {**{t: d[t] == 1 for t in MSI_TYPES},
              # Terciles are cut on all 4,194 colleges, so among ranked colleges this group is Pell share
              # above about 61% (the top 16% of them), not the top third. The label says so.
              "Pell share above 61%": d["pell_tercile"] == "High Pell share",
              "All colleges": pd.Series(True, index=d.index)}
    for name, mask in groups.items():
        g = d[mask]
        rows.append({"group": name, "n": int(len(g)),
                     "share_in_bottom20_raw_earnings": float(g["bottom_raw"].mean()),
                     "share_in_bottom20_value_added": float(g["bottom_va"].mean()),
                     "share_in_bottom20_value_added_no_race_inputs": float(g["bottom_va_nr"].mean())})
    return pd.DataFrame(rows), {"n_colleges_ranked": int(len(d)), "raw_earnings_cutoff_usd": float(raw_cut),
                                "value_added_cutoff_pct": float(va_cut)}


def race_sensitivity(df):
    df = df[df["report_group_size"] == 1]  # value-added is only meaningful for a college's own figure
    rows = []
    for name, mask in {**{t: df[t] == 1 for t in MSI_TYPES},
                       "None of the five": df[MSI_TYPES].sum(axis=1) == 0}.items():
        g = df[mask]
        rows.append({"group": name, "n": int(len(g)),
                     "median_value_added_pct_with_race_inputs": float(g["value_added_pct"].median()),
                     "median_value_added_pct_without_race_inputs": float(g["value_added_no_race_pct"].median())})
    return pd.DataFrame(rows)


def main():
    pred = add_groups(pd.read_parquet(RESULTS / "college_predictions.parquet"))
    test = pred[pred["in_test_set"]]
    errors = pd.concat([error_table(pred, "expected_earnings", "cross-fitted model, all colleges"),
                        error_table(test, "test_pred_gbm", "model, held-out test set"),
                        error_table(test, "test_pred_pell_sector", "Pell + sector baseline, held-out test set")])
    errors.to_csv(RESULTS / "errors_by_group.csv", index=False)

    b20, b20_meta = bottom20(pred)
    b20.to_csv(RESULTS / "bottom20_by_measure.csv", index=False)
    sens = race_sensitivity(pred)
    sens.to_csv(RESULTS / "race_sensitivity.csv", index=False)

    worst = test.assign(error_usd=test["test_pred_gbm"] - test["earnings_dep_10yr"],
                        error_pct=100 * (test["test_pred_gbm"] / test["earnings_dep_10yr"] - 1))
    worst = worst.reindex(worst["error_usd"].abs().sort_values(ascending=False).index).head(15)
    worst[["unit_id", "name", "city", "state", "sector_name", "enrollment", "pell_share", "earnings_dep_10yr",
           "test_pred_gbm", "error_usd", "error_pct"]].to_csv(RESULTS / "worst_test_predictions.csv", index=False)

    cf = errors[errors["predictions"].str.startswith("cross-fitted")]
    summary = {"bottom20_meta": b20_meta,
               "largest_abs_bias_groups_crossfit": cf.reindex(cf["mean_signed_error_pct"].abs()
                                                              .sort_values(ascending=False).index)
               .head(6)[["dimension", "group", "n", "mean_signed_error_pct", "mae_usd"]].to_dict("records")}
    (RESULTS / "fairness_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(cf[["dimension", "group", "n", "mae_usd", "median_abs_pct_error", "mean_signed_error_pct"]].round(1).to_string(index=False))
    print(b20.round(3).to_string(index=False), b20_meta)
    print(sens.round(1).to_string(index=False))
    print(worst[["name", "state", "sector_name", "enrollment", "earnings_dep_10yr", "test_pred_gbm", "error_pct"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
