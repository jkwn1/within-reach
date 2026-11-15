"""Step 5: turn value-added into something a low-income student or counselor can use.

"Opportunity colleges" must meet all of:
  * former students (dependent students on federal aid, working 10 years after entry, whether or not
    they finished) earn at least 10% more than the model expects for that student body (value-added >= +10%),
  * a share of Pell (low-income) students at or above the median of ranked colleges,
  * a net price for students from families earning up to $30,000 at or below that median,
  * at least 500 students, so the earnings median is stable,
  * its own earnings figure. This rule was added after a first draft of the list: the review found that
    branch campuses inherit their system's figure (e.g. a regional campus reporting the flagship's
    earnings), which makes campus-level value-added meaningless for them.
The medians are computed over the ranked colleges, so the bars move with the data, not by hand.

Value-added is an estimate, so each listed college also gets two robustness flags:
  * `robust_no_race`: still qualifies when the model has no race/ethnicity inputs,
  * `borderline`: value-added within 3 points of the +10% bar.

Also reports "hidden gems" (bottom 20% on raw earnings but value-added >= +10%) and what student
parents still face at opportunity colleges (the child-care-adjusted affordability gap). The source's
"weekly hours to close the gap" are hours ON TOP OF 10 hours/week of minimum-wage work it already assumes.

Outputs: results/opportunity_colleges.csv, results/ranked_colleges.parquet, results/opportunity_summary.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import BAY_COUNTIES, INTERIM, MSI_TYPES, RESULTS  # noqa: E402

MIN_ENROLLMENT = 500
VA_THRESHOLD_PCT = 10.0
BORDERLINE_POINTS = 3.0
HOURS_ALREADY_ASSUMED = 10  # the affordability gap already subtracts 10 h/week at minimum wage
BAY_COUNTY_NAMES = {f"{name} County" for name in BAY_COUNTIES.values()}


def msi_counts(frame):
    return {t: int(frame[t].sum()) for t in MSI_TYPES} | {"n": int(len(frame))}


def main():
    d = pd.read_parquet(RESULTS / "college_predictions.parquet")
    extra = pd.read_parquet(INTERIM / "college_analysis.parquet", columns=["unit_id", "parent_weekly_hours_center_care"])
    d = d.merge(extra, on="unit_id", how="left", validate="one_to_one")

    ranked = d[(d["enrollment"] >= MIN_ENROLLMENT) & (d["report_group_size"] == 1)
               & d["net_price_low_income"].notna() & d["pell_share"].notna()].copy()
    pell_med = float(ranked["pell_share"].median())
    price_med = float(ranked["net_price_low_income"].median())
    raw_cut = float(ranked["earnings_dep_10yr"].quantile(0.2))

    access = (ranked["pell_share"] >= pell_med) & (ranked["net_price_low_income"] <= price_med)
    ranked["opportunity"] = access & (ranked["value_added_pct"] >= VA_THRESHOLD_PCT)
    ranked["robust_no_race"] = ranked["opportunity"] & (ranked["value_added_no_race_pct"] >= VA_THRESHOLD_PCT)
    ranked["borderline"] = ranked["opportunity"] & (ranked["value_added_pct"] < VA_THRESHOLD_PCT + BORDERLINE_POINTS)
    ranked["hidden_gem"] = (ranked["earnings_dep_10yr"] <= raw_cut) & (ranked["value_added_pct"] >= VA_THRESHOLD_PCT)
    ranked["bay_area"] = (ranked["state"] == "CA") & ranked["county_name"].isin(BAY_COUNTY_NAMES)
    ranked["total_weekly_work_hours_parent"] = ranked["parent_weekly_hours_center_care"] + HOURS_ALREADY_ASSUMED
    ranked["total_weekly_work_hours_no_children"] = ranked["weekly_hours_to_close_gap"] + HOURS_ALREADY_ASSUMED

    opp = ranked[ranked["opportunity"]].sort_values("value_added_pct", ascending=False)
    cols = ["unit_id", "name", "city", "state", "county_name", "sector_name", "enrollment", "pell_share",
            "net_price_low_income", "earnings_dep_10yr", "expected_earnings", "value_added_pct",
            "value_added_no_race_pct", "robust_no_race", "borderline", "affordability_gap",
            "total_weekly_work_hours_no_children", "parent_gap_center_care", "total_weekly_work_hours_parent"] \
        + MSI_TYPES + ["bay_area", "hidden_gem"]
    opp[cols].to_csv(RESULTS / "opportunity_colleges.csv", index=False)
    ranked.to_parquet(RESULTS / "ranked_colleges.parquet", index=False)

    bay = ranked[ranked["bay_area"]]
    summary = {
        "rules": {"min_enrollment": MIN_ENROLLMENT, "value_added_at_least_pct": VA_THRESHOLD_PCT,
                  "pell_share_at_least_median": pell_med, "net_price_low_income_at_most_median_usd": price_med,
                  "own_earnings_figure_only": True},
        "n_ranked": int(len(ranked)),
        "n_opportunity_colleges": int(len(opp)),
        "n_robust_no_race": int(opp["robust_no_race"].sum()),
        "n_borderline": int(opp["borderline"].sum()),
        "opportunity_by_sector": opp["sector_name"].value_counts().to_dict(),
        "opportunity_msi_counts": msi_counts(opp),
        "ranked_msi_counts": msi_counts(ranked),
        "n_opportunity_california": int((opp["state"] == "CA").sum()),
        "n_opportunity_bay_area": int(opp["bay_area"].sum()),
        "california_opportunity_colleges": opp.loc[opp["state"] == "CA", ["name", "city", "sector_name",
                                                                           "value_added_pct", "net_price_low_income",
                                                                           "pell_share", "robust_no_race",
                                                                           "borderline"]].round(1).to_dict("records"),
        "bay_area": {"n_ranked": int(len(bay)),
                     "n_value_added_at_least_threshold": int((bay["value_added_pct"] >= VA_THRESHOLD_PCT).sum()),
                     "n_pell_at_least_median": int((bay["pell_share"] >= pell_med).sum()),
                     "n_price_at_most_median": int((bay["net_price_low_income"] <= price_med).sum())},
        "n_hidden_gems": int(ranked["hidden_gem"].sum()),
        "hidden_gem_msi_counts": msi_counts(ranked[ranked["hidden_gem"]]),
        "student_parents_at_opportunity_colleges": {
            "n_with_child_care_data": int(opp["parent_gap_center_care"].notna().sum()),
            "median_center_care_gap_usd": float(opp["parent_gap_center_care"].median()),
            "median_total_weekly_work_hours": float(opp["total_weekly_work_hours_parent"].median()),
            "median_gap_without_children_usd": float(opp["affordability_gap"].median()),
            "median_total_weekly_work_hours_without_children": float(opp["total_weekly_work_hours_no_children"].median())},
        "median_net_price_low_income_opportunity_usd": float(opp["net_price_low_income"].median()),
        "median_value_added_opportunity_pct": float(opp["value_added_pct"].median()),
    }
    (RESULTS / "opportunity_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "california_opportunity_colleges"}, indent=1, default=float))
    print(pd.DataFrame(summary["california_opportunity_colleges"]).to_string(index=False))


if __name__ == "__main__":
    main()
