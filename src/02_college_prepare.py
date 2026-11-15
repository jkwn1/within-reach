"""Step 2: build one analysis row per college from the two College Affordability tables.

Key data decisions (details in data_decisions.md):
  * Target = median earnings of DEPENDENT students, 10 years after entry. The all-students column
    ("Median Earnings of Students Working and Not Enrolled 10 Years After Entry") holds only three
    distinct values across 5,831 colleges in this export, so it cannot measure differences between colleges.
  * Campuses of one chain or system share one federal reporting unit, so they report identical
    figures. A reporting unit = colleges with the same dependent-student earnings AND independent-student
    earnings AND dependent-student median debt (one matching number can be a coincidence; three rarely
    are). Each unit is one `report_group`: never split between training and test data.
  * The affordability table has up to 5 rows per college, one per family-income bracket. We keep the
    lowest bracket (family income up to $30,000): the students this project is about.
  * Model inputs describe WHO ENROLLS and WHERE (intake and context), never what the college produces
    (graduation, retention, debt) or spends: value-added should credit those, not explain them away.

Output: data/interim/college_analysis.parquet, results/college_data_quality.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
import common  # noqa: E402
from common import INTERIM, RESULTS  # noqa: E402

ID = "UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"
TARGET_RAW = "Median Earnings of Dependent Students Working and Not Enrolled 10 Years After Entry"
UNIT_KEY_RAW = ["Median Earnings of Independent Students Working and Not Enrolled 10 Years After Entry",
                "Median Debt for Dependent Students"]  # with the target, identifies a reporting unit
SPECIAL_FOCUS = range(51, 60)  # Carnegie 2000 codes 51-59: specialized schools by field (medical, law, art...)
BROKEN_RAW = "Median Earnings of Students Working and Not Enrolled 10 Years After Entry"

NUMERIC_FEATURES = {  # short name -> column in College Results View 2021
    "pell_share": "Percent of First-Time, Full-Time Undergraduates Awarded Pell Grants",
    "pct_black": "Percent of Black or African American Undergraduates",
    "pct_latino": "Percent of Latino Undergraduates",
    "pct_asian": "Percent of Asian Undergraduates",
    "pct_white": "Percent of White Undergraduates",
    "pct_aian": "Percent of American Indian or Alaska Native Undergraduates",
    "pct_nhpi": "Percent of Native Hawaiian or Other Pacific Islander Undergraduates",
    "pct_two_plus": "Percent of Two or More Races Undergraduates",
    "pct_women": "Percent of Women Undergraduates",
    "pct_age25plus": "Percent of Undergraduates Age 25 to 64",
    "pct_part_time": "Percent of Part-Time Undergraduates",
    "pct_distance_only": "Percent of Undergraduates Enrolled Exclusively in Distance Education Courses",
    "pct_ftft": "Percent of Full-time, First-time Degree-Seeking Undergraduates",
    "log_enrollment": "Total Enrollment",
    "admit_rate": "Total Percent of Applicants Admitted",
    "sat_math_25": "SAT Math - 25th Percentile Score",
    "sat_erw_25": "SAT Evidence Based Reading and Writing - 25th Percentile Score",
    "act_math_25": "ACT Math - 25th Percentile Score",
    "open_admissions": "Open Admissions Policy",
}
CATEGORICAL_FEATURES = {
    "sector": "Sector of Institution",
    "carnegie": "Carnegie Classification",
    "urbanization": "Degree of Urbanization",
    "size_category": "Institution Size Category",
    "state": "State of Institution",
}
assert list(NUMERIC_FEATURES) == common.NUMERIC_FEATURES and list(CATEGORICAL_FEATURES) == common.CATEGORICAL_FEATURES

SECTOR_NAMES = {1: "Public 4-year", 2: "Private nonprofit 4-year", 3: "For-profit 4-year",
                4: "Public 2-year", 5: "Private nonprofit 2-year", 6: "For-profit 2-year",
                7: "Public <2-year", 8: "Private nonprofit <2-year", 9: "For-profit <2-year"}
# Minority-serving institution flags: from the results file (97% filled) except HSI, which is empty there
# and only available in the affordability file (so HSI status is unknown for colleges not matched to it).
MSI_FLAGS_RESULTS = {"HBCU": "Historically Black College or University (HBCU)",
                     "TCU": "Tribal College or University (TCU)",
                     "PBI": "Predominantly Black Institution (PBI)",
                     "AANAPISI": "Asian American or Native American Pacific Islander-Serving Institution (AANAPISI)"}
MSI_FLAGS_GAP = {"HSI": "HSI"}

GAP_KEEP = {  # short name -> column in Affordability Gap AY2022-23 (lowest family-income bracket row)
    "net_price_low_income": "Net Price",
    "affordability_gap": "Affordability Gap (net price minus income earned working 10 hrs at min wage)",
    "weekly_hours_to_close_gap": "Weekly Hours to Close Gap",
    "parent_gap_center_care": "Student Parent Affordability Gap: Center-Based Care",
    "parent_weekly_hours_center_care": "Weekly Hours to Close Gap: Center-Based Care",
    "county_name": "County Name",
    "lat": "Latitude",
    "lon": "Longitude",
    "region": "Region",
    "gap_sector": "Sector",
}


def main():
    res = pd.read_parquet(INTERIM / "college_results.parquet")
    gap = pd.read_parquet(INTERIM / "college_gap.parquet")
    quality = {"colleges_in_results_file": int(len(res))}

    broken = res[BROKEN_RAW].dropna()
    quality["all_students_earnings_column"] = {
        "colleges_with_value": int(len(broken)), "distinct_values": int(broken.nunique()),
        "values": sorted(float(v) for v in broken.unique())[:10]}

    df = pd.DataFrame({"unit_id": res[ID].astype("int64"), "name": res["Institution Name"],
                       "city": res["City of Institution"],
                       "earnings_dep_10yr": pd.to_numeric(res[TARGET_RAW], errors="coerce"),
                       "earnings_indep_10yr": pd.to_numeric(res[UNIT_KEY_RAW[0]], errors="coerce"),
                       "debt_dependent": pd.to_numeric(res[UNIT_KEY_RAW[1]], errors="coerce")})
    for short, col in NUMERIC_FEATURES.items():
        df[short] = pd.to_numeric(res[col], errors="coerce")
    df["log_enrollment"] = np.log1p(df["log_enrollment"])
    for short, col in MSI_FLAGS_RESULTS.items():
        df[short] = pd.to_numeric(res[col], errors="coerce").fillna(0).astype(int)
    for short, col in CATEGORICAL_FEATURES.items():
        df[short] = res[col]
    # The Carnegie codes here are the 2000 classification; codes 51-59 name the field a specialized school
    # teaches (medicine, engineering, art...). Field is program mix, which the model must not see, so the
    # nine codes become one "specialized school" code.
    df["carnegie"] = df["carnegie"].where(~df["carnegie"].isin(SPECIAL_FOCUS), 50)
    df["sector_name"] = df["sector"].map(SECTOR_NAMES)
    df["enrollment"] = pd.to_numeric(res["Total Enrollment"], errors="coerce")
    df["grad_rate_6yr"] = pd.to_numeric(res["Bachelor's Degree Graduation Rate Bachelor Degree Within 6 Years - Total"],
                                        errors="coerce")
    df["median_debt_completers"] = pd.to_numeric(res["Median Debt of Completers"], errors="coerce")
    df["net_price_low_income_2021"] = pd.to_numeric(res["Average Net Price for Low-Income Students, 2020-21"],
                                                    errors="coerce")

    quality["target"] = {"column": TARGET_RAW, "colleges_with_value": int(df["earnings_dep_10yr"].notna().sum()),
                         "distinct_values": int(df["earnings_dep_10yr"].nunique())}
    df = df[df["earnings_dep_10yr"].notna()].copy()

    # Reporting units: identical target AND independent earnings AND dependent debt (missing counts as a value)
    counts_target_only = df["earnings_dep_10yr"].map(df["earnings_dep_10yr"].value_counts())
    key = df[["earnings_dep_10yr", "earnings_indep_10yr", "debt_dependent"]].fillna(-1)
    df["report_group"] = key.groupby(list(key.columns), sort=True).ngroup().astype(int)
    df["report_group_size"] = df.groupby("report_group")["report_group"].transform("size").astype(int)
    quality["shared_earnings_values"] = {
        "colleges_sharing_target_value_with_another": int((counts_target_only > 1).sum()),
        "colleges_in_multi_college_reporting_units": int((df["report_group_size"] > 1).sum()),
        "share_in_multi_college_units": float((df["report_group_size"] > 1).mean()),
        "largest_unit_size": int(df["report_group_size"].max()),
        "n_reporting_units": int(df["report_group"].nunique())}

    low = gap[gap["Student Family Earnings Ceiling"] == 30000]
    assert low["Unit ID"].is_unique, "expected one lowest-bracket row per college"
    # HSI status is a property of the college, so read it from any of its rows (some lack the $30k row)
    hsi_any_row = gap.groupby("Unit ID")["HSI"].max()
    low = low[["Unit ID"] + list(GAP_KEEP.values()) + ["HBCU"]].rename(
        columns={v: k for k, v in GAP_KEEP.items()}).rename(columns={"HBCU": "HBCU_gapfile"})
    df = df.merge(low, left_on="unit_id", right_on="Unit ID", how="left").drop(columns="Unit ID")
    matched = df["gap_sector"].notna()
    quality["affordability_join"] = {
        "colleges_matched_to_low_income_bracket": int(matched.sum()), "share_matched": float(matched.mean()),
        "sector_code_agreement_among_matched": float((df.loc[matched, "gap_sector"] == df.loc[matched, "sector"]).mean())}
    df["HSI"] = df["unit_id"].map(hsi_any_row).fillna(0).astype(int)
    quality["colleges_in_affordability_file_any_row"] = int(df["unit_id"].isin(hsi_any_row.index).sum())
    both = matched & df["HBCU_gapfile"].notna()
    quality["hbcu_flag_agreement_between_files"] = float((df.loc[both, "HBCU"] == df.loc[both, "HBCU_gapfile"]).mean())
    df = df.drop(columns="HBCU_gapfile")
    for flag in list(MSI_FLAGS_RESULTS) + list(MSI_FLAGS_GAP):
        quality.setdefault("msi_counts", {})[flag] = int(df[flag].sum())

    df["log_earnings"] = np.log(df["earnings_dep_10yr"])
    df.to_parquet(INTERIM / "college_analysis.parquet", index=False)
    quality["analysis_rows"] = int(len(df))
    quality["feature_missing_share"] = {f: float(df[f].isna().mean()) for f in
                                        list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)}
    (RESULTS / "college_data_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps({k: v for k, v in quality.items() if k != "feature_missing_share"}, indent=2))


if __name__ == "__main__":
    main()
