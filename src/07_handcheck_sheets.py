"""Step 7: write the sheets for checking our data decisions BY HAND (the rubric requires a manual check).

Each sheet lists a random sample (fixed seed) with what our pipeline produced, plus empty columns
for the person checking. Instructions are in handcheck/README.md.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import HANDCHECK, INTERIM, RESULTS, SEED  # noqa: E402


def write(frame, name):
    """Never overwrite a sheet the team may have filled in: write only if it does not exist yet."""
    path = HANDCHECK / name
    if path.exists():
        print(f"kept existing {name} (delete it to regenerate a blank sheet)")
        return
    frame.to_csv(path, index=False)
    print("wrote", name)


def main():
    # 1. College join + target: compare against the two raw Excel files, row by row
    c = pd.read_parquet(INTERIM / "college_analysis.parquet")
    s = c.sample(20, random_state=SEED)[["unit_id", "name", "state", "earnings_dep_10yr", "pell_share",
                                          "net_price_low_income", "HBCU", "HSI"]]
    s = s.assign(earnings_matches_excel="", pell_matches_excel="", net_price_matches_excel="",
                 msi_flags_match="", notes="")
    write(s, "college_rows_20.csv")

    # 2. Indeed parsing: salary text -> annual pay, location -> place, flags
    j = pd.read_parquet(INTERIM / "indeed_clean.parquet")
    p = j[j["bay_area"]].sample(30, random_state=SEED)[["jobid", "job_title", "company_name", "location",
                                                         "salary_formatted", "pay", "city", "entry",
                                                         "degree_req", "min_years_exp"]]
    p = p.assign(pay_correct="", city_correct="", entry_correct="", degree_correct="", notes="")
    write(p, "postings_30.csv")

    # 3. Tract coordinates: is each tract's point inside the right county?
    t = pd.read_csv(RESULTS / "act1_tracts.csv").sample(10, random_state=SEED)[["GEOID", "county", "lat", "lon"]]
    t = t.assign(google_maps_link=[f"https://www.google.com/maps?q={a},{b}" for a, b in zip(t["lat"], t["lon"])],
                 inside_named_county="", notes="")
    write(t, "tracts_10.csv")

    # 4. "Also consider" pairs (a label we created): is the suggested college a fair alternative?
    b = pd.read_csv(RESULTS / "better_options.csv")
    k = b.sample(min(15, len(b)), random_state=SEED)[["unit_id", "name", "city", "state", "level", "option_unit_id",
                                                     "option_name", "option_city", "option_state", "miles"]]
    k = k.assign(same_degree_level="", still_open="", coed_and_in_person="", similar_programs="", notes="")
    write(k, "also_consider_pairs_15.csv")


if __name__ == "__main__":
    main()
