"""Shared paths, constants and helpers for every pipeline step."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
HANDCHECK = ROOT / "handcheck"
for _d in (INTERIM, RESULTS, FIGURES, HANDCHECK):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# The 9 Bay Area counties (state FIPS 6), as listed in the datathon README.
BAY_COUNTIES = {
    1: "Alameda", 13: "Contra Costa", 41: "Marin", 55: "Napa", 75: "San Francisco",
    81: "San Mateo", 85: "Santa Clara", 95: "Solano", 97: "Sonoma",
}

EARTH_RADIUS_MI = 3958.8


def within_radius(src_latlon, dst_latlon, radius_mi):
    """For each source point, the indices of destination points within radius_mi (haversine)."""
    from sklearn.neighbors import BallTree

    tree = BallTree(np.radians(np.asarray(dst_latlon, dtype=float)), metric="haversine")
    return tree.query_radius(np.radians(np.asarray(src_latlon, dtype=float)), r=radius_mi / EARTH_RADIUS_MI)


def nearest(src_latlon, dst_latlon):
    """Index of, and distance in miles to, the nearest destination point for each source point."""
    from sklearn.neighbors import BallTree

    tree = BallTree(np.radians(np.asarray(dst_latlon, dtype=float)), metric="haversine")
    dist, idx = tree.query(np.radians(np.asarray(src_latlon, dtype=float)), k=1)
    return idx[:, 0], dist[:, 0] * EARTH_RADIUS_MI

# College model inputs (built in 02_college_prepare.py). They describe who enrolls and where the
# college is, never what it produces (graduation, debt) or spends.
NUMERIC_FEATURES = ["pell_share", "pct_black", "pct_latino", "pct_asian", "pct_white", "pct_aian", "pct_nhpi",
                    "pct_two_plus", "pct_women", "pct_age25plus", "pct_part_time", "pct_distance_only", "pct_ftft",
                    "log_enrollment", "admit_rate", "sat_math_25", "sat_erw_25", "act_math_25", "open_admissions"]
CATEGORICAL_FEATURES = ["sector", "carnegie", "urbanization", "size_category", "state"]
RACE_FEATURES = ["pct_black", "pct_latino", "pct_asian", "pct_white", "pct_aian", "pct_nhpi", "pct_two_plus"]
MSI_TYPES = ["HBCU", "HSI", "TCU", "PBI", "AANAPISI"]
