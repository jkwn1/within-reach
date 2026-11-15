"""Step 0: read each raw file once, keep only the columns the project uses, save as Parquet.

The raw exports are large (Indeed 2.1 GB, Zillow 3 parts ~7 GB) and carry many bulky
nested-JSON columns we never use, so everything later reads these small Parquet files.
"""
import re
import subprocess
import time

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import INTERIM, RAW  # noqa: E402


def read_csv_columns(path, usecols, dtype=str, chunksize=50_000):
    """Read only `usecols` from a big CSV in chunks (keeps memory low)."""
    parts = [chunk for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize)]
    return pd.concat(parts, ignore_index=True)


def ingest_indeed():
    cols = ["jobid", "company_name", "date_posted_parsed", "job_title", "description_text", "benefits",
            "qualifications", "job_type", "location", "job_location", "salary_formatted", "company_rating",
            "company_reviews_count", "shift_schedule", "discovery_input"]
    df = read_csv_columns(RAW / "indeed_job_listings.csv", cols)
    for c in ["company_rating", "company_reviews_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(INTERIM / "indeed_raw.parquet", index=False)
    return len(df)


def ingest_zillow():
    cols = ["zpid", "city", "state", "county", "countyFIPS", "zipcode", "homeStatus", "homeType", "price",
            "bedrooms", "bathrooms", "livingArea", "yearBuilt", "latitude", "longitude", "zestimate",
            "rentZestimate", "lastSoldPrice", "dateSold", "daysOnZillow"]
    parts = []
    for i in (1, 2, 3):
        part = read_csv_columns(RAW / f"zillow_part{i}.csv", cols)
        part["source_part"] = i
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    for c in ["price", "bedrooms", "bathrooms", "livingArea", "yearBuilt", "latitude", "longitude",
              "zestimate", "rentZestimate", "lastSoldPrice", "daysOnZillow"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_before = len(df)
    df = df.drop_duplicates("zpid", keep="first")
    df.to_parquet(INTERIM / "zillow.parquet", index=False)
    return n_before, len(df)


def ingest_places():
    cols = ["place_id", "name", "category", "all_categories", "address", "lat", "lon", "rating",
            "reviews_count", "permanently_closed", "temporarily_closed", "open_hours", "is_claimed",
            "open_website", "phone_number"]
    df = read_csv_columns(RAW / "google_maps.csv", cols)
    for c in ["lat", "lon", "rating", "reviews_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(INTERIM / "places.parquet", index=False)
    return len(df)


def ingest_reviews():
    cols = ["place_id", "place_name", "address", "review_id", "review_rating", "review", "review_date",
            "number_of_likes", "response_of_owner", "response_date", "local_guide", "reviews_by_reviewer",
            "place_general_rating", "overall_place_riviews"]
    df = read_csv_columns(RAW / "google_maps_reviews.csv", cols)
    for c in ["review_rating", "number_of_likes", "reviews_by_reviewer", "place_general_rating",
              "overall_place_riviews"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(INTERIM / "reviews.parquet", index=False)
    return len(df)


def ingest_college():
    results = pd.read_excel(RAW / "college_results_2021.xlsx", sheet_name=0)
    gap = pd.read_excel(RAW / "affordability_gap_2022_23.xlsx", sheet_name=0)
    # Excel cells hold "." for missing in places; force every column to numeric where possible.
    for df in (results, gap):
        for c in df.columns:
            if df[c].dtype == object:
                conv = pd.to_numeric(df[c].replace({".": None}), errors="coerce")
                if conv.notna().sum() >= 0.5 * df[c].notna().sum() and conv.notna().any():
                    df[c] = conv
                else:  # mixed text/number columns (e.g. zip codes) -> plain strings
                    df[c] = df[c].map(lambda x: None if pd.isna(x) else str(x))
    results.columns = [str(c).strip() for c in results.columns]
    gap.columns = [str(c).strip() for c in gap.columns]
    results.to_parquet(INTERIM / "college_results.parquet", index=False)
    gap.to_parquet(INTERIM / "college_gap.parquet", index=False)
    return len(results), len(gap)


def ingest_atlas():
    out = pd.read_csv(RAW / "tract_outcomes_simple.csv")
    cov = pd.read_csv(RAW / "tract_covariates.csv")
    for df in (out, cov):
        df["GEOID"] = df["state"].astype("int64") * 10**9 + df["county"].astype("int64") * 10**6 \
            + df["tract"].astype("int64")
    out.to_parquet(INTERIM / "atlas_outcomes.parquet", index=False)
    cov.to_parquet(INTERIM / "atlas_covariates.parquet", index=False)
    return len(out), len(cov)


def gazetteer_text(stem):
    """Text of a Census Gazetteer list: the original .txt if present, else text extracted from a PDF print."""
    txt = RAW / f"{stem}.txt"
    if txt.exists():
        return txt.read_text(errors="replace")
    pdf = next(RAW.glob(f"*{stem}*.pdf"))
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True,
                          check=True).stdout


def parse_gazetteer_tracts(text):
    """Records = 'CA <11-digit GEOID> <POP10> <HU10> ... <INTPTLAT> <INTPTLONG>'.

    Works on the original tab-delimited file and on text extracted from a printed copy (where a
    record can wrap across lines): coordinates are the first 7-decimal lat/lon pair after the GEOID
    and before the next record starts.
    """
    rec = re.compile(r"\bCA\s+(06\d{9})\s+(\d+)\s+(\d+)")
    coord = re.compile(r"(-?\d{1,2}\.\d{7})\s+(-\d{2,3}\.\d{7})")
    starts = list(rec.finditer(text))
    rows = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        c = coord.search(text, m.end(), end)
        if c:
            rows.append((int(m.group(1)), int(m.group(2)), float(c.group(1)), float(c.group(2))))
    df = pd.DataFrame(rows, columns=["GEOID", "POP10", "lat", "lon"]).drop_duplicates("GEOID")
    return df


def parse_gazetteer_places(text):
    """Records = 'CA <7-digit GEOID> <ANSICODE> <NAME> <LSAD> <FUNCSTAT> <POP10> ... <lat> <lon>'."""
    rec = re.compile(r"\bCA\s+(06\d{5})\s+(\d{8})\s+(.+?)\s+(\d{2})\s+([A-Z])\s+(\d+)")
    coord = re.compile(r"(-?\d{1,2}\.\d{6})\s+(-\d{2,3}\.\d{6})")
    starts = list(rec.finditer(text))
    rows = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        c = coord.search(text, m.end(), end)
        if c:
            rows.append((m.group(1), re.sub(r"\s+", " ", m.group(3)).strip(), int(m.group(6)),
                         float(c.group(1)), float(c.group(2))))
    return pd.DataFrame(rows, columns=["place_geoid", "NAME", "POP10", "lat", "lon"]).drop_duplicates("place_geoid")


def ingest_gazetteer():
    tracts = parse_gazetteer_tracts(gazetteer_text("census_tracts_list_06"))
    places = parse_gazetteer_places(gazetteer_text("2010_place_list_06"))
    tracts.to_parquet(INTERIM / "gaz_tracts.parquet", index=False)
    places.to_parquet(INTERIM / "gaz_places.parquet", index=False)
    return len(tracts), len(places)


if __name__ == "__main__":
    import sys
    steps = {"gazetteer": ingest_gazetteer, "atlas": ingest_atlas, "college": ingest_college,
             "places": ingest_places, "reviews": ingest_reviews, "indeed": ingest_indeed,
             "zillow": ingest_zillow}
    chosen = sys.argv[1:] or list(steps)
    for name in chosen:
        t0 = time.time()
        print(f"{name}: {steps[name]()} rows ({time.time() - t0:.0f}s)", flush=True)
