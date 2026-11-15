"""Act 1: Are kids from low-mobility Bay Area neighborhoods far from decent entry-level jobs?

Tests the team's first hypothesis: young adults mostly start work near where they grew up, so if
entry-level pay within reach of a neighborhood is low, its kids should end up with lower adult
incomes. We measure posted entry-level pay within 5/10/20 miles of every Bay Area census tract
(Indeed postings) and check it against the Opportunity Atlas (actual adult household income rank
of kids who grew up in that tract in low-income families).

Outputs: data/interim/indeed_clean.parquet, results/act1_tracts.csv, results/act1_summary.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import BAY_COUNTIES, INTERIM, RESULTS, nearest, within_radius  # noqa: E402

RADII_MI = [5, 10, 20]
OUTCOME = "kfr_pooled_pooled_p25"  # mean adult household income rank, kids from 25th-percentile families

# ---------------------------------------------------------------- salary text -> annual pay
UNITS = [(r"\b(?:hour|hora)\b", 2080), (r"\b(?:year|año|annually|yearly)\b", 1), (r"\b(?:month|mes)\b", 12),
         (r"\b(?:week|semana)\b", 52), (r"\b(?:day|día)\b", 260)]
NON_TIME_UNITS = r"\bper (?:mile|session|class|visit|lesson|appointment|delivery|unit|piece|case)\b"
MONEY = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(k)?\b")


def annual_pay(text):
    """'$25 - $32 an hour' -> 59,280 (midpoint x 2,080 h). Pay per mile/session/etc. -> NaN.

    Hourly, daily and weekly pay are annualized at full-time hours, so this is full-time-equivalent pay.
    """
    if not isinstance(text, str):
        return np.nan
    t = text.lower().replace(",", "")
    if re.search(NON_TIME_UNITS, t):
        return np.nan
    nums = [float(n) * (1000 if k else 1) for n, k in MONEY.findall(t)]
    mult = next((m for pat, m in UNITS if re.search(pat, t)), None)
    if not nums or mult is None:
        return np.nan
    pay = float(np.mean(nums[:2])) * mult
    return pay if 15_000 <= pay <= 600_000 else np.nan


# ---------------------------------------------------------------- flags from the description
DEGREE = r"bachelor|\bb\.?s\.? degree|\bb\.?a\.? degree|4[- ]year degree|four[- ]year degree"
SENIOR = (r"\b(?:senior|sr|lead|principal|manager|director|head|chief|vp|vice president)\b"
          r"|\bstaff (?:engineer|software|scientist|data|product|research|machine)")  # not "kitchen staff"
# Jobs that need a professional or doctoral license are not starting jobs, whatever the posting says.
LICENSED = (r"\b(?:physician|doctor|md|dentist|dds|dmd|orthodontist|pharmacist|pharmd|nurse practitioner|np|fnp"
            r"|pmhnp|physician assistant|pa-c|psychiatrist|psychologist|attorney|lawyer|veterinarian|dvm"
            r"|optometrist|crna|anesthesiologist|surgeon|radiologist|hospitalist|podiatrist|audiologist"
            r"|chiropractor)\b")
# "2+ years of relevant experience"; skips "21 years of age" and "years old"
YEARS = r"(\d{1,2})\+?\s*(?:(?:-|to)\s*\d{1,2}\s*)?\+?\s*years?(?: of)? (?!age\b|old\b)(?:\w+ ){0,3}experience"
MAX_PLAUSIBLE_YEARS = 15  # "over 50 years of experience" is company boilerplate, not a requirement
FAMILIES = {  # checked in order: first match wins
    "healthcare": r"nurse|\bcna\b|medical|dental|pharm|caregiver|clinic|therap|health|patient",
    "food_hospitality": r"cook|chef|server|barista|dishwasher|bartender|host|restaurant|kitchen|housekeep|crew",
    "retail": r"cashier|retail|store|sales associate|merchandis",
    "warehouse_driver": r"warehouse|driver|delivery|forklift|package|logistics|material handler",
    "office_admin": r"admin|receptionist|assistant|coordinator|clerk|customer service|office",
    "education": r"teacher|tutor|instructor|educat|school",
    "tech_engineering": r"engineer|developer|software|data|analyst|scientist",
}


def job_family(title):
    return next((f for f, p in FAMILIES.items() if re.search(p, title)), "other")


def clean_postings():
    jobs = pd.read_parquet(INTERIM / "indeed_raw.parquet",
                           columns=["jobid", "company_name", "job_title", "description_text", "location",
                                    "salary_formatted", "job_type", "date_posted_parsed"])
    n_raw = len(jobs)
    jobs = jobs.drop_duplicates(["job_title", "company_name", "location"]).reset_index(drop=True)
    jobs["pay"] = jobs["salary_formatted"].map(annual_pay)
    desc = jobs["description_text"].fillna("").str.lower()
    title = jobs["job_title"].fillna("").str.lower()
    min_years = desc.str.extract(YEARS, expand=False).astype(float)
    min_years = min_years.where(min_years <= MAX_PLAUSIBLE_YEARS)
    jobs["min_years_exp"] = min_years
    jobs["degree_req"] = desc.str.contains(DEGREE, regex=True)
    jobs["licensed"] = title.str.contains(LICENSED, regex=True)
    jobs["entry"] = (~title.str.contains(SENIOR, regex=True) & ~jobs["licensed"]
                     & (min_years.isna() | (min_years <= 1)))
    jobs["family"] = title.map(job_family)
    # "2788 San Tomas Expressway, Santa Clara, CA 95051" -> "Santa Clara"
    jobs["city"] = jobs["location"].str.extract(r"([^,]+),\s*CA(?:\s+\d{5})?\s*$", expand=False).str.strip()
    jobs.loc[jobs["city"].str.contains(r"County|Bay Area", na=False), "city"] = np.nan
    jobs["city_key"] = jobs["city"].str.lower()
    return jobs, n_raw


# The Gazetteer's "internal point" for San Francisco lies offshore near the Farallon Islands, because the
# city's boundary includes 185 square miles of water and only 47 of land. Left as is, every San Francisco
# posting (the region's largest job centre) would sit 24 miles from the nearest populated tract and fall
# outside every tract's job pool. We use San Francisco City Hall instead. A review found no other
# high-volume Bay Area place placed this way (none is more than about 5 miles from a populated tract).
PLACE_POINT_OVERRIDES = {"san francisco": (37.7793, -122.4193)}


def place_table():
    """One row per place name in California, keyed by lower-case name. Official names with a
    parenthetical short name ("San Buenaventura (Ventura) city") are also listed under the short name.
    A name shared by several places (e.g. Burbank city in LA County and Burbank CDP in Santa Clara
    County) goes to the most populous one."""
    gp = pd.read_parquet(INTERIM / "gaz_places.parquet")
    gp["city"] = gp["NAME"].str.replace(r" (city|town|CDP)$", "", regex=True).str.strip()
    alias = gp[gp["city"].str.contains(r"\(.+\)", regex=True)].copy()
    alias["city"] = alias["city"].str.extract(r"\((.+)\)", expand=False).str.strip()
    gp["city"] = gp["city"].str.replace(r"\s*\(.+\)", "", regex=True).str.strip()
    gp = pd.concat([gp, alias], ignore_index=True)
    gp["city_key"] = gp["city"].str.lower()
    gp = gp.sort_values("POP10", ascending=False).drop_duplicates("city_key").reset_index(drop=True)
    for key, (lat, lon) in PLACE_POINT_OVERRIDES.items():
        assert (gp["city_key"] == key).sum() == 1, key
        gp.loc[gp["city_key"] == key, ["lat", "lon"]] = [lat, lon]
    return gp[["city_key", "POP10", "lat", "lon"]]


def tract_table():
    out = pd.read_parquet(INTERIM / "atlas_outcomes.parquet")
    cov = pd.read_parquet(INTERIM / "atlas_covariates.parquet")
    gt = pd.read_parquet(INTERIM / "gaz_tracts.parquet")
    t = out[(out["state"] == 6) & out["county"].isin(BAY_COUNTIES)][
        ["GEOID", "county", OUTCOME, "pooled_pooled_count"]]
    t = t.merge(cov[["GEOID", "poor_share2010", "frac_coll_plus2010", "med_hhinc2016",
                     "jobs_total_5mi_2015", "jobs_highpay_5mi_2015"]], on="GEOID", how="left")
    t = t.merge(gt[["GEOID", "POP10", "lat", "lon"]], on="GEOID", how="left")
    return t


def tract_job_pools(tracts, postings, places, radius):
    """Entry-level postings (with posted pay) in places within `radius` miles of each tract centroid."""
    by_place = {c: g for c, g in postings.groupby("city_key")}
    plist = places[places["city_key"].isin(by_place)].reset_index(drop=True)
    near = within_radius(tracts[["lat", "lon"]], plist[["lat", "lon"]], radius)
    rows = []
    for idx in near:
        parts = [by_place[c] for c in plist["city_key"].iloc[idx]]
        pool = pd.concat(parts) if parts else postings.iloc[:0]
        nodeg = pool.loc[~pool["degree_req"], "pay"]
        rows.append({f"n_entry_{radius}": len(pool),  # entry-level postings WITH posted pay
                     f"pay_entry_{radius}": pool["pay"].median(),
                     f"pay_entry_nodeg_{radius}": nodeg.median(),
                     f"share_nodeg_{radius}": (len(nodeg) / len(pool)) if len(pool) else np.nan})
    return pd.DataFrame(rows, index=tracts.index)


CLUSTER_DEG = 0.15  # ~10-mile grid cells: neighbouring tracts share almost the same job pool


def incremental_r2(t, measure, weighted=False):
    """Does the measure explain adult outcomes beyond tract poverty and college share?

    Standard errors are clustered on ~10-mile grid cells: job measures are spatially smooth, so
    neighbouring tracts are not independent observations and ordinary p-values would be too small.
    """
    d = t.dropna(subset=[OUTCOME, "poor_share2010", "frac_coll_plus2010", measure]).copy()
    d["m_z"] = (d[measure] - d[measure].mean()) / d[measure].std()
    d["cell"] = (np.floor(d["lat"] / CLUSTER_DEG).astype(int).astype(str) + "_"
                 + np.floor(d["lon"] / CLUSTER_DEG).astype(int).astype(str))
    cluster = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(d["cell"])[0]}}
    if weighted:  # each tract weighted by the number of children behind its Atlas estimate
        d = d[d["pooled_pooled_count"] > 0]
        cluster = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(d["cell"])[0]}}
        fit = lambda f: smf.wls(f, d, weights=d["pooled_pooled_count"]).fit(**cluster)  # noqa: E731
    else:
        fit = lambda f: smf.ols(f, d).fit(**cluster)  # noqa: E731
    base = fit(f"{OUTCOME} ~ poor_share2010 + frac_coll_plus2010")
    full = fit(f"{OUTCOME} ~ poor_share2010 + frac_coll_plus2010 + m_z")
    cv = {}
    for name, cols in [("base", ["poor_share2010", "frac_coll_plus2010"]),
                       ("with_measure", ["poor_share2010", "frac_coll_plus2010", "m_z"])]:
        cv[name] = float(cross_val_score(LinearRegression(), d[cols], d[OUTCOME], groups=d["county"],
                                         cv=GroupKFold(n_splits=len(BAY_COUNTIES)), scoring="r2").mean())
    ci = full.conf_int().loc["m_z"].tolist()
    return {"n_tracts": int(len(d)), "n_clusters": int(d["cell"].nunique()), "r2_base": base.rsquared,
            "r2_with_measure": full.rsquared, "delta_r2": full.rsquared - base.rsquared,
            "coef_per_sd": full.params["m_z"], "coef_ci95_clustered": ci, "p_value_clustered": full.pvalues["m_z"],
            "heldout_county_r2_base": cv["base"], "heldout_county_r2_with_measure": cv["with_measure"]}


def degree_premium(entry_bay):
    """Pay gap between degree-required and no-degree entry-level postings, raw and within job family + city."""
    d = entry_bay.dropna(subset=["pay"]).copy()
    d["log_pay"] = np.log(d["pay"])
    d["deg"] = d["degree_req"].astype(int)
    fit = smf.ols("log_pay ~ deg + C(family) + C(city_key)", d).fit(  # postings from one employer are not independent
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["company_name"].fillna("unknown"))[0]})
    return {"n_postings": int(len(d)), "share_degree_required": float(d["deg"].mean()),
            "median_pay_degree": float(d.loc[d["deg"] == 1, "pay"].median()),
            "median_pay_no_degree": float(d.loc[d["deg"] == 0, "pay"].median()),
            "premium_within_family_city_pct": float(100 * (np.exp(fit.params["deg"]) - 1)),
            "premium_ci95_pct_company_clustered": [float(100 * (np.exp(x) - 1)) for x in fit.conf_int().loc["deg"]]}


def main():
    jobs, n_raw = clean_postings()
    places = place_table()
    tracts = tract_table()

    jobs = jobs.merge(places.rename(columns={"lat": "place_lat", "lon": "place_lon", "POP10": "place_pop"}),
                      on="city_key", how="left")
    # Bay Area posting = its place's nearest tract centroid is in one of the 9 counties
    has_place = jobs["place_lat"].notna()
    all_ca_tracts = pd.read_parquet(INTERIM / "gaz_tracts.parquet")
    idx, dist = nearest(jobs.loc[has_place, ["place_lat", "place_lon"]], all_ca_tracts[["lat", "lon"]])
    county = (all_ca_tracts["GEOID"].to_numpy()[idx] // 10**6) % 1000
    jobs.loc[has_place, "bay_area"] = np.isin(county, list(BAY_COUNTIES)) & (dist < 10)
    jobs["bay_area"] = jobs["bay_area"].fillna(False).astype(bool)
    jobs.to_parquet(INTERIM / "indeed_clean.parquet", index=False)

    entry = jobs[jobs["entry"] & jobs["pay"].notna() & jobs["place_lat"].notna()]
    for r in RADII_MI:
        tracts = tracts.join(tract_job_pools(tracts, entry, places, r))
    tracts.to_csv(RESULTS / "act1_tracts.csv", index=False)

    t = tracts.dropna(subset=[OUTCOME])
    tq = tracts.dropna(subset=[OUTCOME, "pay_entry_10"])  # same tracts and quintiles as figure 1
    q = pd.qcut(tq[OUTCOME], 5, labels=False)
    summary = {
        "postings_raw": int(n_raw),
        "posting_date_min": str(pd.to_datetime(jobs["date_posted_parsed"], errors="coerce", utc=True).min().date()),
        "posting_date_max": str(pd.to_datetime(jobs["date_posted_parsed"], errors="coerce", utc=True).max().date()),
        "postings_after_dedupe": int(len(jobs)),
        "share_with_parseable_pay": float(jobs["pay"].notna().mean()),
        "share_salary_text_present": float(jobs["salary_formatted"].notna().mean()),
        "share_matched_to_place": float(jobs["place_lat"].notna().mean()),
        "bay_area_postings": int(jobs["bay_area"].sum()),
        "bay_area_share_with_pay": float(jobs.loc[jobs["bay_area"], "pay"].notna().mean()),
        "entry_share": float(jobs["entry"].mean()),
        "bay_tracts": int(len(tracts)),
        "bay_tracts_with_outcome": int(len(t)),
        "tracts_with_zero_entry_jobs_10mi": int((tracts["n_entry_10"] == 0).sum()),
        "tracts_in_quintile_comparison": int(len(tq)),
        "median_entry_pay_10mi_by_mobility_quintile": [float(tq.loc[q == k, "pay_entry_10"].median()) for k in range(5)],
        "median_n_entry_10mi_by_mobility_quintile": [float(tq.loc[q == k, "n_entry_10"].median()) for k in range(5)],
        "median_entry_pay_by_mobility_quintile_all_radii": {
            str(r): [float(v) for v in tracts.dropna(subset=[OUTCOME, f"pay_entry_{r}"]).pipe(
                lambda x: x.groupby(pd.qcut(x[OUTCOME], 5, labels=False))[f"pay_entry_{r}"].median())]
            for r in RADII_MI},
        "licensed_titles_excluded_from_entry": int(jobs["licensed"].sum()),
        "spearman_with_outcome": {},
        "incremental": {},
        "degree_premium_bay_entry": degree_premium(jobs[jobs["bay_area"] & jobs["entry"]]),
    }
    measures = [f"{m}_{r}" for r in RADII_MI for m in ("pay_entry", "pay_entry_nodeg", "n_entry")] + \
               ["poor_share2010", "frac_coll_plus2010", "med_hhinc2016", "jobs_highpay_5mi_2015",
                "jobs_total_5mi_2015"]
    for m in measures:
        rho, _ = spearmanr(t[m], t[OUTCOME], nan_policy="omit")  # no p-value: tracts are spatially dependent
        summary["spearman_with_outcome"][m] = {"rho": float(rho)}
    for m in [f"pay_entry_{r}" for r in RADII_MI] + [f"pay_entry_nodeg_{r}" for r in RADII_MI] + \
             ["jobs_highpay_5mi_2015"]:
        for weighted, key in [(False, m), (True, f"{m}__weighted_by_children")]:
            summary["incremental"][key] = {k: (float(v) if not isinstance(v, list) else [float(x) for x in v])
                                           for k, v in incremental_r2(tracts, m, weighted).items()}
    (RESULTS / "act1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("spearman_with_outcome", "incremental")},
                     indent=2))
    for m in ["pay_entry_10", "pay_entry_nodeg_10", "jobs_highpay_5mi_2015"]:
        print(m, summary["spearman_with_outcome"][m], {k: round(v, 4) if isinstance(v, float) else v
                                                      for k, v in summary["incremental"][m].items()})


if __name__ == "__main__":
    main()
