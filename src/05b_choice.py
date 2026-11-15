"""Step 5b: what value-added means for a low-income student choosing a college.

All on the ranked colleges from step 5 (500+ students, own earnings figure, net price and Pell share known).

1. Similar students, different results. "High-Pell" colleges: Pell share at or above the median of ranked
   colleges. Among those, "similar students" means the model expects earnings within 10% of the high-Pell
   median (expected earnings summarize everything we know about who enrolls). We compare median earnings by
   value-added quarter, overall and at larger colleges (less noisy earnings figures). Prices are reported
   as they are: they are close overall but not at larger colleges.

2. Could a family see the difference? Correlation of price with raw earnings vs with value-added; how often
   the highest-earning college in a student's local, affordable choice set is also the highest value-added
   one; who makes the top 20% by raw earnings.

3. "Also consider" options within reach. For each starting college, other ranked colleges that are
     * within 25 miles (straight line; TICAS/Hillman 2023: the median community-college student enrolls
       10 miles from home and 79% within 25 miles),
     * the same kind of institution: a degree-granting college is never matched with a technical centre or
       career school that has no Carnegie classification (and the reverse),
     * the same level: bachelor's (50+ bachelor's degrees a year; the file's 2000 Carnegie codes are too old
       to tell, e.g. colleges that became 4-year after 2000), 2-year, or under-2-year,
     * expected to have similar students: expected earnings within 10%,
     * at least as open to low-income students: Pell share no more than 10 points lower,
     * no more expensive for families earning up to $30,000,
     * no harder to get into: open admissions, or an admission rate no more than 10 points lower,
     * clearly stronger: value-added at least 10 points higher, with AND without race inputs.
   Safeguards found in review: tribal colleges (the model is biased for them), specialized colleges,
   online-by-design colleges and colleges known to have closed or merged since the data were collected are
   never starting points or options; single-gender colleges are never options. "Comparable" options meet
   every rule except the value-added one. The tool shows the nearest qualifying option first.

4. Does it hold up? Independent-student earnings (mostly older students) are used neither by the model nor
   by the rules. For each qualifying pair we check whether independent students also earn more at the
   suggested college, against a baseline: the nearest comparable college without the value-added rule.

Reach is counted in first-time, full-time Pell students (an exact count in College Results), never as
dollars x students x years.

Outputs: results/choice_summary.json, results/better_options.csv (one row per starting college with a
stronger option: the nearest one), results/better_option_pairs.csv (every qualifying pair),
results/finder_data.json (for the web page built in step 9).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import EARTH_RADIUS_MI, INTERIM, MSI_TYPES, RESULTS  # noqa: E402

SIMILAR_BAND = 0.10
RADIUS_MI = 25.0
PELL_SLACK_POINTS = 10.0
ADMIT_SLACK_POINTS = 10.0
VA_GAP_POINTS = 10.0
MIN_BACHELORS = 50
ID = "UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"
# Hand-made lists (checked against the colleges that appeared in candidate pairs). Online-by-design:
# a distance-education share can't be used because the intake data cover a pandemic year.
ONLINE_BY_DESIGN = ["Rio Salado College", "Capella University", "American Public University System",
                    "Western Governors University", "University of Phoenix", "Purdue University Global",
                    "Walden University", "University of Arizona Global Campus", "Excelsior",
                    "Charter Oak State College", "Thomas Edison State University", "American InterContinental",
                    "Colorado Technical University", "Southern New Hampshire University", "Ashford University"]
SPECIALIZED_BY_NAME = ["Neumont College of Computer Science", "American Musical and Dramatic Academy",
                       "Atlantic University College"]
CLOSED_OR_MERGED = ["Cabrini University", "Notre Dame College", "Bloomfield College", "California College of the Arts",
                    "Birmingham-Southern College", "University of the Arts", "Eastern Nazarene College", "Wells College",
                    "Fontbonne University", "Limestone University", "Union Institute", "Cardinal Stritch University",
                    "Presentation College", "Iowa Wesleyan University", "Medaille", "Finlandia University",
                    "Holy Names University", "Alderson Broaddus University", "Lincoln College", "Marymount California",
                    "San Francisco Art Institute", "Mills College", "The King's College", "College of Saint Rose",
                    "Northland College", "Goddard College", "Pittsburgh Technical College", "Bacone College",
                    "University of Saint Katherine", "The College of Saint Rose", "Medaille University",
                    "Medaille College", "Union Institute & University", "Marymount California University",
                    "Notre Dame College of Ohio"]


def name_flag(names, patterns):
    low = names.str.lower()
    return np.logical_or.reduce([low.str.contains(p.lower(), regex=False) for p in patterns])


def load():
    r = pd.read_parquet(RESULTS / "ranked_colleges.parquet")
    intake = pd.read_parquet(INTERIM / "college_analysis.parquet")[
        ["unit_id", "admit_rate", "open_admissions", "carnegie", "pct_women", "earnings_indep_10yr"]]
    cr = pd.read_parquet(INTERIM / "college_results.parquet")
    extra = pd.DataFrame({
        "unit_id": pd.to_numeric(cr[ID], errors="coerce"),
        "pell_ftft_count": pd.to_numeric(cr["Number of First-Time, Full-Time Undergraduates Awarded Pell Grants"], errors="coerce"),
        "bachelors": pd.to_numeric(cr["Number of Bachelor Degrees Grand Total"], errors="coerce"),
        "pell_completion_6yr": pd.to_numeric(cr["Percent Full-time, First-time, Pell Grant Recipients Receiving an Award - 6 Years"], errors="coerce")})
    extra = extra.dropna(subset=["unit_id"]).drop_duplicates("unit_id")
    extra["unit_id"] = extra["unit_id"].astype("int64")
    r = (r.merge(intake, on="unit_id", how="left", validate="one_to_one")
          .merge(extra, on="unit_id", how="left", validate="one_to_one").reset_index(drop=True))
    sec = r["sector"].astype(str)
    four = r["bachelors"].fillna(0) >= MIN_BACHELORS  # the 2000 Carnegie codes are too old to tell
    r["level"] = np.where(sec.isin(["7", "8", "9"]), "Under 2-year", np.where(four, "4-year", "2-year"))
    r["online_by_design"] = name_flag(r["name"], ONLINE_BY_DESIGN)
    r["closed_or_merged"] = r["name"].str.lower().isin([n.lower() for n in CLOSED_OR_MERGED])
    r["specialized"] = (r["carnegie"].astype(str) == "50") | name_flag(r["name"], SPECIALIZED_BY_NAME)
    r["tribal"] = r["TCU"].fillna(0).astype(int) == 1
    r["single_gender"] = (r["pct_women"] >= 95) | (r["pct_women"] <= 5)
    r["not_compared"] = r["online_by_design"] | r["closed_or_merged"] | r["specialized"] | r["tribal"]
    return r


def quarter_table(group):
    g = group.copy()
    g["quarter"] = pd.qcut(g["value_added_pct"], 4, labels=["worst", "second", "third", "best"])
    t = g.groupby("quarter", observed=True).agg(
        n=("unit_id", "size"), median_earnings=("earnings_dep_10yr", "median"),
        median_expected=("expected_earnings", "median"), median_net_price=("net_price_low_income", "median"),
        median_value_added_pct=("value_added_pct", "median"))
    return {q: {k: (int(v) if k == "n" else round(float(v), 1)) for k, v in row.items()} for q, row in t.iterrows()}


def spread(group):
    t = quarter_table(group)
    return {"n": int(len(group)), "quarters": t,
            "gap_best_vs_worst_usd": round(t["best"]["median_earnings"] - t["worst"]["median_earnings"]),
            "price_best_minus_worst_usd": round(t["best"]["median_net_price"] - t["worst"]["median_net_price"])}


def haversine_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rule_masks(r, radius=RADIUS_MI, va_gap=VA_GAP_POINTS):
    """Boolean matrices [start, option]: `comparable` (every rule but value-added) and `stronger`."""
    dist = haversine_matrix(r["lat"].to_numpy(float), r["lon"].to_numpy(float))
    exp = r["expected_earnings"].to_numpy(float)
    pell = r["pell_share"].to_numpy(float)
    price = r["net_price_low_income"].to_numpy(float)
    level = r["level"].to_numpy()
    is_open = (r["open_admissions"] == 1).to_numpy()
    admit = r["admit_rate"].to_numpy(float)
    nc = r["not_compared"].to_numpy()
    sg = r["single_gender"].to_numpy()
    # No Carnegie 2000 code and below bachelor's level: mostly technical centres and career schools. (Colleges
    # founded or reclassified after 2000, e.g. UC Merced, also lack a code, but they are bachelor's level.)
    unclassified = ((r["carnegie"].astype(str) == "-2") & (r["level"] != "4-year")).to_numpy()
    # NaN comparisons are False, so an open-admissions start only accepts open-admissions options.
    access = is_open[None, :] | (~is_open[:, None] & (admit[None, :] >= admit[:, None] - ADMIT_SLACK_POINTS))
    comparable = ((dist <= radius) & (level[:, None] == level[None, :]) & access
                  & (unclassified[:, None] == unclassified[None, :])
                  & (np.abs(exp[None, :] / exp[:, None] - 1) <= SIMILAR_BAND)
                  & (pell[None, :] >= pell[:, None] - PELL_SLACK_POINTS)
                  & (price[None, :] <= price[:, None])
                  & ~nc[:, None] & ~nc[None, :] & ~sg[None, :])
    np.fill_diagonal(comparable, False)
    va = r["value_added_pct"].to_numpy(float)
    va_nr = r["value_added_no_race_pct"].to_numpy(float)
    stronger = comparable & (va[None, :] >= va[:, None] + va_gap) & (va_nr[None, :] >= va_nr[:, None] + va_gap)
    same_level_near = ((dist <= radius) & (level[:, None] == level[None, :]) & ~nc[None, :])
    np.fill_diagonal(same_level_near, False)
    return dist, comparable, stronger, same_level_near


def nearest_pairs(dist, mask):
    i, j = np.nonzero(mask)
    p = pd.DataFrame({"from_idx": i, "to_idx": j, "miles": dist[i, j]}).sort_values(["from_idx", "miles"])
    return p, p.drop_duplicates("from_idx")


def reach(r, starts, idx_with):
    keep = starts.intersection(idx_with)
    return {"colleges": int(len(keep)), "first_time_pell_students": int(r.loc[keep, "pell_ftft_count"].sum())}


def replication(r, pairs, base, ci=False):
    """Share of pairs where independent-student earnings are also higher at the option. With ci=True, a 95%
    interval from a bootstrap that resamples suggested colleges (pairs sharing an option are not independent)."""
    def share(p):
        a = r["earnings_indep_10yr"].to_numpy(float)[p["from_idx"].to_numpy()]
        b = r["earnings_indep_10yr"].to_numpy(float)[p["to_idx"].to_numpy()]
        ok = ~np.isnan(a) & ~np.isnan(b)
        hit = (b[ok] > a[ok]).astype(float)
        out = {"all_pairs": int(len(p)), "pairs": int(ok.sum()), "starting_colleges": int(p["from_idx"].to_numpy()[ok].size and
                                                               np.unique(p["from_idx"].to_numpy()[ok]).size),
               "suggested_colleges": int(np.unique(p["to_idx"].to_numpy()[ok]).size),
               "share_higher": round(float(hit.mean()), 3),
               "median_gap_pct": round(float(np.median((b[ok] / a[ok] - 1) * 100)), 1)}
        if ci:
            cl = p["to_idx"].to_numpy()[ok]
            groups = [hit[cl == c] for c in np.unique(cl)]
            rng = np.random.default_rng(42)
            draws = []
            for _ in range(2000):
                pick = rng.integers(0, len(groups), len(groups))
                g = np.concatenate([groups[i] for i in pick])
                draws.append(g.mean())
            out["ci95_cluster_bootstrap"] = [round(float(np.percentile(draws, 2.5)), 3),
                                             round(float(np.percentile(draws, 97.5)), 3)]
        return out
    return {"stronger_option_pairs": share(pairs), "baseline_nearest_comparable": share(base)}


def replication_same_start(r, dist, comparable, stronger):
    """Fair version: for each starting college that has both a stronger option and a comparable college that is
    not stronger, compare the nearest of each on independent-student earnings. Bootstrap over starting colleges."""
    ind = r["earnings_indep_10yr"].to_numpy(float)
    weaker = comparable & ~stronger
    hit_s, hit_c = [], []
    for i in np.nonzero(stronger.any(axis=1) & weaker.any(axis=1))[0]:
        js = np.nonzero(stronger[i])[0]; jc = np.nonzero(weaker[i])[0]
        j1, j2 = js[np.argmin(dist[i, js])], jc[np.argmin(dist[i, jc])]
        if np.isnan(ind[i]) or np.isnan(ind[j1]) or np.isnan(ind[j2]):
            continue
        hit_s.append(ind[j1] > ind[i]); hit_c.append(ind[j2] > ind[i])
    hs, hc = np.array(hit_s, float), np.array(hit_c, float)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        k = rng.integers(0, len(hs), len(hs))
        diffs.append(hs[k].mean() - hc[k].mean())
    # Benchmark: across all comparable pairs, how often do independent earnings move with dependent earnings?
    dep = r["earnings_dep_10yr"].to_numpy(float)
    i, j = np.nonzero(comparable)
    ok = ~np.isnan(ind[i]) & ~np.isnan(ind[j]) & (dep[j] > dep[i])
    return {"starting_colleges": int(len(hs)), "nearest_stronger_share_higher": round(float(hs.mean()), 3),
            "nearest_other_comparable_share_higher": round(float(hc.mean()), 3),
            "difference_ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
            "benchmark_dep_higher_then_indep_higher": round(float((ind[j][ok] > ind[i][ok]).mean()), 3),
            "benchmark_pairs": int(ok.sum())}


def option_rows(r, best):
    a = r.iloc[best["from_idx"].to_numpy()].reset_index(drop=True)
    b = r.iloc[best["to_idx"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "unit_id": a["unit_id"], "name": a["name"], "city": a["city"], "state": a["state"], "level": a["level"],
        "pell_share": a["pell_share"], "first_time_pell_students": a["pell_ftft_count"],
        "net_price_low_income": a["net_price_low_income"], "earnings_dep_10yr": a["earnings_dep_10yr"],
        "expected_earnings": a["expected_earnings"].round(0), "value_added_pct": a["value_added_pct"].round(1),
        "option_unit_id": b["unit_id"], "option_name": b["name"], "option_city": b["city"], "option_state": b["state"],
        "option_pell_share": b["pell_share"], "option_net_price_low_income": b["net_price_low_income"],
        "option_earnings_dep_10yr": b["earnings_dep_10yr"], "option_expected_earnings": b["expected_earnings"].round(0),
        "option_value_added_pct": b["value_added_pct"].round(1), "miles": best["miles"].round(1).to_numpy(),
        "earnings_difference_usd": (b["earnings_dep_10yr"] - a["earnings_dep_10yr"]).to_numpy(),
        "price_difference_usd": (b["net_price_low_income"] - a["net_price_low_income"]).to_numpy()})
    for m in MSI_TYPES:
        out[m] = a[m].fillna(0).astype(int).to_numpy()
        out["option_" + m] = b[m].fillna(0).astype(int).to_numpy()
    return out


def main():
    r = load()
    pell_median = float(r["pell_share"].median())
    high_pell = (r["pell_share"] >= pell_median).to_numpy()
    hp = r[high_pell]
    exp_median = float(hp["expected_earnings"].median())
    similar = hp[(hp["expected_earnings"] - exp_median).abs() <= SIMILAR_BAND * exp_median]

    summary = {"ranked_colleges": int(len(r)), "pell_median": pell_median, "high_pell_colleges": int(high_pell.sum()),
               "similar_students": {"definition": f"high-Pell colleges with expected earnings within {SIMILAR_BAND:.0%} "
                                                  f"of the high-Pell median", "expected_median_usd": round(exp_median),
                                    "all": spread(similar)}}
    for min_enr in [2000, 5000]:
        summary["similar_students"][f"enrollment_{min_enr}_plus"] = spread(similar[similar["enrollment"] >= min_enr])
    for lev in ["4-year", "2-year"]:
        summary["similar_students"][f"level_{lev}"] = spread(similar[similar["level"] == lev])

    def sp(d, a, b):
        return round(float(d[[a, b]].corr("spearman").iloc[0, 1]), 3)
    summary["price_correlation_high_pell"] = {
        "with_value_added": sp(hp, "value_added_pct", "net_price_low_income"),
        "with_raw_earnings": sp(hp, "earnings_dep_10yr", "net_price_low_income")}
    n_top = int(round(0.2 * len(r)))
    top_raw = r.nlargest(n_top, "earnings_dep_10yr")
    summary["top_20_percent_by_raw_earnings"] = {
        "n": n_top, "high_pell_share": round(float((top_raw["pell_share"] >= pell_median).mean()), 3),
        "high_pell_share_all_ranked": round(float(high_pell.mean()), 3),
        "median_net_price_usd": round(float(top_raw["net_price_low_income"].median())),
        "median_net_price_all_ranked_usd": round(float(r["net_price_low_income"].median()))}

    dist, comparable, stronger, near = rule_masks(r)
    # Local choice sets: a high-Pell starting college plus every same-level college within reach at the same or
    # lower price (other rules off). How often is the highest raw earner also the highest value-added one?
    price = r["net_price_low_income"].to_numpy(float)
    earn = r["earnings_dep_10yr"].to_numpy(float)
    va_all = r["value_added_pct"].to_numpy(float)
    nc_all = r["not_compared"].to_numpy()
    agree, sets = 0, 0
    for i in np.nonzero(high_pell & ~nc_all)[0]:
        members = np.r_[i, np.nonzero(near[i] & (price <= price[i]))[0]]
        if len(members) < 2:
            continue
        sets += 1
        agree += int(members[np.argmax(earn[members])] == members[np.argmax(va_all[members])])
    summary["local_choice_sets"] = {"sets": sets, "top_raw_is_top_value_added": round(agree / sets, 3)}

    pairs, best = nearest_pairs(dist, stronger)
    base_all, base = nearest_pairs(dist, comparable)
    starts_idx = pd.Index(np.nonzero(high_pell & ~nc_all)[0])
    with_stronger = pd.Index(best["from_idx"])
    with_comparable = pd.Index(base["from_idx"])
    with_any_near = pd.Index(np.nonzero(near.any(axis=1))[0])
    rows = option_rows(r, best)
    hp_rows = rows[rows["unit_id"].isin(r.loc[starts_idx, "unit_id"])]
    msi_any = hp_rows[MSI_TYPES].sum(axis=1) > 0
    hbcu_pbi = (hp_rows["HBCU"] + hp_rows["PBI"]) > 0
    summary["also_consider"] = {
        "rule": {"radius_miles": RADIUS_MI,
                 "level": f"bachelor's ({MIN_BACHELORS}+ bachelor's degrees a year) / 2-year / under 2-year",
                 "expected_earnings_band": SIMILAR_BAND, "pell_share_at_most_lower_by_points": PELL_SLACK_POINTS,
                 "net_price_no_higher": True, "admit_rate_at_most_lower_by_points_or_open": ADMIT_SLACK_POINTS,
                 "value_added_gap_points_both_versions": VA_GAP_POINTS,
                 "never_compared": "tribal, specialized, online-by-design, closed or merged; single-gender never an option"},
        "high_pell_starting_colleges": int(len(starts_idx)),
        "excluded_high_pell_not_compared": int(high_pell.sum() - len(starts_idx)),
        "stronger_option": reach(r, starts_idx, with_stronger),
        "comparable_but_none_stronger": reach(r, starts_idx, with_comparable.difference(with_stronger)),
        "no_comparable_option": reach(r, starts_idx, starts_idx.difference(with_comparable)),
        "no_same_level_college_within_25mi": reach(r, starts_idx, starts_idx.difference(with_any_near)),
        "median_earnings_difference_usd": round(float(hp_rows["earnings_difference_usd"].median())),
        "median_price_difference_usd": round(float(hp_rows["price_difference_usd"].median())),
        "median_miles": round(float(hp_rows["miles"].median()), 1),
        "minority_serving_starting_with_option": int(msi_any.sum()),
        "option_is_minority_serving": int((hp_rows[["option_" + m for m in MSI_TYPES]].sum(axis=1) > 0).sum()),
        "hbcu_or_pbi_starting_with_option": int(hbcu_pbi.sum()),
        "hbcu_or_pbi_pointed_to_non_hbcu_pbi": int((hbcu_pbi & ((hp_rows["option_HBCU"] + hp_rows["option_PBI"]) == 0)).sum()),
        "replication_independent_students": replication(r, pairs, base, ci=True),
        "replication_same_start": replication_same_start(r, dist, comparable, stronger),
        "distinct_suggested_colleges": int(hp_rows["option_unit_id"].nunique()),
        "distinct_suggested_minority_serving": int(hp_rows.drop_duplicates("option_unit_id")[["option_" + m for m in MSI_TYPES]].sum(axis=1).gt(0).sum()),
        "all_ranked_with_stronger_option": int(len(with_stronger)),
    }
    cr = pd.read_parquet(INTERIM / "college_results.parquet")
    pell_total_all = float(pd.to_numeric(cr["Number of First-Time, Full-Time Undergraduates Awarded Pell Grants"],
                                         errors="coerce").sum())
    pell_total_ranked = float(r["pell_ftft_count"].sum())
    summary["coverage_first_time_pell_students"] = {"ranked": round(pell_total_ranked), "all_colleges": round(pell_total_all),
                                                    "share": round(pell_total_ranked / pell_total_all, 3)}
    sens = {}
    for radius in [15.0, 25.0, 50.0]:
        for gap in [5.0, 10.0, 15.0]:
            d2, cmp2, st2, _ = rule_masks(r, radius, gap)
            p2, b2 = nearest_pairs(d2, st2)
            _, base2 = nearest_pairs(d2, cmp2)
            sens[f"{int(radius)}mi_{int(gap)}pts"] = {
                "high_pell_with_stronger_option": int(b2["from_idx"].isin(starts_idx).sum()),
                "replication_share": replication(r, p2, base2)["stronger_option_pairs"]["share_higher"]}
    summary["also_consider"]["sensitivity"] = sens
    ex = hp_rows.sort_values("first_time_pell_students", ascending=False)
    summary["also_consider"]["largest_examples"] = ex.head(8)[
        ["name", "city", "state", "option_name", "option_city", "miles", "net_price_low_income",
         "option_net_price_low_income", "earnings_dep_10yr", "option_earnings_dep_10yr", "value_added_pct",
         "option_value_added_pct", "first_time_pell_students"]].to_dict("records")

    rows.to_csv(RESULTS / "better_options.csv", index=False)
    allp = pairs.assign(unit_id=r["unit_id"].to_numpy()[pairs["from_idx"].to_numpy()],
                        option_unit_id=r["unit_id"].to_numpy()[pairs["to_idx"].to_numpy()], miles=pairs["miles"].round(1))
    allp[["unit_id", "option_unit_id", "miles"]].to_csv(RESULTS / "better_option_pairs.csv", index=False)

    # Finder data: per college, stronger options and other comparable ones, nearest first.
    ids = r["unit_id"].to_numpy()

    def lists(p):
        return {int(k): [[int(ids[j]), round(float(m), 1)] for j, m in zip(g["to_idx"], g["miles"])][:6]
                for k, g in p.groupby("from_idx")}
    strong_map = lists(pairs)
    weaker_pairs = base_all.merge(pairs[["from_idx", "to_idx"]], how="left", indicator=True)
    comp_map = lists(weaker_pairs[weaker_pairs["_merge"] == "left_only"].drop(columns="_merge"))
    recs = []
    for i, row in r.iterrows():
        strong_ids = {x[0] for x in strong_map.get(i, [])}
        recs.append({"id": int(row["unit_id"]), "n": row["name"], "c": row["city"], "s": row["state"], "lv": row["level"],
                     "sec": row["sector_name"], "en": int(row["enrollment"]), "pe": float(row["pell_share"]),
                     "pr": int(round(row["net_price_low_income"])), "ea": int(row["earnings_dep_10yr"]),
                     "ex": int(round(row["expected_earnings"])), "va": round(float(row["value_added_pct"]), 1),
                     "vn": round(float(row["value_added_no_race_pct"]), 1), "op": bool(row["opportunity"]),
                     "pc": None if pd.isna(row["pell_completion_6yr"]) else float(row["pell_completion_6yr"]),
                     "msi": [m for m in MSI_TYPES if row[m] == 1],
                     "nc": ("closed" if row["closed_or_merged"] else "tribal" if row["tribal"] else
                            "specialized" if row["specialized"] else "online" if row["online_by_design"] else ""),
                     "st": strong_map.get(i, []),
                     "cm": comp_map.get(i, []),
                     "near": bool(near[i].any())})
    meta = {"pell_median": pell_median, "price_median": float(r["net_price_low_income"].median()),
            "rule": summary["also_consider"]["rule"], "n": len(recs)}
    (RESULTS / "finder_data.json").write_text(json.dumps({"meta": meta, "colleges": recs}, separators=(",", ":")))
    (RESULTS / "choice_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    show = {k: v for k, v in summary.items() if k != "similar_students"}
    show["also_consider"] = {k: v for k, v in summary["also_consider"].items() if k != "largest_examples"}
    print(json.dumps(show, indent=1, default=str))
    for e in summary["also_consider"]["largest_examples"]:
        print(e)


if __name__ == "__main__":
    main()
