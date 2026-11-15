"""Step 6: every chart used in the slides, drawn from the results files (no numbers typed by hand)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import FIGURES, RESULTS  # noqa: E402

# Validated reference palette (light mode): categorical slots 1-2, text and surface tokens.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ab"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


SLIDE_MODE = False  # True: no title/subtitle/source (the slide carries them), bigger type, figures/slides/
SLIDE_SCALE = 1.55


def fs(size):
    return size * SLIDE_SCALE if SLIDE_MODE else size


def finish(fig, name, subtitle=None, source=None):
    if SLIDE_MODE:
        fig.axes[0].set_title("")
        out = FIGURES / "slides"
        out.mkdir(exist_ok=True)
        fig.savefig(out / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    if subtitle:
        fig.axes[0].text(0, 1.02, subtitle, transform=fig.axes[0].transAxes, color=INK2, fontsize=fs(11), va="bottom")
    if source:  # below the x-axis label; bbox_inches="tight" keeps it in the image
        fig.text(0.01, -0.06, source, color=INK2, fontsize=fs(9))
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def money(x):
    return f"${x / 1000:,.1f}k"


def act1_quintiles():
    a = json.loads((RESULTS / "act1_summary.json").read_text())
    g = pd.DataFrame({"pay": a["median_entry_pay_10mi_by_mobility_quintile"],
                      "jobs": a["median_n_entry_10mi_by_mobility_quintile"]}, index=range(1, 6))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(g.index, g["pay"], color=BLUE, width=0.62)
    for x, pay in zip(g.index, g["pay"]):
        ax.text(x, pay + 800, money(pay), ha="center", va="bottom", fontsize=fs(12), color=INK, fontweight="bold")
    names = ["1 lowest", "2", "3", "4", "5 highest"]
    ax.set_xticks(g.index, [f"{n}\n{j:,.0f}\npostings" for n, j in zip(names, g["jobs"])])
    ax.set_xlabel("Neighborhood upward mobility (Opportunity Atlas quintile), with the median number of\n"
                  "entry-level postings with posted pay within 10 miles")
    ax.set_ylabel("Median posted entry-level pay within 10 miles")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.set_ylim(0, g["pay"].max() * 1.15)
    ax.grid(axis="x", visible=False)
    spread = (g["pay"].max() - g["pay"].min()) / g["pay"].mean()
    title = ("Within 10 miles of home, entry-level pay is about the same in low- and high-mobility neighborhoods"
             if spread < 0.10 else "Entry-level pay within 10 miles differs across neighborhoods")
    ax.set_title(title, pad=28)
    finish(fig, "fig1_job_access.png",
           subtitle=f"{a['tracts_in_quintile_comparison']:,} Bay Area tracts · entry-level Indeed postings with posted pay, "
                    f"{a['posting_date_min'][:4]}-{a['posting_date_max'][:4]} · mobility measured for kids born 1978-83",
           source="Sources: Indeed postings (datathon), Opportunity Atlas tract outcomes, Census 2010 Gazetteer. "
                  "Today's job map vs. an older generation's outcomes: a comparison, not a causal test.")


def model_accuracy():
    m = json.loads((RESULTS / "college_metrics.json").read_text())
    order = [("national_median", "National median\n(same guess for every college)"),
             ("pell_plus_sector_linear", "Pell share + sector\n(linear regression)"),
             ("linear_all_inputs", "All inputs\n(linear regression)"),
             ("gradient_boosting", "Our model\n(gradient-boosted trees)")]
    mae = [m["test_metrics"][k]["mae_usd"] for k, _ in order]
    ci = [m["bootstrap"]["mae_ci95"][k] for k, _ in order]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(order))[::-1]
    colors = [GRAY, GRAY, GRAY, BLUE]
    ax.barh(y, mae, color=colors, height=0.58)
    ax.errorbar(mae, y, xerr=[[v - lo for v, (lo, hi) in zip(mae, ci)], [hi - v for v, (lo, hi) in zip(mae, ci)]],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    for yi, v, (lo, hi) in zip(y, mae, ci):
        ax.text(hi + 250, yi, f"${v:,.0f}", va="center", fontsize=fs(12), color=INK, fontweight="bold")
    ax.set_yticks(y, [lab for _, lab in order])
    ax.set_xlabel("Average error predicting a college's median earnings of former students (USD, held-out test set)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.set_xlim(0, max(hi for _, hi in ci) * 1.18)
    ax.grid(axis="y", visible=False)
    r2 = m["test_metrics"]["gradient_boosting"]["r2_log"]
    cut = 1 - m["test_metrics"]["gradient_boosting"]["mae_usd"] / m["test_metrics"]["pell_plus_sector_linear"]["mae_usd"]
    ax.set_title(f"Our model's average error is {100 * cut:.0f}% smaller than the Pell-share adjustment's", pad=28)
    finish(fig, "fig2_model_accuracy.png",
           subtitle=f"{m['n_test']:,} colleges never used in training · whiskers = 95% bootstrap interval · our R² = {r2:.2f}",
           source="Target: median earnings of dependent students on federal aid, working 10 years after entry, "
                  "completers or not (College Results View 2021)")


def bottom20():
    b = pd.read_csv(RESULTS / "bottom20_by_measure.csv")
    b = b[b["group"] != "All colleges"].sort_values("share_in_bottom20_raw_earnings")
    labels = {"HBCU": "HBCUs", "PBI": "Predominantly Black\ninstitutions", "TCU": "Tribal colleges",
              "HSI": "Hispanic-serving\ninstitutions", "AANAPISI": "AANAPISIs",
              "Pell share above 61%": "Pell share\nabove 61%"}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(b))
    h = 0.38
    ax.barh(y + h / 2, 100 * b["share_in_bottom20_raw_earnings"], height=h, color=ORANGE, label="Ranked by raw earnings")
    ax.barh(y - h / 2, 100 * b["share_in_bottom20_value_added"], height=h, color=BLUE, label="Ranked by value-added (our model)")
    for yi, (r, v) in zip(y, zip(b["share_in_bottom20_raw_earnings"], b["share_in_bottom20_value_added"])):
        ax.text(100 * r + 1, yi + h / 2, f"{100 * r:.0f}%", va="center", fontsize=fs(11), color=INK)
        ax.text(100 * v + 1, yi - h / 2, f"{100 * v:.0f}%", va="center", fontsize=fs(11), color=INK)
    ax.axvline(20, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(20.6, -0.75, "20% = no penalty", color=INK2, fontsize=fs(10), va="bottom")
    ax.set_yticks(y, [f"{labels.get(g, g)} (n={n})" for g, n in zip(b["group"], b["n"])])
    ax.set_ylim(-0.85, len(b) - 0.4)
    ax.set_xlabel("Share of the group placed in the bottom 20% of colleges")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(0, 105)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    key = b.set_index("group")
    penalized = all(key.loc[g, "share_in_bottom20_raw_earnings"] > 0.30
                    and key.loc[g, "share_in_bottom20_raw_earnings"] > key.loc[g, "share_in_bottom20_value_added"]
                    for g in ["HBCU", "PBI", "Pell share above 61%"])
    ax.set_title("Raw-earnings rankings push colleges serving Black and low-income students to the bottom" if penalized
                 else "Who lands in the bottom 20%: raw earnings vs. value-added", pad=28)
    finish(fig, "fig3_bottom20.png",
           subtitle="Colleges with 500+ students and their own earnings figure · a fair measure puts about 20% of any group there",
           source="Value-added = actual earnings vs. what the model expects for that student body (cross-fitted)")


def opportunity():
    r = pd.read_parquet(RESULTS / "ranked_colleges.parquet")
    s = json.loads((RESULTS / "opportunity_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 6))
    other = r[~r["opportunity"]]
    opp = r[r["opportunity"]]
    ax.scatter(other["net_price_low_income"], other["value_added_pct"], s=10, color=GRAY, alpha=0.55,
               linewidths=0, label="Other ranked colleges")
    ax.scatter(opp["net_price_low_income"], opp["value_added_pct"], s=34, color=BLUE, edgecolors=SURFACE,
               linewidths=0.8, label=f"Opportunity colleges (n={len(opp)})", zorder=3)
    ax.axhline(s["rules"]["value_added_at_least_pct"], color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.axvline(s["rules"]["net_price_low_income_at_most_median_usd"], color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ca = opp[opp["state"] == "CA"].sort_values("value_added_pct", ascending=False)
    x_text = s["rules"]["net_price_low_income_at_most_median_usd"] + 4000
    for i, (_, row) in enumerate(ca.iterrows()):  # labels stacked in a column right of the median line
        ax.annotate(f"{row['name'].replace('University of California-', 'UC ')}  {row['value_added_pct']:+.0f}%",
                    (row["net_price_low_income"], row["value_added_pct"]), xytext=(x_text, 46 - 9 * i),
                    textcoords="data", fontsize=fs(10.5), color=INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=3))
    ax.set_xlabel("Net price for students from families earning up to $30k (2022-23)")
    ax.set_ylabel("Value-added: earnings vs. what the model expects")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlim(left=min(0, r["net_price_low_income"].min()) - 500, right=r["net_price_low_income"].quantile(0.995))
    ax.set_ylim(r["value_added_pct"].quantile(0.002) - 2, r["value_added_pct"].quantile(0.998) + 4)
    ax.legend(loc="lower right", frameon=False, markerscale=2)
    ax.set_title(f"{len(opp)} low-price colleges where former students out-earn what their intake predicts", pad=28)
    finish(fig, "fig4_opportunity.png",
           subtitle="Blue = value-added ≥ +10%, net price ≤ median, Pell share ≥ median · California colleges labeled",
           source="Sources: College Results View 2021, Affordability Gap AY2022-23 (lowest family-income bracket)")


def predicted_vs_actual():
    p = pd.read_parquet(RESULTS / "college_predictions.parquet")
    t = p[p["in_test_set"]]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(t["earnings_dep_10yr"], t["test_pred_gbm"], s=12, color=BLUE, alpha=0.6, linewidths=0)
    both = pd.concat([t["earnings_dep_10yr"], t["test_pred_gbm"]])
    lo, hi = both.min() * 0.9, both.max() * 1.1
    ax.plot([lo, hi], [lo, hi], color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [15_000, 25_000, 40_000, 60_000, 100_000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual median earnings of former students"); ax.set_ylabel("Predicted (model never saw these colleges)")
    ax.set_title("Predictions track actual earnings across the range", pad=28)
    finish(fig, "fig5_predicted_vs_actual.png", subtitle=f"Held-out test set, {len(t):,} colleges · dashed = perfect prediction")


def errors_by_group():
    e = pd.read_csv(RESULTS / "errors_by_group.csv")
    e = e[e["predictions"].str.startswith("cross-fitted") & e["dimension"].isin(["Minority-serving designation", "Sector"])]
    e = e.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(e))
    colors = [ORANGE if d == "Minority-serving designation" else BLUE for d in e["dimension"]]
    ax.scatter(e["mean_signed_error_pct"], y, color=colors, s=60, zorder=3)
    for yi, b in zip(y, e["mean_signed_error_pct"]):
        ax.text(b + 0.45, yi, f"{b:+.1f}%", va="center", ha="left", fontsize=fs(10))
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-2, max(12, e["mean_signed_error_pct"].max() + 2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
    ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(e["group"], e["n"])])
    ax.set_xlabel("Average signed error: + means the model expects more than former students earn")
    ax.grid(axis="y", visible=False)
    big = e[e["mean_signed_error_pct"].abs() > 5]
    names = {"TCU": "tribal colleges", "HBCU": "HBCUs", "PBI": "predominantly Black institutions", "HSI": "HSIs",
             "AANAPISI": "AANAPISIs"}
    if big.empty:
        title = "The model's average error is within ±5% for every group"
    else:
        title = ("The model is close to unbiased for most groups, but not for "
                 + ", ".join(names.get(g, g) for g in big["group"]))
    ax.set_title(title, pad=28)
    finish(fig, "fig6_bias_by_group.png", subtitle="Cross-fitted predictions for all colleges · orange = minority-serving designations, blue = sectors")


def feature_importance():
    f = pd.read_csv(RESULTS / "feature_importance.csv").head(10).iloc[::-1]
    names = {"pell_share": "Share of Pell (low-income) students", "sector": "Sector", "carnegie": "Carnegie type",
             "state": "State", "pct_women": "Share of women", "log_enrollment": "Enrollment",
             "pct_asian": "Share of Asian students", "open_admissions": "Open admissions",
             "pct_ftft": "Share full-time first-time", "admit_rate": "Admission rate",
             "sat_math_25": "SAT math (25th pct.)", "pct_black": "Share of Black students",
             "pct_latino": "Share of Latino students", "pct_age25plus": "Share aged 25+",
             "pct_part_time": "Share part-time", "urbanization": "Urbanization", "pct_white": "Share of white students"}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh([names.get(x, x) for x in f["feature"]], 100 * f["mae_increase_log"], color=BLUE, height=0.6)
    ax.set_xlabel("Increase in error when the input is shuffled (percentage points of log earnings)")
    ax.grid(axis="y", visible=False)
    top3 = [names.get(x, x) for x in f["feature"].iloc[::-1].head(3)]
    ax.set_title(f"Biggest drivers of expected earnings: {top3[0].lower()}, {top3[1].lower()}, {top3[2].lower()}", pad=28)
    finish(fig, "fig7_feature_importance.png", subtitle="Permutation importance on the held-out test set")


def similar_students():
    c = json.loads((RESULTS / "choice_summary.json").read_text())["similar_students"]
    q = c["all"]["quarters"]
    order = ["worst", "second", "third", "best"]
    earn = [q[k]["median_earnings"] for k in order]
    price = [q[k]["median_net_price"] for k in order]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = [GRAY, GRAY, GRAY, BLUE]
    ax.bar(range(4), earn, color=colors, width=0.62)
    for x, e in enumerate(earn):
        ax.text(x, e + 600, money(e), ha="center", va="bottom", fontsize=fs(13), color=INK, fontweight="bold")
    names = ["Lowest", "Second", "Third", "Highest"]
    ax.set_xticks(range(4), [f"{n}\nprice {money(p)}" for n, p in zip(names, price)])
    ax.set_xlabel("Value-added quarter (price = net price for families earning up to $30,000)")
    ax.set_ylabel("Median earnings, 10 years after entry")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.set_ylim(0, max(earn) * 1.16)
    ax.grid(axis="x", visible=False)
    gap = c["all"]["gap_best_vs_worst_usd"]
    ax.set_title(f"Colleges with similar low-income students: ${gap:,.0f} a year apart")
    big = c["enrollment_5000_plus"]["gap_best_vs_worst_usd"]
    finish(fig, "fig8_similar_students.png",
           subtitle=(f"{c['all']['n']} colleges with Pell share at or above the median and similar expected earnings "
                     f"(within 10% of ${c['expected_median_usd']:,.0f}) · at colleges with 5,000+ students the gap is ${big:,.0f}"),
           source="Price = net price for families earning up to $30,000. Source: College Results 2021, Affordability Gap AY2022-23; our model.")


def holds_up():
    rs = json.loads((RESULTS / "choice_summary.json").read_text())["also_consider"]["replication_same_start"]
    rows = [("Nearest stronger option\n(value-added rule)", rs["nearest_stronger_share_higher"], BLUE),
            ("Nearest other comparable\ncollege", rs["nearest_other_comparable_share_higher"], GRAY)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.barh(yi, v * 100, color=col, height=0.56)
        ax.text(v * 100 + 1, yi, f"{v:.0%}", va="center", fontsize=fs(14), fontweight="bold", color=INK)
    ax.axvline(50, color=INK2, linewidth=1.2, linestyle="--")
    ax.text(50.6, y.max() + 0.42, "coin flip", color=INK2, fontsize=fs(10), va="bottom")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Share where independent (mostly older) students also earn more, same {rs['starting_colleges']} starting colleges")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title("A different group of students points the same way")
    lo, hi = rs["difference_ci95"]
    finish(fig, "fig9_holds_up.png", subtitle=f"Independent-student earnings are not a model input · difference "
                                              f"{lo * 100:+.0f} to {hi * 100:+.0f} points (95% interval)",
           source="Source: College Results 2021; our model.")

def set_mode(slide):
    global SLIDE_MODE
    SLIDE_MODE = slide
    base = 12 * (SLIDE_SCALE if slide else 1)
    plt.rcParams.update({"font.size": base, "axes.titlesize": base * 1.25, "xtick.labelsize": base * 0.95,
                         "ytick.labelsize": base * 0.95, "legend.fontsize": base * 0.95, "axes.labelsize": base})


if __name__ == "__main__":
    for slide in (False, True):
        set_mode(slide)
        for fn in [act1_quintiles, model_accuracy, bottom20, opportunity, predicted_vs_actual, errors_by_group,
                   feature_importance, similar_students, holds_up]:
            fn()
            print("drew", fn.__name__, "(slide version)" if slide else "")
