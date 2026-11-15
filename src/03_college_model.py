"""Step 3: train the expected-earnings model, compare it with baselines, and estimate value-added.

Model: gradient-boosted trees predicting log median earnings (dependent students, 10 years after
entry) from who enrolls and where. Expected earnings = what a college "should" produce given its
students; value-added = actual vs expected.

Each earnings figure counts once: a college whose figure is shared by k campuses gets weight 1/k when
fitting (so a chain with 20 campuses does not count 20 times). Test metrics are reported per college
and per reporting unit (weighted the same way).

Evaluation discipline:
  * 20% of colleges are held out as a test set, split by `report_group` (campuses that share one
    earnings figure stay together). Hyperparameters are chosen by 5-fold grouped cross-validation on
    the training 80% only; the test set is scored once, at the end.
  * Value-added uses cross-fitted predictions: every college's expected earnings come from models
    that never saw that college (or its sibling campuses). Averaged over 5 different fold assignments.

Outputs: results/college_metrics.json, results/college_predictions.parquet, results/feature_importance.csv
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import common.py from any working dir
from common import (CATEGORICAL_FEATURES, INTERIM, NUMERIC_FEATURES, RACE_FEATURES, RESULTS,  # noqa: E402
                    SEED)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# A first, smaller grid picked values on its edges (0.03 / 31 / 20 / 600), so the grid extends past them.
GRID = {"learning_rate": [0.02, 0.03, 0.06], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 1.0], "max_iter": [600, 1200]}


def load():
    df = pd.read_parquet(INTERIM / "college_analysis.parquet")
    for c in CATEGORICAL_FEATURES:  # categories fixed on the full table so codes match across splits
        df[c] = df[c].astype(str).astype("category")
    return df.reset_index(drop=True)


def gbm(params, features=FEATURES):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED, **params)


def dollars_mae(y_log, pred_log, weight=None):
    return mean_absolute_error(np.exp(y_log), np.exp(pred_log), sample_weight=weight)


def unit_weight(frame):
    return 1.0 / frame["report_group_size"].to_numpy()


def tune(df, train_idx):
    tr = df.iloc[train_idx]
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr["report_group"]))
    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        maes = []
        for f_tr, f_va in folds:
            fit_part, val_part = tr.iloc[f_tr], tr.iloc[f_va]
            m = gbm(params).fit(fit_part[FEATURES], fit_part["log_earnings"], sample_weight=unit_weight(fit_part))
            maes.append(dollars_mae(val_part["log_earnings"], m.predict(val_part[FEATURES]), unit_weight(val_part)))
        results.append({**params, "cv_mae_usd": float(np.mean(maes))})
    results = sorted(results, key=lambda r: r["cv_mae_usd"])
    best = {k: results[0][k] for k in GRID}
    return best, results


def linear_all_inputs(alpha=1.0):
    """Ridge regression on every input: median-imputed numerics (+ missing flags), one-hot categoricals."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL_FEATURES)])
    return make_pipeline(pre, Ridge(alpha=alpha))


def tune_ridge(df, train_idx):
    """Pick the ridge penalty the same way as the model's settings: grouped 5-fold CV on training data."""
    tr = df.iloc[train_idx]
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr["report_group"]))
    scores = {}
    for alpha in np.logspace(-2, 3, 11):
        maes = []
        for f_tr, f_va in folds:
            fit_part, val_part = tr.iloc[f_tr], tr.iloc[f_va]
            m = linear_all_inputs(alpha).fit(fit_part, fit_part["log_earnings"], ridge__sample_weight=unit_weight(fit_part))
            maes.append(dollars_mae(val_part["log_earnings"], m.predict(val_part), unit_weight(val_part)))
        scores[float(alpha)] = float(np.mean(maes))
    return min(scores, key=scores.get)


def pell_sector_linear():
    """The simple adjustment many 'value-added' rankings use: Pell share + sector."""
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["pell_share"]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["sector"])])
    return make_pipeline(pre, LinearRegression())


def metrics(y_log, pred_log, weight=None):
    true, pred = np.exp(y_log), np.exp(pred_log)
    return {"mae_usd": float(mean_absolute_error(true, pred)),
            "mae_usd_per_reporting_unit": float(mean_absolute_error(true, pred, sample_weight=weight))
            if weight is not None else None,
            "rmse_usd": float(np.sqrt(np.mean((pred - true) ** 2))),
            "median_abs_pct_error": float(100 * np.median(np.abs(pred - true) / true)),
            "r2_log": float(r2_score(y_log, pred_log)),
            # rank correlation is undefined for a constant prediction (the national-median baseline)
            "spearman_rank": float(spearmanr(true, pred)[0]) if np.ptp(pred) > 0 else None}


def bootstrap_mae(y_log, preds, groups, n=2000):
    """Group-resampled bootstrap: 95% intervals for each model's MAE and for GBM minus each baseline."""
    rng = np.random.default_rng(SEED)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    errs = {k: np.abs(np.exp(p) - np.exp(y_log)) for k, p in preds.items()}
    draws = {k: [] for k in preds}
    diffs = {k: [] for k in preds if k != "gradient_boosting"}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        m = {k: e[idx].mean() for k, e in errs.items()}
        for k in preds:
            draws[k].append(m[k])
        for k in diffs:
            diffs[k].append(m["gradient_boosting"] - m[k])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"mae_ci95": {k: ci(v) for k, v in draws.items()},
            "gbm_minus_baseline_mae_ci95": {k: ci(v) for k, v in diffs.items()}}


def cross_fit(df, params, features, repeats=5):
    """Average out-of-fold prediction for every college over `repeats` grouped 5-fold assignments."""
    preds = np.zeros((repeats, len(df)))
    for r in range(repeats):
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=SEED + r).split(df, groups=df["report_group"]):
            part = df.iloc[tr]
            m = gbm(params).fit(part[features], part["log_earnings"], sample_weight=unit_weight(part))
            preds[r, te] = m.predict(df[features].iloc[te])
    return preds.mean(axis=0), preds.std(axis=0)


def main():
    t0 = time.time()
    df = load()
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                               .split(df, groups=df["report_group"]))
    assert not set(df["report_group"].iloc[train_idx]) & set(df["report_group"].iloc[test_idx])
    tr, te = df.iloc[train_idx], df.iloc[test_idx]

    best, grid_results = tune(df, train_idx)
    ridge_alpha = tune_ridge(df, train_idx)
    w_tr = unit_weight(tr)
    model = gbm(best).fit(tr[FEATURES], tr["log_earnings"], sample_weight=w_tr)
    unit_median = tr.groupby("report_group")["earnings_dep_10yr"].first().median()  # each figure once
    preds = {
        "national_median": np.full(len(te), np.log(unit_median)),
        "pell_plus_sector_linear": pell_sector_linear().fit(tr, tr["log_earnings"], linearregression__sample_weight=w_tr).predict(te),
        "linear_all_inputs": linear_all_inputs(ridge_alpha).fit(tr, tr["log_earnings"], ridge__sample_weight=w_tr).predict(te),
        "gradient_boosting": model.predict(te[FEATURES]),
    }
    test_metrics = {k: metrics(te["log_earnings"], p, unit_weight(te)) for k, p in preds.items()}
    boot = bootstrap_mae(te["log_earnings"].to_numpy(), preds, te["report_group"].to_numpy())

    imp = permutation_importance(model, te[FEATURES], te["log_earnings"], n_repeats=10, random_state=SEED,
                                 scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({"feature": FEATURES, "mae_increase_log": imp.importances_mean,
                               "std": imp.importances_std}).sort_values("mae_increase_log", ascending=False)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)

    # Value-added for every college from cross-fitted predictions (with and without race/ethnicity inputs)
    expected, expected_sd = cross_fit(df, best, FEATURES)
    no_race = [f for f in FEATURES if f not in RACE_FEATURES]
    expected_nr, _ = cross_fit(df, best, no_race)
    out = df[["unit_id", "name", "city", "state", "sector", "sector_name", "pell_share", "enrollment",
              "earnings_dep_10yr", "report_group", "report_group_size", "HBCU", "HSI", "TCU", "PBI", "AANAPISI", "pct_black",
              "pct_latino", "net_price_low_income", "affordability_gap", "weekly_hours_to_close_gap",
              "parent_gap_center_care", "county_name", "lat", "lon", "grad_rate_6yr",
              "median_debt_completers"]].copy()
    out["state"] = out["state"].astype(str)
    out["sector"] = out["sector"].astype(str)
    out["expected_earnings"] = np.exp(expected)
    out["expected_sd_log"] = expected_sd
    out["value_added_pct"] = 100 * (np.exp(df["log_earnings"] - expected) - 1)
    out["expected_earnings_no_race"] = np.exp(expected_nr)
    out["value_added_no_race_pct"] = 100 * (np.exp(df["log_earnings"] - expected_nr) - 1)
    out["in_test_set"] = False
    out.loc[test_idx, "in_test_set"] = True
    out["test_pred_gbm"] = np.nan
    out.loc[test_idx, "test_pred_gbm"] = np.exp(preds["gradient_boosting"])
    out["test_pred_pell_sector"] = np.nan
    out.loc[test_idx, "test_pred_pell_sector"] = np.exp(preds["pell_plus_sector_linear"])
    out["test_pred_linear_all"] = np.nan
    out.loc[test_idx, "test_pred_linear_all"] = np.exp(preds["linear_all_inputs"])
    out.to_parquet(RESULTS / "college_predictions.parquet", index=False)

    cf = metrics(df["log_earnings"], expected, unit_weight(df))
    summary = {
        "n_colleges": int(len(df)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_report_groups_train": int(tr["report_group"].nunique()), "n_report_groups_test": int(te["report_group"].nunique()),
        "best_params": best, "grid_top5": grid_results[:5], "n_grid_configs": len(grid_results),
        "ridge_alpha": ridge_alpha,
        "test_metrics": test_metrics, "bootstrap": boot,
        "cross_fitted_all_colleges": cf,
        "cross_fitted_no_race_all_colleges": metrics(df["log_earnings"], expected_nr, unit_weight(df)),
        "median_earnings_test": float(np.median(np.exp(te["log_earnings"])))}
    # Run time is printed, not saved, so a rerun leaves the results files byte-identical.
    (RESULTS / "college_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["n_colleges", "n_train", "n_test", "best_params", "test_metrics",
                                              "bootstrap", "cross_fitted_all_colleges"]}, indent=1))
    print(f"runtime {time.time() - t0:.0f}s")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
