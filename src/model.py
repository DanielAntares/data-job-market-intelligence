"""Phase 4: predict advertised salary — baseline-first, honestly evaluated.

    python -m src.model

Philosophy (and the point of the exercise): with only a few dozen genuinely
advertised salaries, the right move is *not* a fancy model. It's:
  1. establish dumb baselines (global median, role median),
  2. fit a small, regularised, interpretable model (RidgeCV),
  3. evaluate with k-fold cross-validation (a single split of ~60 rows is noise),
  4. report error in dollars against the baselines, and
  5. read off interpretable coefficients ("Deep Learning ~ +$X").

The model is deliberately humble; its value is that it *improves automatically*
as the scheduled collector accumulates more advertised salaries.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis
from .skills import TAXONOMY as _TAXONOMY
from .storage import load_skills, load_warehouse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("model")

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

# Feature columns the model consumes (besides the one-hot "role").
_NUMERIC = [
    "seniority", "remote", "n_skills",
    "has_python", "has_sql", "has_machine_learning", "has_deep_learning",
    "has_cloud", "has_ml_library", "has_big_data",
]


def seniority_level(title: str | None) -> int:
    """Ordinal seniority from a title: junior=1, mid=2, senior=3, principal+=4."""
    t = (title or "").lower()
    if any(k in t for k in ("principal", "staff", "lead ", " lead", "head of",
                            "director", "vp", "vice president", "chief",
                            "distinguished", "manager")):
        return 4
    if any(k in t for k in ("senior", "sr.", "sr ", "snr")):
        return 3
    if any(k in t for k in ("junior", "jr.", "entry", "intern", "associate",
                            "graduate", "trainee")):
        return 1
    return 2


# Map each skill to its category once, so the same feature logic powers both
# training (skills from postings) and the live predictor (skills a user picks).
_CAT_BY_SKILL = {s.name: s.category for s in _TAXONOMY}

# The feature columns the estimators consume (role is one-hot encoded).
FEATURE_COLS = ["role"] + _NUMERIC


def row_features(role: str, seniority: int, remote, skill_set: set[str]) -> dict:
    """Build one feature row from raw ingredients. Shared by train + predict."""
    cats = {_CAT_BY_SKILL.get(s) for s in skill_set}
    return {
        "role": role,
        "seniority": int(seniority),
        "remote": int(bool(remote)),
        "n_skills": len(skill_set),
        "has_python": int("Python" in skill_set),
        "has_sql": int("SQL" in skill_set),
        "has_machine_learning": int("Machine Learning" in skill_set),
        "has_deep_learning": int("Deep Learning" in skill_set),
        "has_cloud": int("cloud" in cats),
        "has_ml_library": int("ml_library" in cats),
        "has_big_data": int("big_data" in cats),
    }


def build_features(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    """Assemble the modeling table: one row per advertised-salary posting."""
    sal = analysis.build_salary_table(jobs)  # job_id, salary (advertised USD/yr)
    df = jobs.merge(sal, on="job_id")
    skill_sets = skills.groupby("job_id")["skill"].apply(set)

    rows = []
    for _, r in df.iterrows():
        s = skill_sets.get(r["job_id"], set())
        feat = row_features(analysis.classify_role(r["title"]),
                            seniority_level(r["title"]), r["remote"], s)
        feat["salary"] = r["salary"]
        rows.append(feat)
    return pd.DataFrame(rows)


def _ridge_pipeline():
    """The interpretable model, reused for coefficients and live prediction."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    prep = ColumnTransformer(
        [("role", OneHotEncoder(handle_unknown="ignore"), ["role"])],
        remainder="passthrough")
    return Pipeline([("prep", prep), ("m", RidgeCV(alphas=np.logspace(-1, 4, 30)))])


def train_salary_model(jobs: pd.DataFrame, skills: pd.DataFrame):
    """Fit the RidgeCV model on all advertised-salary postings. Returns (pipe, n)."""
    feats = build_features(jobs, skills)
    pipe = _ridge_pipeline()
    pipe.fit(feats[FEATURE_COLS], feats["salary"])
    return pipe, len(feats)


def predict_salary(pipe, role: str, seniority: int, remote, selected_skills) -> float:
    """Predict salary for a hypothetical posting (used by the dashboard)."""
    row = row_features(role, seniority, remote, set(selected_skills))
    X = pd.DataFrame([row])[FEATURE_COLS]
    return float(pipe.predict(X)[0])


def _make_estimators():
    """Return {name: estimator}: baselines first, then the real models."""
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    role_ohe = OneHotEncoder(handle_unknown="ignore")
    full_prep = ColumnTransformer([("role", role_ohe, ["role"])], remainder="passthrough")
    role_only_prep = ColumnTransformer([("role", role_ohe, ["role"])], remainder="drop")
    return {
        "Baseline: global median": DummyRegressor(strategy="median"),
        "Baseline: role mean": Pipeline([("prep", role_only_prep),
                                         ("m", LinearRegression())]),
        "RidgeCV (skills+role+seniority)": _ridge_pipeline(),
        "RandomForest": Pipeline(
            [("prep", full_prep),
             ("m", RandomForestRegressor(n_estimators=300, random_state=42))]),
    }


def evaluate() -> None:
    from sklearn.model_selection import KFold, cross_val_predict, cross_validate

    jobs, skills = load_warehouse(), load_skills()
    if jobs.empty or skills.empty:
        log.error("Need data — run collect + extract_skills first.")
        return

    feats = build_features(jobs, skills)
    n = len(feats)
    if n < 20:
        log.warning("Only %d advertised-salary postings — results are indicative "
                    "at best. Keep collecting.", n)
    X = feats[["role"] + _NUMERIC]
    y = feats["salary"]

    print("=" * 66)
    print(f"SALARY MODEL - predicting advertised USD annual salary  (n={n})")
    print("=" * 66)
    print(f"Target: median ${y.median():,.0f}, range ${y.min():,.0f}-${y.max():,.0f}\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"MAE": "neg_mean_absolute_error",
               "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

    print(f"{'Model':<34}{'MAE':>12}{'RMSE':>12}{'R2':>8}")
    print("-" * 66)
    results = {}
    for name, est in _make_estimators().items():
        cvres = cross_validate(est, X, y, cv=cv, scoring=scoring)
        mae = -cvres["test_MAE"].mean()
        rmse = -cvres["test_RMSE"].mean()
        r2 = cvres["test_R2"].mean()
        results[name] = mae
        print(f"{name:<34}{'$'+format(mae,',.0f'):>12}{'$'+format(rmse,',.0f'):>12}{r2:>8.2f}")

    base = results["Baseline: global median"]
    best_name = min((k for k in results if not k.startswith("Baseline")),
                    key=lambda k: results[k])
    best = results[best_name]
    gain = 100 * (base - best) / base
    print("-" * 66)
    print(f"Best model: {best_name} - MAE ${best:,.0f}, "
          f"{gain:+.0f}% vs. the global-median baseline.\n")

    _report_coefficients(X, y)
    _save_eval_plot(X, y, cv, best_name)


def _report_coefficients(X, y) -> None:
    """Fit RidgeCV on all data and show the $ each feature adds/subtracts."""
    pipe = _ridge_pipeline()
    pipe.fit(X, y)
    names = [n.split("__")[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    coefs = pipe.named_steps["m"].coef_
    pairs = sorted(zip(names, coefs), key=lambda p: -abs(p[1]))

    print("INTERPRETATION - RidgeCV coefficients (~ $ vs. an average posting)")
    for name, c in pairs[:10]:
        print(f"  {name:<26} {'+' if c >= 0 else '-'}${abs(c):>8,.0f}")
    print("  (directional, not causal - small sample, correlated features)\n")


def _save_eval_plot(X, y, cv, best_name) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_predict

    pipe = _make_estimators()[best_name]
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    mae = mean_absolute_error(y, y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y / 1000, y_pred / 1000, alpha=0.6, color="#2b6cb0", edgecolor="white")
    lim = [min(y.min(), y_pred.min()) / 1000 - 5, max(y.max(), y_pred.max()) / 1000 + 5]
    ax.plot(lim, lim, "--", color="gray", label="perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual salary ($k)")
    ax.set_ylabel("Cross-validated prediction ($k)")
    ax.set_title(f"Predicted vs. actual (out-of-fold)\n{best_name} — MAE ${mae:,.0f}")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "model_eval.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote reports/figures/model_eval.png")


if __name__ == "__main__":
    evaluate()
