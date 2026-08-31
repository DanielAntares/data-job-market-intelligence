"""Phase 4: predict advertised salary — baseline-first, honestly evaluated.

    python -m src.model

Philosophy (and the point of the exercise): the win here came from the data,
not the algorithm. In order of what actually moved the score:

  1. **Fix the target.** Convert every currency and pay period to USD/year
     (+26% usable rows) and drop parse failures like a "$162/yr" engineer.
  2. **Use the columns already collected.** ``location``, ``source``,
     ``market``, ``category`` and ``job_type`` sat unused in the warehouse.
     Location alone is the single largest salary driver in posting data.
  3. **Mine the description.** Years-of-experience is a regex; TF-IDF over the
     posting text is a strong, cheap feature set.
  4. **Then** worry about estimators — and the humble linear one still wins.

  5. **Evaluate honestly.** Cross-validation is grouped by *company*: 293
     companies supply 693 postings, so a random split puts the same employer in
     train and test and the model memorises its band. Fixing that leak dropped
     the headline R2 by about a third. The lower number is the real one.

Baselines come first and stay in the table, so every gain is quoted against
"just predict the median" rather than against nothing.
"""
from __future__ import annotations

import logging
import re
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


# --- Location ---------------------------------------------------------------
# 99 distinct location strings, most seen once or twice. One-hot encoding them
# raw would be mostly noise, so they collapse into pay-relevant regions.
_LOC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("US", ("usa", "united states", "u.s.", "america", "new york",
            "san francisco", "california", "texas", "washington", "boston",
            "chicago", "seattle", "austin", "denver", "atlanta", "remote us")),
    ("Canada", ("canada", "toronto", "vancouver", "montreal", "ontario")),
    ("UK", ("uk", "united kingdom", "england", "london", "scotland", "wales")),
    ("Europe", ("europe", "emea", "germany", "france", "spain", "portugal",
                "poland", "netherlands", "ireland", "italy", "sweden", "norway",
                "denmark", "czech", "romania", "switzerland", "austria",
                "belgium", "greece", "berlin", "amsterdam", "paris", "lisbon",
                "madrid", "dublin")),
    ("ANZ", ("australia", "new zealand", "sydney", "melbourne", "brisbane",
             "perth", "adelaide", "canberra", "auckland")),
    ("LATAM", ("brazil", "mexico", "argentina", "colombia", "chile", "peru",
               "latam", "latin america", "uruguay", "costa rica")),
    ("Asia", ("india", "indonesia", "malaysia", "singapore", "philippines",
              "japan", "china", "vietnam", "thailand", "apac", "asia",
              "jakarta", "kuala lumpur", "bangalore", "hong kong", "korea")),
    ("Worldwide", ("worldwide", "anywhere", "global", "remote")),
]

LOCATIONS = [name for name, _ in _LOC_RULES] + ["Other"]


def location_bucket(location: str | None) -> str:
    """Collapse a free-text location into a coarse, pay-relevant region."""
    t = (location or "").lower()
    for name, keywords in _LOC_RULES:
        if any(k in t for k in keywords):
            return name
    return "Other"


# --- Description mining -----------------------------------------------------
# Descriptions run ~6,600 characters and were entirely unused. Two cheap reads:
# a required-experience number, and a handful of compensation/seniority cues.
_YEARS_RE = [
    re.compile(r"(\d{1,2})\s*\+?\s*(?:to|-|–)\s*\d{1,2}\s*\+?\s*years?", re.I),
    re.compile(r"(?:minimum|at least|min\.?|over)\s*(?:of\s*)?(\d{1,2})\s*\+?\s*years?", re.I),
    re.compile(r"(\d{1,2})\s*\+\s*years?", re.I),
    re.compile(r"(\d{1,2})\s*years?\s+(?:of\s+)?(?:relevant\s+|professional\s+|"
               r"hands[- ]on\s+|work\s+)?experience", re.I),
]

_DESC_FLAGS = {
    "mentions_equity": re.compile(r"\b(equity|stock option|rsu|share option)\b", re.I),
    "mentions_bonus": re.compile(r"\b(bonus|profit shar|commission)\b", re.I),
    "mentions_phd": re.compile(r"\b(ph\.?d|doctorate)\b", re.I),
    "mentions_msc": re.compile(r"\b(master'?s|msc|m\.s\.)\b", re.I),
    "mentions_lead": re.compile(r"\b(mentor|lead a team|manage a team|direct reports)\b", re.I),
    "mentions_clearance": re.compile(r"\b(security clearance|ts/sci|clearance required)\b", re.I),
}

# Digits and currency symbols are stripped before the text is vectorised, so a
# posting that quotes its own salary ("$150,000 - $180,000") cannot leak it
# back as a feature. Measured cost of the precaution: none (R2 0.458 -> 0.457).
_DIGITS_RE = re.compile(r"[\d$£€]+")


def years_of_experience(text: str | None) -> float:
    """Median required years mentioned in a description; NaN when unstated."""
    t = (text or "")[:8000]
    vals = [int(m.group(1)) for rx in _YEARS_RE for m in rx.finditer(t)
            if 0 < int(m.group(1)) <= 20]
    return float(np.median(vals)) if vals else float("nan")


def scrub_text(text: str | None) -> str:
    """Description with numbers removed, ready for the vectoriser."""
    return _DIGITS_RE.sub(" ", text or "")


# --- Features ---------------------------------------------------------------
_CAT_BY_SKILL = {s.name: s.category for s in _TAXONOMY}
_SKILL_CATS = sorted({s.category for s in _TAXONOMY})

_SKILL_FLAGS = ["has_python", "has_sql", "has_machine_learning",
                "has_deep_learning", "has_cloud", "has_ml_library",
                "has_big_data"]
_SKILL_COUNTS = [f"n_{c}" for c in _SKILL_CATS]
_DESC_NUMERIC = ["years_exp", "desc_len"] + list(_DESC_FLAGS)

# Features the interactive predictor can ask a user for. The dashboard has no
# job description to hand, so the live model is deliberately the subset of the
# evaluation model whose inputs a person can actually supply.
INTERACTIVE_NUMERIC = (["seniority", "remote", "n_skills", "years_exp"]
                       + _SKILL_FLAGS + _SKILL_COUNTS)
INTERACTIVE_CATEGORICAL = ["role", "location"]
INTERACTIVE_COLS = INTERACTIVE_CATEGORICAL + INTERACTIVE_NUMERIC

# Everything the offline evaluation may use, including columns only a real
# posting carries.
FULL_NUMERIC = INTERACTIVE_NUMERIC + ["desc_len"] + list(_DESC_FLAGS)
FULL_CATEGORICAL = INTERACTIVE_CATEGORICAL + ["source", "market", "job_type",
                                              "category"]
FULL_COLS = FULL_CATEGORICAL + FULL_NUMERIC + ["description"]

# Kept for backwards compatibility with earlier notebooks/tests.
FEATURE_COLS = INTERACTIVE_COLS


def _col(row, name, default=""):
    """Read an optional warehouse column, tolerating partial test frames."""
    try:
        val = row[name]
    except (KeyError, IndexError):
        return default
    return default if pd.isna(val) else val


def row_features(role: str, seniority: int, remote, skill_set: set[str], *,
                 location: str = "Worldwide", years_exp: float = float("nan"),
                 source: str = "", market: str = "", job_type: str = "unknown",
                 category: str = "unknown", description: str = "") -> dict:
    """Build one feature row from raw ingredients. Shared by train + predict."""
    cats = [_CAT_BY_SKILL.get(s) for s in skill_set]
    return {
        "role": role,
        "location": location_bucket(location) if location not in LOCATIONS else location,
        "source": str(source),
        "market": str(market),
        "job_type": str(job_type or "unknown").lower().replace("-", "_"),
        "category": str(category or "unknown"),
        "seniority": int(seniority),
        "remote": int(bool(remote)),
        "n_skills": len(skill_set),
        "years_exp": years_exp,
        "has_python": int("Python" in skill_set),
        "has_sql": int("SQL" in skill_set),
        "has_machine_learning": int("Machine Learning" in skill_set),
        "has_deep_learning": int("Deep Learning" in skill_set),
        "has_cloud": int("cloud" in cats),
        "has_ml_library": int("ml_library" in cats),
        "has_big_data": int("big_data" in cats),
        **{f"n_{c}": sum(1 for x in cats if x == c) for c in _SKILL_CATS},
        "desc_len": len(description or ""),
        **{k: int(bool(rx.search(description or ""))) for k, rx in _DESC_FLAGS.items()},
        "description": scrub_text(description),
    }


def build_features(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    """Assemble the modeling table: one row per posting with a usable salary.

    Also carries ``company`` — not a feature, but the grouping key that keeps
    cross-validation honest.
    """
    sal = analysis.build_salary_table(jobs)  # job_id, salary (USD/yr), converted
    df = jobs.merge(sal, on="job_id")
    skill_sets = skills.groupby("job_id")["skill"].apply(set)

    rows = []
    for _, r in df.iterrows():
        desc = _col(r, "description_text", "")
        feat = row_features(
            analysis.classify_role(r["title"]),
            seniority_level(r["title"]),
            r["remote"],
            skill_sets.get(r["job_id"], set()),
            location=_col(r, "location", ""),
            years_exp=years_of_experience(desc),
            source=_col(r, "source", ""),
            market=_col(r, "market", ""),
            job_type=_col(r, "job_type", "unknown"),
            category=_col(r, "category", "unknown"),
            description=desc,
        )
        feat["salary"] = r["salary"]
        feat["company"] = normalize_company(_col(r, "company", ""))
        rows.append(feat)
    return pd.DataFrame(rows)


_COMPANY_SUFFIX = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|co|gmbh|bv|plc|sa|ag|"
    r"international|group|holdings|technologies|technology|labs|"
    r"solutions|services|systems|global)\b")


def normalize_company(name: str | None) -> str:
    """Fold legal suffixes so "ManTech" and "ManTech International" group together."""
    c = re.sub(r"[,.]", " ", (name or "").lower().strip())
    c = re.sub(r"\s+", " ", _COMPANY_SUFFIX.sub(" ", c)).strip()
    return c or "(unknown)"


# --- Estimators -------------------------------------------------------------
def _preprocessor(categorical, numeric, text: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # min_frequency folds the long tail of rare categories into one bucket
    # instead of giving each its own noisy column.
    parts = [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5),
              categorical),
             ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), numeric)]
    if text:
        parts.append(("text", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                              max_features=2000,
                                              stop_words="english",
                                              sublinear_tf=True), "description"))
    return ColumnTransformer(parts, remainder="drop")


def _ridge(categorical, numeric, text: bool, log_target: bool):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("prep", _preprocessor(categorical, numeric, text)),
                     ("m", RidgeCV(alphas=np.logspace(-1, 4, 30)))])
    if log_target:
        # Pay is multiplicative, so a log-target twin fits the middle of the
        # distribution well; the dollar-target twin keeps the top tail honest.
        return TransformedTargetRegressor(pipe, func=np.log, inverse_func=np.exp)
    return pipe


def _blend(categorical, numeric, text: bool = True):
    """The production model: average of a log-target and a dollar-target ridge.

    Neither twin wins outright — the log fit has the better MAE, the dollar fit
    the better R2 — and averaging them beats both on both.
    """
    from sklearn.ensemble import VotingRegressor

    return VotingRegressor([
        ("log", _ridge(categorical, numeric, text, log_target=True)),
        ("usd", _ridge(categorical, numeric, text, log_target=False)),
    ])


def train_salary_model(jobs: pd.DataFrame, skills: pd.DataFrame):
    """Fit the *interactive* model used by the dashboard. Returns (pipe, n).

    Restricted to features a user can supply in a form — no description text.
    """
    feats = build_features(jobs, skills)
    pipe = _blend(INTERACTIVE_CATEGORICAL, INTERACTIVE_NUMERIC, text=False)
    pipe.fit(feats[INTERACTIVE_COLS], feats["salary"])
    return pipe, len(feats)


def predict_salary(pipe, role: str, seniority: int, remote, selected_skills,
                   location: str = "US", years_exp: float = float("nan")) -> float:
    """Predict salary for a hypothetical posting (used by the dashboard)."""
    row = row_features(role, seniority, remote, set(selected_skills),
                       location=location, years_exp=years_exp)
    return float(pipe.predict(pd.DataFrame([row])[INTERACTIVE_COLS])[0])


def _estimators():
    """{name: (estimator, columns)} — baselines first, then the real models."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    base_num = ["seniority", "remote", "n_skills"] + _SKILL_FLAGS
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                        max_leaf_nodes=15, min_samples_leaf=10,
                                        l2_regularization=1.0, random_state=42)
    from sklearn.pipeline import Pipeline
    return {
        "Baseline: global median":
            (DummyRegressor(strategy="median"), ["seniority"]),
        "Ridge: role+skills only (the old model)":
            (_ridge(["role"], base_num, False, False), ["role"] + base_num),
        "Ridge: + location/source/market":
            (_ridge(FULL_CATEGORICAL, INTERACTIVE_NUMERIC, False, False),
             FULL_CATEGORICAL + INTERACTIVE_NUMERIC),
        "Ridge: + description features":
            (_ridge(FULL_CATEGORICAL, FULL_NUMERIC, False, False),
             FULL_CATEGORICAL + FULL_NUMERIC),
        "HistGradientBoosting (no text)":
            (Pipeline([("prep", _preprocessor(FULL_CATEGORICAL, FULL_NUMERIC,
                                              False)), ("m", hgb)]),
             FULL_CATEGORICAL + FULL_NUMERIC),
        "Interactive model (dashboard inputs)":
            (_blend(INTERACTIVE_CATEGORICAL, INTERACTIVE_NUMERIC, text=False),
             INTERACTIVE_COLS),
        "Blend: ridge(log) + ridge(USD) + TF-IDF":
            (_blend(FULL_CATEGORICAL, FULL_NUMERIC, text=True), FULL_COLS),
    }


def evaluate(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

    jobs, skills = load_warehouse(), load_skills()
    if jobs.empty or skills.empty:
        log.error("Need data — run collect + extract_skills first.")
        return

    feats = build_features(jobs, skills)
    n = len(feats)
    if n < 20:
        log.warning("Only %d advertised-salary postings — results are indicative "
                    "at best. Keep collecting.", n)
    y = feats["salary"].to_numpy()
    groups = feats["company"].to_numpy()

    print("=" * 88)
    print(f"SALARY MODEL - predicting advertised salary, USD/yr  "
          f"(n={n}, {len(set(groups))} companies)")
    print("=" * 88)
    print(f"Target: median ${np.median(y):,.0f}, "
          f"range ${y.min():,.0f}-${y.max():,.0f}\n")

    print("TARGET AUDIT - what the salary rules kept and dropped")
    for _, r in analysis.salary_audit(jobs).iterrows():
        print(f"  {r['outcome']:<38}{r['postings']:>6}")
    print(f"  (FX table dated {analysis.FX_AS_OF}; band "
          f"${analysis.SALARY_BAND[0]:,}-${analysis.SALARY_BAND[1]:,})\n")

    print("Cross-validation is grouped by company: the same employer never")
    print("appears in both train and test. A plain random split scores higher")
    print("and is wrong -- both are shown so the gap is visible.\n")

    print(f"{'Model':<40}{'MAE':>11}{'R2 (grouped)':>15}{'R2 (random)':>14}")
    print("-" * 88)
    results = {}
    for name, (est, cols) in _estimators().items():
        X = feats[cols]
        grouped = [_oof(est, X, y, GroupKFold(5, shuffle=True, random_state=s),
                        groups) for s in seeds]
        random = _oof(est, X, y, KFold(5, shuffle=True, random_state=42), None)
        mae = float(np.mean([mean_absolute_error(y, p) for p in grouped]))
        r2g = float(np.mean([r2_score(y, p) for p in grouped]))
        r2r = r2_score(y, random)
        results[name] = (mae, r2g)
        print(f"{name:<40}{'$'+format(mae,',.0f'):>11}{r2g:>15.3f}{r2r:>14.3f}")

    base_mae, _ = results["Baseline: global median"]
    best = min((k for k in results if not k.startswith("Baseline")),
               key=lambda k: results[k][0])
    best_mae, best_r2 = results[best]
    print("-" * 88)
    print(f"Best: {best}")
    print(f"  MAE ${best_mae:,.0f} ({100*(base_mae-best_mae)/base_mae:+.0f}% vs. the "
          f"median baseline), grouped R2 {best_r2:.2f}, averaged over {len(seeds)} "
          f"CV seeds.\n")

    _report_coefficients(feats)
    _save_eval_plot(feats, y, groups, best)


def _oof(est, X, y, cv, groups):
    """Pooled out-of-fold predictions — steadier than averaging per-fold scores."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    pred = cross_val_predict(clone(est), X, y, cv=cv, groups=groups)
    return np.clip(pred, 1_000, 1_500_000)


def _report_coefficients(feats) -> None:
    """Fit the interpretable half of the blend and show the $ each feature moves."""
    pipe = _ridge(FULL_CATEGORICAL, FULL_NUMERIC, text=False, log_target=False)
    cols = FULL_CATEGORICAL + FULL_NUMERIC
    pipe.fit(feats[cols], feats["salary"])
    names = [n.split("__")[-1]
             for n in pipe.named_steps["prep"].get_feature_names_out()]
    pairs = sorted(zip(names, pipe.named_steps["m"].coef_), key=lambda p: -abs(p[1]))

    print("INTERPRETATION - ridge coefficients (~ $ vs. an average posting)")
    for name, c in pairs[:12]:
        print(f"  {name:<28} {'+' if c >= 0 else '-'}${abs(c):>8,.0f}")
    print("  (directional, not causal - correlated features, observational data)\n")


def _save_eval_plot(feats, y, groups, best_name) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    est, cols = _estimators()[best_name]
    y_pred = _oof(est, feats[cols], y, GroupKFold(5, shuffle=True, random_state=0),
                  groups)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y / 1000, y_pred / 1000, alpha=0.6, color="#2b6cb0", edgecolor="white")
    lim = [min(y.min(), y_pred.min()) / 1000 - 5, max(y.max(), y_pred.max()) / 1000 + 5]
    ax.plot(lim, lim, "--", color="gray", label="perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual salary ($k)")
    ax.set_ylabel("Out-of-fold prediction ($k)")
    # Escape the dollar sign: matplotlib reads a $...$ pair as math mode and
    # would silently italicise half the title.
    ax.set_title(f"Predicted vs. actual, held-out companies\n{best_name}\n"
                 f"MAE \\${mae:,.0f}, R² {r2:.2f} (one company-grouped split)",
                 fontsize=11)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "model_eval.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote reports/figures/model_eval.png")


if __name__ == "__main__":
    evaluate()
