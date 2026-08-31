"""Data Job Market Intelligence — interactive dashboard.

    streamlit run app.py

Reuses the same analysis / figures / model code as the CLI pipeline, so the
dashboard never drifts from the numbers in the report. Deployed for free on
Streamlit Community Cloud (see README "Deploy").
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable whether run locally or on Streamlit Cloud.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import analysis, figures, model  # noqa: E402
from src.skills import TAXONOMY  # noqa: E402
from src.storage import load_skills, load_warehouse  # noqa: E402

st.set_page_config(page_title="Data Job Market Intelligence",
                   page_icon="📊", layout="wide")

SKILL_NAMES = sorted(s.name for s in TAXONOMY)
SENIORITY = {"Junior": 1, "Mid": 2, "Senior": 3, "Principal / Lead": 4}


@st.cache_data
def get_data():
    return load_warehouse(), load_skills()


@st.cache_resource
def get_model():
    jobs, skills = load_warehouse(), load_skills()
    return model.train_salary_model(jobs, skills)


def apply_filters(jobs, skills, markets, roles, sources, remote_choice):
    j = jobs.copy()
    j["role"] = j["title"].map(analysis.classify_role)
    if markets and "market" in j.columns:
        j = j[j["market"].isin(markets)]
    if roles:
        j = j[j["role"].isin(roles)]
    if sources:
        j = j[j["source"].isin(sources)]
    if remote_choice == "Remote only":
        j = j[j["remote"]]
    elif remote_choice == "On-site / hybrid":
        j = j[~j["remote"]]
    js = skills[skills["job_id"].isin(j["job_id"])]
    return j, js


def show_fig(fig, msg="Not enough data for this view with the current filters."):
    if fig is None:
        st.info(msg)
    else:
        st.pyplot(fig)


# --------------------------------------------------------------------------- #
jobs, skills = get_data()

st.title("📊 Data Job Market Intelligence")
st.caption("What skills are in demand for data roles, what do they pay, and how "
           "do roles compare, from public job-posting APIs, end to end.")

if jobs.empty:
    st.error("No data found. Run `python -m src.collect` then "
             "`python -m src.extract_skills` to build the warehouse.")
    st.stop()

with st.expander("ℹ️ How to read this (sources, caveats)"):
    st.markdown(
        "- **Sources:** six public APIs (Remotive, Jobicy, RemoteOK, JSearch, "
        "Jooble, Adzuna). Adzuna's salaries are mostly *estimates*, which are "
        "flagged and excluded from every salary figure.\n"
        "- **Salary numbers** use only *genuinely advertised* pay, normalized to "
        "USD/year; the Pay tab's salary audit shows exactly what was excluded.\n"
        "- **Skill extraction** is dictionary-based, so demand %s are lower bounds.\n"
        "- Treat everything as directional; it firms up as the collector "
        "accumulates data.")

# --- Sidebar filters ---
st.sidebar.header("Filters")
markets = []
if "market" in jobs.columns:
    markets = st.sidebar.multiselect("Market", sorted(jobs["market"].dropna().unique()),
                                     default=[], help="e.g. Indonesia vs Remote vs Australia")
all_roles = sorted(jobs["title"].map(analysis.classify_role).unique())
roles = st.sidebar.multiselect("Role", all_roles, default=[])
sources = st.sidebar.multiselect("Source", sorted(jobs["source"].unique()), default=[])
remote_choice = st.sidebar.radio("Work type", ["All", "Remote only", "On-site / hybrid"])
fjobs, fskills = apply_filters(jobs, skills, markets, roles, sources, remote_choice)

if "last_seen_at" in jobs.columns and not jobs["last_seen_at"].isna().all():
    st.sidebar.caption(f"Data last collected: {str(jobs['last_seen_at'].max())[:10]}")

# --- Headline metrics ---
adv = len(analysis.build_salary_table(fjobs))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Postings", f"{len(fjobs):,}")
c2.metric("Advertised salaries", f"{adv:,}")
c3.metric("Remote", f"{fjobs['remote'].mean() * 100:.0f}%" if len(fjobs) else "—")
c4.metric("Companies", f"{fjobs['company'].nunique():,}")

if fjobs.empty:
    st.warning("No postings match these filters.")
    st.stop()

# --- Tabs ---
t_demand, t_pay, t_roles, t_cooc, t_predict, t_explore = st.tabs(
    ["📈 Demand", "💸 Pay by skill", "🧑‍💼 Roles", "🔗 Co-occurrence",
     "🤖 Salary predictor", "🗂 Explore"])

with t_demand:
    top = st.slider("How many skills", 5, 25, 15)
    rich_only = st.checkbox(
        f"Exclude truncated-description postings (< {analysis.RICH_DESC_CHARS} chars, e.g. Adzuna)",
        value=False,
        help="In a short teaser a missing skill is missing data, not 'not required'. "
             "Turn on to measure demand only over detailed postings.")
    show_fig(figures.build_demand(fjobs, fskills, top=top, rich_only=rich_only))
    with st.expander("📋 Data quality by source (description length & skill coverage)"):
        st.caption("Some sources (Adzuna) return short teasers, so their low skill "
                   "counts are missing data, not low demand.")
        st.dataframe(analysis.quality_by_source(jobs, skills),
                     hide_index=True, width="stretch")

with t_pay:
    st.caption("Median advertised salary among postings mentioning each skill "
               "(USD/yr, skills with ≥5 such postings).")
    show_fig(figures.build_salary_by_skill(fjobs, fskills),
             "Not enough advertised-salary postings under these filters yet.")
    with st.expander("📋 Salary audit (what the salary rules kept and dropped)"):
        st.caption(f"Non-USD and non-annual pay is converted with a static FX "
                   f"table dated {analysis.FX_AS_OF}, static so a re-run in six "
                   f"months reproduces today's chart. Salaries outside "
                   f"${analysis.SALARY_BAND[0]:,}–${analysis.SALARY_BAND[1]:,} are "
                   f"treated as parse failures (a $162/yr engineer is a lost "
                   f"'per hour'), and counted here rather than dropped silently.")
        st.dataframe(analysis.salary_audit(fjobs), hide_index=True, width="stretch")

with t_roles:
    show_fig(figures.build_roles(fjobs))

with t_cooc:
    st.caption("How often the most common skills appear in the same posting.")
    show_fig(figures.build_cooccurrence(fskills))

with t_predict:
    st.subheader("Estimate a salary")
    st.caption("Two ridge regressions (one on log-pay, one on dollars) averaged, "
               "trained on advertised salaries normalized to USD/year. "
               "Directional only; see caveats above.")
    pipe, n_train = get_model()
    pc1, pc2 = st.columns(2)
    with pc1:
        role = st.selectbox("Role", ["Data Analyst", "Data Scientist",
                                     "Data Engineer", "ML Engineer", "Other"])
        seniority = st.select_slider("Seniority", list(SENIORITY), value="Mid")
        location = st.selectbox(
            "Location", model.LOCATIONS, index=model.LOCATIONS.index("US"),
            help="The single largest driver of advertised pay in this data: "
                 "a US posting carries roughly +$38k over the average.")
        remote = st.checkbox("Remote", value=True)
    with pc2:
        picked = st.multiselect("Skills", SKILL_NAMES,
                                default=["Python", "SQL"])
        years = st.slider("Years of experience required", 0, 15, 3)
    pred = model.predict_salary(pipe, role, SENIORITY[seniority], remote, picked,
                                location=location, years_exp=float(years))
    st.metric("Predicted advertised salary", f"${pred:,.0f}")
    st.caption(f"Trained on {n_train} advertised-salary postings · "
               "MAE ≈ $39k, R² 0.32 when cross-validated across *held-out "
               "companies* (the honest split, see the README). The offline "
               "model, which also reads the job description, reaches R² 0.46. "
               "Don't quote either as fact.")

with t_explore:
    st.caption("Filtered postings. Use the sidebar to narrow down.")
    cols = [c for c in ["market", "source", "title", "company", "location", "remote",
                        "salary_min", "salary_max", "salary_currency", "url"]
            if c in fjobs.columns]
    st.dataframe(fjobs[cols].reset_index(drop=True), width="stretch",
                 hide_index=True)

st.divider()
st.caption("Built end-to-end: collection → skill extraction → analysis → model → "
           "dashboard. Code: see the project README.")
