"""Smoke test for the Streamlit dashboard.

The unit tests cover `src/` directly, which means a broken *dashboard* still
ships green: `app.py` can reference a constant or keyword argument that no
longer exists in `src/` and nothing catches it until the deployed app throws an
AttributeError in front of a visitor. This runs the whole script the way
Streamlit does and fails if any of it raises.

Skipped when the warehouse is empty, since the app legitimately calls
``st.stop()`` there and there is nothing to exercise.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.storage import load_warehouse

APP = Path(__file__).resolve().parent.parent / "app.py"

pytest.importorskip("streamlit.testing.v1",
                    reason="needs streamlit >= 1.28 for AppTest")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    if load_warehouse().empty:
        pytest.skip("empty warehouse - nothing for the dashboard to render")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    """The failure this exists to catch: app.py drifting from src/."""
    assert not app.exception, "\n".join(str(e) for e in app.exception)


def test_app_renders_every_tab(app):
    assert len(app.tabs) == 6


def test_headline_metrics_are_populated(app):
    labels = {m.label for m in app.metric}
    assert {"Postings", "Advertised salaries", "Companies"} <= labels
    postings = next(m for m in app.metric if m.label == "Postings")
    assert postings.value not in ("", "0", "—")


def test_salary_predictor_produces_a_number(app):
    """Exercises train_salary_model + predict_salary through the real UI path."""
    pred = next(m for m in app.metric
                if m.label == "Predicted advertised salary")
    assert pred.value.startswith("$")
    assert int(pred.value.lstrip("$").replace(",", "")) > 0
