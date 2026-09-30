from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results" / "risk" / "sensor_risk.csv").exists(), reason="study outputs not generated")
def test_dashboard_renders_from_study_outputs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "TWIN STATE" in labels and "NOVELTY DISTANCE" in labels
