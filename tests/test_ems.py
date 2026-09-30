import zipfile

import numpy as np

from src.ems import parse_env, read_archive
from src.risk_engine import gated_risk_config, overall_state, sensor_state

ENV = ("WS    [V]      TBC1  [\xb0C]     TWS3  [\xb0C]     TWS3  [\xb0C]     \n"
       "0.0 6.5 3276.7 5.5 -244.3 9.9\n"
       "0.0 6.7 3276.7 5.7 -244.2 9.9\n"
       "\n\nEnvScan started : Mon Nov 10 14:00:13 1997\n")


def test_parse_env_sentinels_duplicates_and_timestamp():
    out = parse_env(ENV)
    assert out["timestamp"].year == 1997 and out["timestamp"].hour == 14
    assert np.isclose(out["values"]["TBC1"], 6.6)
    assert np.isnan(out["values"]["TWS3"])          # 3276.7 sentinel removed
    assert np.isclose(out["values"]["TWS3_2"], 5.6)  # duplicate header name disambiguated


def test_read_archive(tmp_path):
    z = tmp_path / "01C14.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("01C1403.aaa", "01C1403.aaa\n4\n0.01\n1\n2\n3\n4\n")
        f.writestr("01C14PRE.env", ENV.encode("latin1"))
    rec = read_archive(z)
    assert rec["channels"]["03"][1] == 0.01 and rec["channels"]["03"][0].tolist() == [1, 2, 3, 4]
    assert rec["timestamp"].month == 11


def test_risk_gate_and_states():
    cfg, inc = gated_risk_config({"bocpd_weight": .3, "ml_weight": .5, "attribution_weight": .2}, 0.62, 0.7)
    assert not inc and cfg["ml_weight"] == 0
    cfg, inc = gated_risk_config({"ml_weight": .5}, 0.75, 0.7)
    assert inc and cfg["ml_weight"] == .5
    assert sensor_state(.9, .9, .1, .8) == "ANOMALY"
    assert sensor_state(.9, .9, .1, .8, use_classifier=False) == "WARNING"
    assert overall_state(["NORMAL", "WARNING"]) == "POTENTIAL ANOMALY"
    assert overall_state(["NORMAL"] * 9 + ["WARNING"]) == "WATCH"


def test_attribution_cannot_raise_risk_without_change():
    from src.risk_engine import calculate_risk, gated_risk_config
    cfg, _ = gated_risk_config({"bocpd_weight": .3, "ml_weight": .5, "attribution_weight": .2}, 0.6)
    assert calculate_risk(0.0, None, 1.0, cfg)["risk_score"] == 0.0
    r = calculate_risk(1.0, None, 1.0, cfg)["risk_score"]
    assert abs(r - 1.0) < 1e-12
    assert abs(calculate_risk(0.5, None, 0.5, cfg)["risk_score"] - 0.5 * (0.3 + 0.2 * 0.5) / 0.5) < 1e-12
