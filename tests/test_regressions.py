"""Offline regression tests. Synthetic fixtures are NOT research evidence."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ppe import add_ppe_features
from src.regimes import add_regimes


def sample(n=2000):
    rng = np.random.default_rng(731)
    quote = rng.lognormal(8, 0.4, n)
    return pd.DataFrame({
        "open_time": pd.date_range("2026-01-05", periods=n, freq="min", tz="UTC"),
        "close": 100 * np.exp(rng.normal(0, 0.0005, n).cumsum()),
        "quote_volume": quote,
        "taker_buy_quote_volume": quote * rng.uniform(0.3, 0.7, n),
    })


def test_all_unresolved_targets_stay_missing():
    y = add_ppe_features(sample())
    assert y["fwd_ret_5m"].tail(5).isna().all()
    assert y["up_5m"].tail(5).isna().all()


def test_target_is_exact_five_minute_forward_log_return():
    x = sample()
    y = add_ppe_features(x)
    expected = np.log(x["close"].shift(-5) / x["close"])
    np.testing.assert_allclose(y["fwd_ret_5m"], expected, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("kind", ["ppe", "regimes"])
def test_future_mutation_does_not_change_past_features(kind):
    x = sample()
    changed = x.copy()
    cut = 1500
    changed.loc[cut:, "close"] *= 1.5
    changed.loc[cut:, ["quote_volume", "taker_buy_quote_volume"]] *= 3
    transform = add_ppe_features if kind == "ppe" else add_regimes
    before, after = transform(x), transform(changed)
    columns = [c for c in before.columns if c not in {"fwd_ret_5m", "up_5m"}]
    pd.testing.assert_frame_equal(before.loc[:cut-1, columns], after.loc[:cut-1, columns])


def test_zero_activity_never_yields_infinite_efficiency():
    x = sample()
    x.loc[300:320, ["quote_volume", "taker_buy_quote_volume"]] = 0.0
    y = add_ppe_features(x)
    columns = ["ppe_1", "ppe_3", "ppe_5", "ppe_15", "taker_imbalance"]
    assert not np.isinf(y[columns].to_numpy(dtype=float)).any()


def test_study_cli_produces_reports_offline(tmp_path):
    """Test execution only; never upload the synthetic report as a market result."""
    x = sample()
    data = tmp_path / "data" / "raw"
    data.mkdir(parents=True)
    x.to_csv(data / "btcusdt_1m.csv.gz", index=False, compression="gzip")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / "artifacts"
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["rows"] == len(x)
    assert 0 < meta["sample_rows"] <= len(x) // 5
    for filename in ("ppe_overall.csv", "ppe_regimes.csv"):
        report = pd.read_csv(out / filename)
        assert not report.empty
        assert report["p_up_5m"].between(0, 1).all()
