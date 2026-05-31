from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.drawdown import drawdown_episodes, equity_drawdown_statistics, max_drawdown_duration


def test_equity_drawdown_statistics_reports_episode_and_underwater_state() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-25", periods=6),
            "net_value": [1.0, 1.2, 1.0, 0.9, 1.21, 1.15],
        }
    )
    net_value = equity["net_value"]

    stats = equity_drawdown_statistics(equity, net_value)

    drawdown = net_value / net_value.cummax() - 1.0
    assert stats["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert stats["max_drawdown_duration"] == 2.0
    assert stats["max_drawdown_start_at"] == "2026-05-26 00:00:00"
    assert stats["max_drawdown_trough_at"] == "2026-05-28 00:00:00"
    assert stats["max_drawdown_recovery_at"] == "2026-05-29 00:00:00"
    assert stats["current_drawdown"] == pytest.approx(1.15 / 1.21 - 1.0)
    assert stats["current_underwater_bars"] == 1.0
    assert stats["avg_drawdown"] == pytest.approx(drawdown.mean())
    assert stats["ulcer_index"] == pytest.approx((drawdown.pow(2).mean()) ** 0.5)
    assert stats["time_under_water_ratio"] == pytest.approx(float(drawdown.lt(0).mean()))


def test_equity_drawdown_statistics_uses_trade_number_when_date_is_missing() -> None:
    equity = pd.DataFrame({"trade_no": [0, 1, 2, 3], "net_value": [1.0, 1.3, 1.1, 1.31]})

    stats = equity_drawdown_statistics(equity, equity["net_value"])

    assert stats["max_drawdown_start_at"] == "1"
    assert stats["max_drawdown_trough_at"] == "2"
    assert stats["max_drawdown_recovery_at"] == "3"


def test_equity_drawdown_statistics_keeps_labels_aligned_after_invalid_values() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-25", periods=4),
            "net_value": [1.0, 1.2, None, 1.1],
        }
    )

    stats = equity_drawdown_statistics(equity, equity["net_value"])

    assert stats["max_drawdown_start_at"] == "2026-05-26 00:00:00"
    assert stats["max_drawdown_trough_at"] == "2026-05-28 00:00:00"


def test_equity_drawdown_statistics_returns_stable_empty_values() -> None:
    stats = equity_drawdown_statistics(pd.DataFrame(), pd.Series(dtype=float))

    assert stats == {
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0.0,
        "max_drawdown_start_at": "",
        "max_drawdown_trough_at": "",
        "max_drawdown_recovery_at": "",
        "current_drawdown": 0.0,
        "current_underwater_bars": 0.0,
        "avg_drawdown": 0.0,
        "ulcer_index": 0.0,
        "time_under_water_ratio": 0.0,
    }


def test_max_drawdown_duration_counts_consecutive_underwater_points() -> None:
    assert max_drawdown_duration(pd.Series([1.0, 1.2, 1.1, 1.05, 1.3, 1.25])) == 2


def test_drawdown_episodes_reports_ranked_recovery_windows() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-25", periods=9),
            "net_value": [1.0, 1.2, 1.0, 0.9, 1.21, 1.15, 1.22, 1.1, 1.23],
        }
    )

    episodes = drawdown_episodes(equity, equity["net_value"], limit=2)

    assert episodes["episode_rank"].tolist() == [1, 2]
    assert episodes["start_at"].tolist() == ["2026-05-26 00:00:00", "2026-05-31 00:00:00"]
    assert episodes["trough_at"].tolist() == ["2026-05-28 00:00:00", "2026-06-01 00:00:00"]
    assert episodes["recovery_at"].tolist() == ["2026-05-29 00:00:00", "2026-06-02 00:00:00"]
    assert episodes["depth"].tolist() == pytest.approx([0.9 / 1.2 - 1.0, 1.1 / 1.22 - 1.0])
    assert episodes["underwater_bars"].tolist() == [2, 1]
    assert episodes["recovery_bars"].tolist() == [3, 2]
    assert episodes["recovered"].tolist() == [True, True]


def test_drawdown_episodes_keeps_unrecovered_current_drawdown() -> None:
    equity = pd.DataFrame(
        {
            "trade_no": [0, 1, 2, 3],
            "net_value": [1.0, 1.2, 1.1, 1.05],
        }
    )

    episodes = drawdown_episodes(equity, equity["net_value"])

    assert episodes.loc[0, "start_at"] == "1"
    assert episodes.loc[0, "trough_at"] == "3"
    assert episodes.loc[0, "recovery_at"] == ""
    assert episodes.loc[0, "recovered"] is False
