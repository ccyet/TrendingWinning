from __future__ import annotations

import pytest
import pandas as pd

from trending_winning.backtest.stats import (
    build_equity_curve,
    compute_decision_reason_statistics,
    compute_equity_statistics,
    compute_grouped_trade_statistics,
    compute_period_return_statistics,
    compute_period_returns,
    compute_trade_statistics,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)


def test_build_equity_curve_keeps_trade_dates_for_period_statistics() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-05-30 10:00:00", "2026-06-02 10:00:00"]),
            "exit_date": pd.to_datetime(["2026-06-01 10:00:00", "2026-06-03 10:00:00"]),
            "return_pct": [10.0, -5.0],
        }
    )

    equity = build_equity_curve(trades, initial_equity=2.0)

    assert equity["trade_no"].tolist() == [0, 1, 2]
    assert equity["date"].tolist() == [
        pd.Timestamp("2026-05-30 10:00:00"),
        pd.Timestamp("2026-06-01 10:00:00"),
        pd.Timestamp("2026-06-03 10:00:00"),
    ]
    assert equity["net_value"].tolist() == pytest.approx([2.0, 2.2, 2.09])
    monthly = compute_period_returns(equity, freq="M").set_index("period")
    assert monthly.loc["2026-05", "return"] == pytest.approx(0.0)
    assert monthly.loc["2026-06", "return"] == pytest.approx(2.09 / 2.0 - 1.0)


def test_compute_equity_statistics_preserves_same_date_trade_order() -> None:
    equity = pd.DataFrame(
        {
            "trade_no": [0, 1, 2],
            "date": pd.to_datetime(["2026-05-30", "2026-05-30", "2026-05-30"]),
            "net_value": [1.0, 1.1, 1.045],
        }
    )

    stats = compute_equity_statistics(equity, periods_per_year=2)

    assert stats["total_return"] == pytest.approx(0.045)
    assert stats["max_drawdown"] == pytest.approx(1.045 / 1.1 - 1.0)


def test_compute_trade_statistics_reports_risk_adjusted_and_streak_metrics() -> None:
    trades = pd.DataFrame(
        {
            "return_pct": [5.0, -2.0, -1.0, 4.0, -3.0],
            "holding_bars": [3, 2, 4, 2, 5],
            "r_multiple": [1.5, -0.8, -0.3, 1.2, -1.0],
            "mae_pct": [-1.0, -2.5, -1.2, -0.8, -3.4],
            "mfe_pct": [6.0, 1.5, 0.4, 4.8, 0.7],
            "mae_r": [-0.3, -1.0, -0.4, -0.2, -1.1],
            "mfe_r": [1.8, 0.6, 0.1, 1.4, 0.2],
        }
    )

    stats = compute_trade_statistics(trades)

    assert stats["trade_count"] == 5.0
    assert stats["gross_profit"] == 0.09
    assert stats["gross_loss"] == 0.06
    assert stats["profit_factor"] == 1.5
    assert stats["max_consecutive_losses"] == 2.0
    assert stats["max_consecutive_wins"] == 1.0
    assert stats["avg_holding_bars"] == 3.2
    assert stats["return_std"] > 0
    assert "sharpe_per_trade" in stats
    assert "sortino_per_trade" in stats
    assert "max_drawdown_duration" in stats
    assert stats["avg_r_multiple"] == pytest.approx(0.12)
    assert stats["median_r_multiple"] == pytest.approx(-0.3)
    assert stats["avg_mae_pct"] == pytest.approx(-1.78)
    assert stats["avg_mfe_pct"] == pytest.approx(2.68)
    assert stats["avg_mae_r"] == pytest.approx(-0.6)
    assert stats["avg_mfe_r"] == pytest.approx(0.82)


def test_compute_trade_statistics_reports_return_distribution_and_tail_risk() -> None:
    trades = pd.DataFrame(
        {
            "return_pct": [-10.0, -5.0, 0.0, 5.0, 20.0],
            "holding_bars": [1, 1, 1, 1, 1],
        }
    )

    stats = compute_trade_statistics(trades)

    assert stats["return_p05"] == pytest.approx(-0.09)
    assert stats["return_p25"] == pytest.approx(-0.05)
    assert stats["return_p50"] == pytest.approx(0.0)
    assert stats["return_p75"] == pytest.approx(0.05)
    assert stats["return_p95"] == pytest.approx(0.17)
    assert stats["cvar_95"] == pytest.approx(-0.1)


def test_compute_trade_statistics_reports_sample_confidence_metrics() -> None:
    trades = pd.DataFrame(
        {
            "return_pct": [5.0, -2.0, 4.0, -1.0],
            "holding_bars": [1, 1, 1, 1],
        }
    )

    stats = compute_trade_statistics(trades)

    returns = trades["return_pct"] / 100.0
    win_rate = 0.5
    z_score = 1.96
    wilson_denominator = 1 + z_score**2 / 4
    wilson_center = win_rate + z_score**2 / 8
    wilson_margin = z_score * ((win_rate * (1 - win_rate) / 4 + z_score**2 / 64) ** 0.5)
    expected_se = returns.std(ddof=1) / (len(returns) ** 0.5)

    assert stats["win_rate_ci_lower"] == pytest.approx((wilson_center - wilson_margin) / wilson_denominator)
    assert stats["win_rate_ci_upper"] == pytest.approx((wilson_center + wilson_margin) / wilson_denominator)
    assert stats["avg_return_standard_error"] == pytest.approx(expected_se)
    assert stats["avg_return_ci_lower"] == pytest.approx(returns.mean() - z_score * expected_se)
    assert stats["avg_return_ci_upper"] == pytest.approx(returns.mean() + z_score * expected_se)
    assert stats["positive_expectancy_probability"] > 0.5


def test_compute_grouped_trade_statistics_includes_sample_confidence_metrics() -> None:
    trades = pd.DataFrame(
        {
            "strategy_name": ["trend", "trend", "range", "range"],
            "return_pct": [5.0, -2.0, 4.0, -1.0],
            "holding_bars": [1, 1, 1, 1],
        }
    )

    grouped = compute_grouped_trade_statistics(trades, by="strategy_name")

    assert "win_rate_ci_lower" in grouped.columns
    assert "avg_return_ci_upper" in grouped.columns
    assert grouped.set_index("strategy_name").loc["trend", "avg_return_standard_error"] > 0


def test_compute_trade_statistics_reports_r_profit_factor_and_sqn() -> None:
    trades = pd.DataFrame(
        {
            "return_pct": [5.0, -2.0, 1.0, -1.0],
            "holding_bars": [1, 1, 1, 1],
            "r_multiple": [2.0, -1.0, 0.5, -0.5],
        }
    )

    stats = compute_trade_statistics(trades)

    expected_r = pd.Series([2.0, -1.0, 0.5, -0.5])
    expected_sqn = (len(expected_r) ** 0.5) * expected_r.mean() / expected_r.std(ddof=0)
    assert stats["r_profit_factor"] == pytest.approx(2.5 / 1.5)
    assert stats["system_quality_number"] == pytest.approx(expected_sqn)


def test_compute_trade_statistics_reports_portfolio_capital_contribution_metrics() -> None:
    trades = pd.DataFrame(
        {
            "strategy_name": ["trend_signal_bar", "range_signal_bar"],
            "return_pct": [2.0, -1.5],
            "raw_return_pct": [10.0, -5.0],
            "capital_fraction": [0.2, 0.3],
            "margin_fraction": [0.2, 0.45],
            "holding_bars": [2, 3],
        }
    )

    stats = compute_trade_statistics(trades)
    grouped = compute_grouped_trade_statistics(trades, by="strategy_name").set_index("strategy_name")

    assert stats["return_contribution"] == pytest.approx(0.005)
    assert stats["capital_turnover"] == pytest.approx(0.5)
    assert stats["avg_capital_fraction"] == pytest.approx(0.25)
    assert stats["max_capital_fraction"] == pytest.approx(0.3)
    assert stats["margin_turnover"] == pytest.approx(0.65)
    assert stats["avg_margin_fraction"] == pytest.approx(0.325)
    assert stats["max_margin_fraction"] == pytest.approx(0.45)
    assert stats["capital_exposure_bars"] == pytest.approx(1.3)
    assert stats["margin_exposure_bars"] == pytest.approx(1.75)
    assert stats["avg_capital_exposure_per_trade"] == pytest.approx(0.65)
    assert stats["avg_margin_exposure_per_trade"] == pytest.approx(0.875)
    assert stats["return_per_exposure_bar"] == pytest.approx(0.005 / 5.0)
    assert stats["return_per_capital_exposure_bar"] == pytest.approx(0.005 / 1.3)
    assert stats["return_per_margin_exposure_bar"] == pytest.approx(0.005 / 1.75)
    assert stats["capital_weighted_raw_return"] == pytest.approx(0.01)
    assert grouped.loc["trend_signal_bar", "return_contribution"] == pytest.approx(0.02)
    assert grouped.loc["trend_signal_bar", "capital_weighted_raw_return"] == pytest.approx(0.1)
    assert grouped.loc["range_signal_bar", "return_contribution"] == pytest.approx(-0.015)


def test_compute_trade_statistics_reports_single_strategy_return_contribution() -> None:
    trades = pd.DataFrame(
        {
            "event_type": ["bull_h2_setup", "bull_h2_setup", "failed_breakout"],
            "return_pct": [5.0, -2.0, 1.0],
            "holding_bars": [3, 2, 1],
        }
    )

    stats = compute_trade_statistics(trades)
    grouped = compute_grouped_trade_statistics(trades, by="event_type").set_index("event_type")

    assert stats["return_contribution"] == pytest.approx(0.04)
    assert stats["return_per_exposure_bar"] == pytest.approx(0.04 / 6.0)
    assert stats["return_per_capital_exposure_bar"] == pytest.approx(0.0)
    assert stats["return_per_margin_exposure_bar"] == pytest.approx(0.0)
    assert grouped.loc["bull_h2_setup", "return_contribution"] == pytest.approx(0.03)
    assert grouped.loc["bull_h2_setup", "return_per_exposure_bar"] == pytest.approx(0.03 / 5.0)
    assert grouped.loc["failed_breakout", "return_contribution"] == pytest.approx(0.01)


def test_compute_grouped_trade_statistics_reports_strategy_and_symbol_breakdowns() -> None:
    trades = pd.DataFrame(
        {
            "strategy_name": ["trend", "trend", "range", "range"],
            "stock_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "return_pct": [5.0, -2.0, 1.0, 3.0],
            "holding_bars": [3, 2, 1, 4],
        }
    )

    by_strategy = compute_grouped_trade_statistics(trades, by="strategy_name")
    by_symbol = compute_grouped_trade_statistics(trades, by="stock_code")

    assert by_strategy.set_index("strategy_name").loc["trend", "trade_count"] == 2.0
    assert by_strategy.set_index("strategy_name").loc["trend", "total_return"] == pytest.approx(0.029)
    assert by_strategy.set_index("strategy_name").loc["range", "win_rate"] == 1.0
    assert by_symbol.set_index("stock_code").loc["000002.SZ", "trade_count"] == 2.0


def test_compute_grouped_trade_statistics_supports_setup_breakdown_fields() -> None:
    trades = pd.DataFrame(
        {
            "detector_name": ["trend", "trend", "trend", "range"],
            "event_type": ["bull_h2_setup", "bull_h2_setup", "bull_h1_setup", "failed_breakout"],
            "side": ["long", "long", "long", "short"],
            "return_pct": [5.0, -2.0, 1.0, 3.0],
            "holding_bars": [3, 2, 1, 4],
        }
    )

    setup = compute_grouped_trade_statistics(trades, by=("detector_name", "event_type", "side"))
    by_setup = setup.set_index(["detector_name", "event_type", "side"])

    assert setup.columns[:3].tolist() == ["detector_name", "event_type", "side"]
    assert by_setup.loc[("trend", "bull_h2_setup", "long"), "trade_count"] == 2.0
    assert by_setup.loc[("trend", "bull_h2_setup", "long"), "total_return"] == pytest.approx(0.029)
    assert by_setup.loc[("range", "failed_breakout", "short"), "win_rate"] == 1.0


def test_compute_equity_statistics_reports_annualized_return_and_exposure_metrics() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29"]),
            "net_value": [1.0, 1.02, 1.01, 1.05, 1.04],
            "gross_exposure": [0.0, 0.5, 1.0, 0.25, 0.0],
            "open_positions": [0, 1, 2, 1, 0],
        }
    )

    stats = compute_equity_statistics(equity, periods_per_year=4)
    period_returns = equity["net_value"].pct_change().dropna()
    expected_volatility = period_returns.std(ddof=0) * (4**0.5)
    drawdown = equity["net_value"] / equity["net_value"].cummax() - 1.0
    expected_ulcer_index = (drawdown.pow(2).mean()) ** 0.5

    assert stats["total_return"] == pytest.approx(0.04)
    assert stats["annualized_return"] == pytest.approx(0.04)
    assert stats["annualized_volatility"] == pytest.approx(expected_volatility)
    assert stats["calmar_ratio"] == pytest.approx(0.04 / abs(stats["max_drawdown"]))
    assert stats["avg_drawdown"] == pytest.approx(drawdown.mean())
    assert stats["ulcer_index"] == pytest.approx(expected_ulcer_index)
    assert stats["time_under_water_ratio"] == pytest.approx(2 / 5)
    assert stats["avg_gross_exposure"] == pytest.approx(0.35)
    assert stats["max_gross_exposure"] == pytest.approx(1.0)
    assert stats["exposure_bar_ratio"] == pytest.approx(0.6)
    assert stats["avg_open_positions"] == pytest.approx(0.8)
    assert stats["max_open_positions"] == pytest.approx(2.0)


def test_compute_equity_statistics_reports_cash_and_net_exposure_ratios() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28"]),
            "net_value": [1.0, 1.1, 1.2, 1.05],
            "cash": [1.0, 0.3, 1.55, 0.5],
            "position_value": [0.0, 0.8, -0.35, 0.55],
        }
    )

    stats = compute_equity_statistics(equity, periods_per_year=4)

    cash_ratio = equity["cash"] / equity["net_value"]
    net_exposure = equity["position_value"] / equity["net_value"]
    assert stats["avg_cash_ratio"] == pytest.approx(cash_ratio.mean())
    assert stats["min_cash_ratio"] == pytest.approx(cash_ratio.min())
    assert stats["max_cash_ratio"] == pytest.approx(cash_ratio.max())
    assert stats["avg_net_exposure"] == pytest.approx(net_exposure.mean())
    assert stats["min_net_exposure"] == pytest.approx(net_exposure.min())
    assert stats["max_net_exposure"] == pytest.approx(net_exposure.max())


def test_compute_equity_statistics_reports_drawdown_episode_boundaries() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-06-01"]
            ),
            "trade_no": [0, 1, 2, 3, 4, 5],
            "net_value": [1.0, 1.2, 1.1, 0.9, 1.21, 1.15],
        }
    )

    stats = compute_equity_statistics(equity, periods_per_year=252)

    assert stats["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert stats["max_drawdown_start_at"] == "2026-05-26 00:00:00"
    assert stats["max_drawdown_trough_at"] == "2026-05-28 00:00:00"
    assert stats["max_drawdown_recovery_at"] == "2026-05-29 00:00:00"
    assert stats["current_drawdown"] == pytest.approx(1.15 / 1.21 - 1.0)
    assert stats["current_underwater_bars"] == 1.0


def test_compute_equity_statistics_keeps_empty_drawdown_episode_fields_stable() -> None:
    stats = compute_equity_statistics(pd.DataFrame())

    assert stats["max_drawdown_start_at"] == ""
    assert stats["max_drawdown_trough_at"] == ""
    assert stats["max_drawdown_recovery_at"] == ""
    assert stats["current_drawdown"] == 0.0
    assert stats["current_underwater_bars"] == 0.0


def test_compute_equity_statistics_keeps_single_point_curve_numeric() -> None:
    stats = compute_equity_statistics(pd.DataFrame({"date": [pd.Timestamp("2026-05-25")], "net_value": [1.0]}))

    assert stats["total_return"] == 0.0
    assert stats["equity_return_std"] == 0.0
    assert stats["annualized_return"] == 0.0
    assert stats["annualized_volatility"] == 0.0
    assert stats["avg_gross_exposure"] == 0.0
    assert stats["avg_cash_ratio"] == 0.0
    assert stats["avg_net_exposure"] == 0.0


def test_compute_period_returns_reports_monthly_equity_returns() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-29", "2026-05-31", "2026-06-01", "2026-06-30"]),
            "net_value": [1.0, 1.1, 1.2, 1.5],
        }
    )

    monthly = compute_period_returns(equity, freq="M")

    assert monthly["period"].tolist() == ["2026-05", "2026-06"]
    by_period = monthly.set_index("period")
    assert by_period.loc["2026-05", "return"] == pytest.approx(0.1)
    assert by_period.loc["2026-06", "start_net_value"] == pytest.approx(1.1)
    assert by_period.loc["2026-06", "return"] == pytest.approx((1.5 - 1.1) / 1.1)


def test_compute_period_returns_reports_period_drawdown_and_observation_count() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01"]),
            "net_value": [1.0, 1.2, 1.1, 1.3],
        }
    )

    monthly = compute_period_returns(equity, freq="M")

    by_period = monthly.set_index("period")
    assert by_period.loc["2026-05", "observation_count"] == 3
    assert by_period.loc["2026-05", "max_drawdown"] == pytest.approx(1.1 / 1.2 - 1.0)
    assert by_period.loc["2026-06", "observation_count"] == 1
    assert by_period.loc["2026-06", "max_drawdown"] == 0.0


def test_compute_period_return_statistics_reports_stability_summary() -> None:
    period_returns = pd.DataFrame(
        {
            "period": ["2026-01", "2026-02", "2026-03", "2026-04"],
            "return": [0.10, -0.05, 0.0, 0.20],
            "max_drawdown": [-0.02, -0.08, 0.0, -0.01],
            "observation_count": [2, 3, 1, 4],
            "end_net_value": [1.1, 1.045, 1.045, 1.254],
        }
    )

    stats = compute_period_return_statistics(period_returns, prefix="monthly")

    assert stats["monthly_count"] == 4.0
    assert stats["monthly_positive_count"] == 2.0
    assert stats["monthly_negative_count"] == 1.0
    assert stats["monthly_win_rate"] == pytest.approx(0.5)
    assert stats["monthly_avg_return"] == pytest.approx(0.0625)
    assert stats["monthly_return_std"] == pytest.approx(period_returns["return"].std(ddof=0))
    assert stats["monthly_best_return"] == pytest.approx(0.20)
    assert stats["monthly_best_return_period"] == "2026-04"
    assert stats["monthly_worst_return"] == pytest.approx(-0.05)
    assert stats["monthly_worst_return_period"] == "2026-02"
    assert stats["monthly_avg_drawdown"] == pytest.approx(-0.0275)
    assert stats["monthly_worst_drawdown"] == pytest.approx(-0.08)
    assert stats["monthly_worst_drawdown_period"] == "2026-02"
    assert stats["monthly_avg_observation_count"] == pytest.approx(2.5)
    assert stats["monthly_max_consecutive_gains"] == 1.0
    assert stats["monthly_max_consecutive_losses"] == 1.0
    assert stats["monthly_max_recovery_periods"] == 2.0
    assert stats["monthly_underwater_ratio"] == pytest.approx(0.5)
    assert stats["monthly_current_underwater_periods"] == 0.0


def test_compute_period_return_statistics_handles_empty_period_table() -> None:
    stats = compute_period_return_statistics(pd.DataFrame(), prefix="monthly")

    assert stats == {
        "monthly_count": 0.0,
        "monthly_win_rate": 0.0,
        "monthly_positive_count": 0.0,
        "monthly_negative_count": 0.0,
        "monthly_avg_return": 0.0,
        "monthly_return_std": 0.0,
        "monthly_best_return": 0.0,
        "monthly_best_return_period": "",
        "monthly_worst_return": 0.0,
        "monthly_worst_return_period": "",
        "monthly_avg_drawdown": 0.0,
        "monthly_worst_drawdown": 0.0,
        "monthly_worst_drawdown_period": "",
        "monthly_avg_observation_count": 0.0,
        "monthly_max_consecutive_gains": 0.0,
        "monthly_max_consecutive_losses": 0.0,
        "monthly_max_recovery_periods": 0.0,
        "monthly_underwater_ratio": 0.0,
        "monthly_current_underwater_periods": 0.0,
    }


def test_compute_period_returns_includes_first_period_observation_against_prior_close() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-31", "2026-06-01"]),
            "net_value": [1.2, 1.1],
        }
    )

    monthly = compute_period_returns(equity, freq="M")
    june = monthly.set_index("period").loc["2026-06"]

    assert june["start_net_value"] == pytest.approx(1.2)
    assert june["end_net_value"] == pytest.approx(1.1)
    assert june["return"] == pytest.approx(1.1 / 1.2 - 1.0)
    assert june["max_drawdown"] == pytest.approx(1.1 / 1.2 - 1.0)
    assert june["observation_count"] == 1


def test_summarize_order_decisions_reports_rates_and_allocation_usage() -> None:
    decisions = pd.DataFrame(
        {
            "status": ["accepted", "rejected", "accepted", "rejected", "rejected", "rejected", "rejected"],
            "reason": ["", "no_fill", "", "no_capital", "actual_risk_too_high", "chase_too_far", "invalid_order"],
            "capital_fraction": [0.25, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
            "risk_fraction": [0.01, 0.0, 0.015, 0.0, 0.0, 0.0, 0.0],
            "margin_fraction": [0.25, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
            "actual_entry_price": [10.0, 0.0, 20.0, 30.0, 11.0, 12.0, 0.0],
            "actual_risk_pct": [0.03, 0.0, 0.04, 0.06, 0.1, 0.08, 0.0],
            "actual_chase_pct": [0.01, 0.0, 0.02, 0.03, 0.1, 0.05, 0.0],
            "actual_reward_to_risk": [2.0, 0.0, 1.5, 1.0, 0.9, 1.2, 0.0],
        }
    )

    stats = summarize_order_decisions(decisions)

    assert stats["order_count"] == 7.0
    assert stats["acceptance_rate"] == pytest.approx(2 / 7)
    assert stats["rejection_rate"] == pytest.approx(5 / 7)
    assert stats["rejected_no_fill_count"] == 1.0
    assert stats["rejected_no_capital_count"] == 1.0
    assert stats["rejected_actual_risk_too_high_count"] == 1.0
    assert stats["rejected_chase_too_far_count"] == 1.0
    assert stats["rejected_invalid_order_count"] == 1.0
    assert stats["rejected_duplicate_order_id_count"] == 0.0
    assert stats["avg_accepted_capital_fraction"] == pytest.approx(0.3)
    assert stats["max_accepted_capital_fraction"] == pytest.approx(0.35)
    assert stats["avg_accepted_risk_fraction"] == pytest.approx(0.0125)
    assert stats["max_accepted_risk_fraction"] == pytest.approx(0.015)
    assert stats["avg_accepted_margin_fraction"] == pytest.approx(0.475)
    assert stats["max_accepted_margin_fraction"] == pytest.approx(0.7)
    assert stats["avg_executed_actual_risk_pct"] == pytest.approx(0.062)
    assert stats["max_executed_actual_risk_pct"] == pytest.approx(0.1)
    assert stats["avg_executed_actual_chase_pct"] == pytest.approx(0.042)
    assert stats["max_executed_actual_chase_pct"] == pytest.approx(0.1)
    assert stats["avg_executed_actual_reward_to_risk"] == pytest.approx(1.32)
    assert stats["min_executed_actual_reward_to_risk"] == pytest.approx(0.9)


def test_summarize_order_decisions_reports_custom_rejection_reasons() -> None:
    decisions = pd.DataFrame(
        {
            "status": ["rejected", "rejected", "rejected", "accepted"],
            "reason": ["price_limit_blocked", "price_limit_blocked", "daily_open_filtered", ""],
        }
    )

    stats = summarize_order_decisions(decisions)

    assert stats["rejected_price_limit_blocked_count"] == 2.0
    assert stats["rejected_daily_open_filtered_count"] == 1.0


def test_compute_decision_reason_statistics_groups_by_strategy_status_and_reason() -> None:
    decisions = pd.DataFrame(
        {
            "strategy_name": ["trend_signal_bar", "trend_signal_bar", "range_signal_bar", "range_signal_bar"],
            "detector_name": ["trend", "trend", "range", "range"],
            "status": ["accepted", "rejected", "rejected", "rejected"],
            "reason": ["", "no_fill", "no_capital", "no_capital"],
            "actual_risk_pct": [0.03, 0.0, 0.04, 0.06],
            "actual_chase_pct": [0.01, 0.0, 0.02, 0.03],
            "actual_reward_to_risk": [2.0, 0.0, 1.5, 1.0],
        }
    )

    stats = compute_decision_reason_statistics(decisions)
    by_key = stats.set_index(["strategy_name", "detector_name", "status", "reason"])

    assert by_key.loc[("trend_signal_bar", "trend", "accepted", ""), "decision_count"] == 1
    assert by_key.loc[("trend_signal_bar", "trend", "accepted", ""), "decision_rate"] == pytest.approx(0.25)
    assert by_key.loc[("trend_signal_bar", "trend", "accepted", ""), "group_decision_count"] == 2
    assert by_key.loc[("trend_signal_bar", "trend", "accepted", ""), "group_decision_rate"] == pytest.approx(0.5)
    assert by_key.loc[("trend_signal_bar", "trend", "rejected", "no_fill"), "decision_count"] == 1
    assert by_key.loc[("trend_signal_bar", "trend", "rejected", "no_fill"), "group_decision_rate"] == pytest.approx(0.5)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "decision_count"] == 2
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "decision_rate"] == pytest.approx(0.5)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "group_decision_count"] == 2
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "group_decision_rate"] == pytest.approx(1.0)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "avg_actual_risk_pct"] == pytest.approx(0.05)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "max_actual_risk_pct"] == pytest.approx(0.06)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "avg_actual_chase_pct"] == pytest.approx(0.025)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "max_actual_chase_pct"] == pytest.approx(0.03)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "avg_actual_reward_to_risk"] == pytest.approx(1.25)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "min_actual_reward_to_risk"] == pytest.approx(1.0)


def test_compute_decision_reason_statistics_handles_missing_execution_metric_columns() -> None:
    decisions = pd.DataFrame(
        {
            "strategy_name": ["trend_signal_bar"],
            "detector_name": ["trend"],
            "status": ["rejected"],
            "reason": ["higher_timeframe_mismatch"],
        }
    )

    stats = compute_decision_reason_statistics(decisions)
    row = stats.iloc[0]

    assert row["decision_count"] == 1
    assert row["avg_actual_risk_pct"] == 0.0
    assert row["max_actual_risk_pct"] == 0.0
    assert row["avg_actual_chase_pct"] == 0.0
    assert row["max_actual_chase_pct"] == 0.0
    assert row["avg_actual_reward_to_risk"] == 0.0
    assert row["min_actual_reward_to_risk"] == 0.0


def test_summarize_strategy_filter_decisions_reports_higher_timeframe_reasons() -> None:
    decisions = pd.DataFrame(
        {
            "status": ["accepted", "rejected", "rejected", "rejected", "rejected"],
            "reason": [
                "",
                "higher_timeframe_mismatch",
                "higher_timeframe_no_context",
                "higher_timeframe_stale",
                "signal_bar_no_liquidity",
            ],
        }
    )

    stats = summarize_strategy_filter_decisions(decisions)

    assert stats["strategy_signal_count"] == 5.0
    assert stats["strategy_accepted_signal_count"] == 1.0
    assert stats["strategy_rejected_signal_count"] == 4.0
    assert stats["strategy_filter_acceptance_rate"] == pytest.approx(0.2)
    assert stats["strategy_rejected_higher_timeframe_mismatch_count"] == 1.0
    assert stats["strategy_rejected_higher_timeframe_no_context_count"] == 1.0
    assert stats["strategy_rejected_higher_timeframe_stale_count"] == 1.0
    assert stats["strategy_rejected_signal_bar_no_liquidity_count"] == 1.0


def test_summarize_strategy_filter_decisions_reports_custom_rejection_reasons() -> None:
    decisions = pd.DataFrame(
        {
            "status": ["rejected", "rejected", "accepted"],
            "reason": ["same_timeframe_middle", "same_timeframe_middle", ""],
        }
    )

    stats = summarize_strategy_filter_decisions(decisions)

    assert stats["strategy_rejected_same_timeframe_middle_count"] == 2.0
