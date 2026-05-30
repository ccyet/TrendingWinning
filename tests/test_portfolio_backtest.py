from __future__ import annotations

from inspect import getsource

import pandas as pd
import pytest

from trending_winning.backtest.engine import BacktestConfig
from trending_winning.backtest.portfolio import (
    PortfolioConfig,
    _build_portfolio_equity_curve_from_normalized,
    _candidate_trades_by_entry_time,
    run_portfolio_backtest,
    run_portfolio_order_backtest,
    run_portfolio_order_backtest_from_normalized,
)
from trending_winning.data.schema import normalize_bars
from trending_winning.strategies.base import ORDER_COLUMNS
from trending_winning.strategies.runtime import StrategyRunResult


class FixedOrderStrategy:
    def __init__(self, name: str, orders: list[dict[str, object]]) -> None:
        self.name = name
        self._orders = orders

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        if not self._orders:
            return pd.DataFrame(columns=ORDER_COLUMNS)
        return pd.DataFrame(self._orders, columns=ORDER_COLUMNS)


class ExplicitPlanStrategy:
    name = "explicit_portfolio"

    def __init__(self, orders: list[dict[str, object]], filter_decisions: pd.DataFrame) -> None:
        self._orders = orders
        self._filter_decisions = filter_decisions

    def generate_order_plan(self, bars: pd.DataFrame, *, timeframe: str = "") -> StrategyRunResult:
        return StrategyRunResult(
            orders=pd.DataFrame(self._orders, columns=ORDER_COLUMNS),
            filter_decisions=self._filter_decisions,
        )

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        raise AssertionError("组合回测应优先消费显式策略运行结果。")


def _portfolio_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "000001.SZ": [10.0, 10.8, 11.4, 11.8],
        "000002.SZ": [20.0, 20.8, 21.4, 21.8],
    }
    for symbol, closes in values.items():
        for index, close in enumerate(closes):
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                    "stock_code": symbol,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.4,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _drawdown_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = [10.0, 9.8, 11.2]
    highs = [10.2, 10.5, 11.3]
    lows = [9.8, 9.5, 10.9]
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.1,
                "high": highs[index],
                "low": lows[index],
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _compound_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = [10.0, 10.6, 11.1, 11.6, 12.1]
    highs = [10.2, 10.8, 11.2, 11.8, 12.2]
    lows = [9.8, 10.3, 10.9, 11.3, 11.9]
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.1,
                "high": highs[index],
                "low": lows[index],
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _short_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = [20.0, 19.2, 18.8]
    highs = [20.2, 19.4, 19.0]
    lows = [19.8, 19.0, 18.6]
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000002.SZ",
                "open": close + 0.1,
                "high": highs[index],
                "low": lows[index],
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _staggered_entry_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    payload = {
        "000001.SZ": [
            ("2026-05-25 09:30:00", 10.0, 10.2, 9.8, 10.0),
            ("2026-05-27 10:00:00", 10.4, 11.1, 10.3, 11.0),
        ],
        "000002.SZ": [
            ("2026-05-26 09:30:00", 20.0, 20.2, 19.8, 20.0),
            ("2026-05-26 10:00:00", 20.4, 21.1, 20.3, 21.0),
        ],
    }
    for symbol, bars in payload.items():
        for date, open_price, high, low, close in bars:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "stock_code": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _gap_fill_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]
            ),
            "stock_code": ["000001.SZ"] * 3,
            "open": [10.0, 11.0, 11.1],
            "high": [10.2, 11.2, 11.3],
            "low": [9.8, 10.9, 11.0],
            "close": [10.0, 11.0, 11.2],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [10000.0, 12100.0, 13440.0],
        }
    )


def _order(
    *,
    strategy_name: str,
    symbol: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    side: str = "long",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "order_id": f"{strategy_name}:{symbol}",
        "strategy_name": strategy_name,
        "detector_name": strategy_name.replace("_strategy", ""),
        "event_id": f"event:{strategy_name}:{symbol}",
        "stock_code": symbol,
        "timeframe": "30m",
        "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
        "signal_bar_index": 0,
        "side": side,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "max_holding_bars": 3,
        "metadata": metadata or {},
    }


def _portfolio_filter_decisions(reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": "portfolio-filter",
                "event_id": "event:portfolio-filter",
                "strategy_name": "explicit_portfolio",
                "base_strategy_name": "explicit_portfolio",
                "detector_name": "trend",
                "event_type": "custom_event",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                "signal_bar_index": 0,
                "side": "long",
                "status": "rejected",
                "reason": reason,
                "filter_name": "pure_filter",
                "context_timeframe": "",
                "context_date": pd.NaT,
                "context_state": "",
            }
        ]
    )


def test_portfolio_backtest_allocates_capital_across_independent_strategies() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [_order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.8, target_price=11.6)],
        ),
        FixedOrderStrategy(
            "range_strategy",
            [_order(strategy_name="range_strategy", symbol="000002.SZ", entry_price=20.4, stop_price=19.8, target_price=21.6)],
        ),
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=2),
    )

    assert result.stats["trade_count"] == 2.0
    assert result.trades["capital_fraction"].tolist() == [0.5, 0.5]
    assert {"order_id", "event_id", "signal_date", "signal_bar_index", "planned_entry_price", "metadata"}.issubset(
        result.trades.columns
    )
    assert result.trades["order_id"].tolist() == ["trend_strategy:000001.SZ", "range_strategy:000002.SZ"]
    assert result.trades["event_id"].tolist() == ["event:trend_strategy:000001.SZ", "event:range_strategy:000002.SZ"]
    assert (result.trades["return_pct"] == result.trades["raw_return_pct"] * result.trades["capital_fraction"]).all()


def test_portfolio_backtest_consumes_explicit_strategy_run_results() -> None:
    strategy = ExplicitPlanStrategy(
        [
            _order(
                strategy_name="explicit_portfolio",
                symbol="000001.SZ",
                entry_price=10.4,
                stop_price=9.8,
                target_price=11.6,
            )
        ],
        _portfolio_filter_decisions("custom_portfolio_filter"),
    )

    result = run_portfolio_backtest(
        _portfolio_bars(),
        [strategy],
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades["order_id"].tolist() == ["explicit_portfolio:000001.SZ"]
    assert result.strategy_filter_decisions["reason"].tolist() == ["custom_portfolio_filter"]
    assert result.stats["strategy_rejected_custom_portfolio_filter_count"] == 1.0


def test_portfolio_backtest_marks_equity_to_market_on_each_bar() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [_order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.8, target_price=11.6)],
        ),
        FixedOrderStrategy(
            "range_strategy",
            [_order(strategy_name="range_strategy", symbol="000002.SZ", entry_price=20.4, stop_price=19.8, target_price=21.6)],
        ),
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=2),
    )

    assert {"date", "net_value", "gross_exposure", "open_positions"}.issubset(result.equity_curve.columns)
    assert result.equity_curve["date"].tolist() == sorted(result.equity_curve["date"].tolist())
    assert result.equity_curve.loc[0, "net_value"] == 1.0
    assert result.equity_curve.loc[1, "gross_exposure"] == 1.0
    assert result.equity_curve.loc[1, "open_positions"] == 2
    assert result.equity_curve.iloc[-1]["net_value"] == 1.0 + result.trades["return_pct"].sum() / 100.0


def test_portfolio_equity_curve_builds_entry_queue_without_record_conversion() -> None:
    source = getsource(_build_portfolio_equity_curve_from_normalized)

    assert ".to_dict(" not in source


def test_portfolio_statistics_use_time_marked_equity_drawdown() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [_order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.0, target_price=11.0)],
        )
    ]

    result = run_portfolio_backtest(
        _drawdown_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    equity_drawdown = result.equity_curve["net_value"] / result.equity_curve["net_value"].cummax() - 1.0
    assert result.stats["total_return"] == pytest.approx(result.equity_curve.iloc[-1]["net_value"] - 1.0)
    assert result.stats["max_drawdown"] == pytest.approx(equity_drawdown.min())
    assert result.stats["max_drawdown"] < 0.0
    assert "annualized_return" in result.stats
    assert "annualized_volatility" in result.stats
    assert "calmar_ratio" in result.stats
    assert result.stats["max_gross_exposure"] > 0.0


def test_portfolio_equity_reinvests_realized_cash_for_later_trades() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [
                _order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.0, target_price=11.0),
                {
                    **_order(
                        strategy_name="trend_strategy",
                        symbol="000001.SZ",
                        entry_price=11.4,
                        stop_price=10.5,
                        target_price=12.0,
                    ),
                    "order_id": "trend_strategy:000001.SZ:second",
                    "signal_date": pd.Timestamp("2026-05-25 10:30:00"),
                    "signal_bar_index": 2,
                },
            ],
        )
    ]

    result = run_portfolio_backtest(
        _compound_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    expected = (1.0 + result.trades.loc[0, "raw_return_pct"] / 100.0) * (
        1.0 + result.trades.loc[1, "raw_return_pct"] / 100.0
    )
    assert result.stats["trade_count"] == 2.0
    assert result.equity_curve.iloc[-1]["net_value"] == pytest.approx(expected)
    assert result.equity_curve.iloc[-1]["net_value"] > 1.0 + result.trades["return_pct"].sum() / 100.0


def test_portfolio_equity_models_short_cash_and_negative_position_value() -> None:
    strategies = [
        FixedOrderStrategy(
            "channel_strategy",
            [
                _order(
                    strategy_name="channel_strategy",
                    symbol="000002.SZ",
                    entry_price=19.5,
                    stop_price=20.5,
                    target_price=18.7,
                    side="short",
                )
            ],
        )
    ]

    result = run_portfolio_backtest(
        _short_bars(),
        strategies,
        BacktestConfig(max_holding_bars=2),
        PortfolioConfig(max_open_positions=1),
    )

    open_row = result.equity_curve.loc[result.equity_curve["open_positions"] == 1].iloc[0]
    assert open_row["cash"] > 1.0
    assert open_row["position_value"] < 0.0
    assert open_row["net_value"] > 1.0
    assert result.equity_curve.iloc[-1]["open_positions"] == 0
    assert result.equity_curve.iloc[-1]["cash"] == pytest.approx(result.equity_curve.iloc[-1]["net_value"])


def test_portfolio_backtest_limits_short_notional_by_margin_rate() -> None:
    strategies = [
        FixedOrderStrategy(
            "channel_strategy",
            [
                _order(
                    strategy_name="channel_strategy",
                    symbol="000002.SZ",
                    entry_price=19.5,
                    stop_price=20.5,
                    target_price=18.7,
                    side="short",
                )
            ],
        )
    ]

    result = run_portfolio_backtest(
        _short_bars(),
        strategies,
        BacktestConfig(max_holding_bars=2),
        PortfolioConfig(max_open_positions=1, capital_per_trade=1.0, short_margin_rate=2.0),
    )

    assert result.trades.loc[0, "capital_fraction"] == pytest.approx(0.5)
    assert result.trades.loc[0, "margin_fraction"] == pytest.approx(1.0)
    assert result.trades.loc[0, "return_pct"] == pytest.approx(result.trades.loc[0, "raw_return_pct"] * 0.5)


def test_portfolio_backtest_normalizes_short_side_before_margin_allocation() -> None:
    strategies = [
        FixedOrderStrategy(
            "channel_strategy",
            [
                _order(
                    strategy_name="channel_strategy",
                    symbol="000002.SZ",
                    entry_price=19.5,
                    stop_price=20.5,
                    target_price=18.7,
                    side=" SHORT ",
                )
            ],
        )
    ]

    result = run_portfolio_backtest(
        _short_bars(),
        strategies,
        BacktestConfig(max_holding_bars=2),
        PortfolioConfig(max_open_positions=1, capital_per_trade=1.0, short_margin_rate=2.0),
    )

    assert result.trades["side"].tolist() == ["short"]
    assert result.order_decisions.loc[0, "side"] == "short"
    assert result.trades.loc[0, "capital_fraction"] == pytest.approx(0.5)
    assert result.trades.loc[0, "margin_fraction"] == pytest.approx(1.0)


def test_portfolio_backtest_respects_strategy_priority_when_capacity_is_full() -> None:
    strategies = [
        FixedOrderStrategy(
            "range_strategy",
            [_order(strategy_name="range_strategy", symbol="000002.SZ", entry_price=20.4, stop_price=19.8, target_price=21.6)],
        ),
        FixedOrderStrategy(
            "trend_strategy",
            [_order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.8, target_price=11.6)],
        ),
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1, strategy_priority={"trend_strategy": 0, "range_strategy": 1}),
    )

    assert result.stats["trade_count"] == 1.0
    assert result.trades["strategy_name"].tolist() == ["trend_strategy"]
    assert result.trades["capital_fraction"].tolist() == [1.0]


def test_portfolio_backtest_records_order_decisions_for_capacity_rejections() -> None:
    strategies = [
        FixedOrderStrategy(
            "range_strategy",
            [_order(strategy_name="range_strategy", symbol="000002.SZ", entry_price=20.4, stop_price=19.8, target_price=21.6)],
        ),
        FixedOrderStrategy(
            "trend_strategy",
            [_order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.8, target_price=11.6)],
        ),
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1, strategy_priority={"trend_strategy": 0, "range_strategy": 1}),
    )

    decisions = result.order_decisions.set_index("order_id")
    assert decisions.loc["trend_strategy:000001.SZ", "status"] == "accepted"
    assert decisions.loc["trend_strategy:000001.SZ", "reason"] == ""
    assert decisions.loc["range_strategy:000002.SZ", "status"] == "rejected"
    assert decisions.loc["range_strategy:000002.SZ", "reason"] == "max_open_positions"
    assert decisions.loc["range_strategy:000002.SZ", "entry_date"] == pd.Timestamp("2026-05-25 10:00:00")
    assert decisions.loc["range_strategy:000002.SZ", "actual_entry_price"] == pytest.approx(20.6)
    assert decisions.loc["range_strategy:000002.SZ", "actual_risk_pct"] == pytest.approx((20.6 - 19.8) / 20.6)
    assert decisions.loc["range_strategy:000002.SZ", "actual_reward_to_risk"] == pytest.approx((21.6 - 20.6) / (20.6 - 19.8))
    assert result.stats["accepted_order_count"] == 1.0
    assert result.stats["rejected_order_count"] == 1.0
    assert result.stats["rejected_max_open_positions_count"] == 1.0


def test_portfolio_backtest_records_execution_metrics_for_preallocation_rejections() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [
                {
                    **_order(
                        strategy_name="trend_strategy",
                        symbol="000001.SZ",
                        entry_price=10.2,
                        stop_price=9.9,
                        target_price=12.0,
                    ),
                    "signal_price": 10.0,
                    "max_actual_risk_pct": 0.05,
                    "max_chase_pct": 0.2,
                }
            ],
        )
    ]

    result = run_portfolio_backtest(
        _gap_fill_bars(),
        strategies,
        BacktestConfig(max_holding_bars=1),
        PortfolioConfig(max_open_positions=1),
    )

    decision = result.order_decisions.set_index("order_id").loc["trend_strategy:000001.SZ"]
    assert result.trades.empty
    assert decision["status"] == "rejected"
    assert decision["reason"] == "actual_risk_too_high"
    assert decision["actual_entry_price"] == pytest.approx(11.0)
    assert decision["actual_risk_pct"] == pytest.approx((11.0 - 9.9) / 11.0)
    assert decision["actual_chase_pct"] == pytest.approx(0.1)


def test_portfolio_backtest_records_order_decisions_for_unfilled_orders() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [_order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=99.0, stop_price=98.0, target_price=101.0)],
        )
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    decisions = result.order_decisions.set_index("order_id")
    assert result.trades.empty
    assert decisions.loc["trend_strategy:000001.SZ", "status"] == "rejected"
    assert decisions.loc["trend_strategy:000001.SZ", "reason"] == "no_fill"
    assert pd.isna(decisions.loc["trend_strategy:000001.SZ", "entry_date"])
    assert result.stats["accepted_order_count"] == 0.0
    assert result.stats["rejected_order_count"] == 1.0
    assert result.stats["rejected_no_fill_count"] == 1.0


def test_portfolio_order_decisions_are_sorted_by_signal_time_after_mixed_acceptance_and_rejection() -> None:
    early_order = {
        **_order(
            strategy_name="trend_strategy",
            symbol="000001.SZ",
            entry_price=10.4,
            stop_price=9.8,
            target_price=11.6,
        ),
        "order_id": "early-accepted",
        "event_id": "event-early-accepted",
        "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
        "signal_bar_index": 0,
    }
    late_unfilled = {
        **_order(
            strategy_name="range_strategy",
            symbol="000002.SZ",
            entry_price=99.0,
            stop_price=98.0,
            target_price=101.0,
        ),
        "order_id": "late-rejected",
        "event_id": "event-late-rejected",
        "signal_date": pd.Timestamp("2026-05-25 10:00:00"),
        "signal_bar_index": 1,
    }
    orders = pd.DataFrame([late_unfilled, early_order], columns=ORDER_COLUMNS)

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=2),
    )

    assert result.order_decisions["order_id"].tolist() == ["early-accepted", "late-rejected"]
    assert result.order_decisions["status"].tolist() == ["accepted", "rejected"]
    assert result.order_decisions["reason"].tolist() == ["", "no_fill"]


def test_portfolio_order_backtest_from_normalized_reuses_prepared_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    from trending_winning.backtest import portfolio as portfolio_module

    normalized = normalize_bars(_portfolio_bars())
    orders = pd.DataFrame(
        [
            _order(
                strategy_name="trend_strategy",
                symbol="000001.SZ",
                entry_price=10.4,
                stop_price=9.8,
                target_price=11.6,
            )
        ],
        columns=ORDER_COLUMNS,
    )

    def fail_normalize(_: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("组合订单回测 normalized 入口不应重复 normalize。")

    monkeypatch.setattr(portfolio_module, "normalize_bars", fail_normalize)

    result = run_portfolio_order_backtest_from_normalized(
        normalized,
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades["order_id"].tolist() == ["trend_strategy:000001.SZ"]
    assert result.stats["accepted_order_count"] == 1.0


def test_portfolio_order_backtest_rejects_duplicate_order_ids_before_allocation() -> None:
    first = {
        **_order(
            strategy_name="trend_strategy",
            symbol="000001.SZ",
            entry_price=10.4,
            stop_price=9.8,
            target_price=11.6,
        ),
        "order_id": "shared-order",
        "event_id": "event-first",
    }
    duplicate = {
        **_order(
            strategy_name="range_strategy",
            symbol="000002.SZ",
            entry_price=20.4,
            stop_price=19.8,
            target_price=21.6,
        ),
        "order_id": "shared-order",
        "event_id": "event-duplicate",
    }
    orders = pd.DataFrame([first, duplicate], columns=ORDER_COLUMNS)

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=2),
    )

    decisions = result.order_decisions.set_index("event_id")
    assert result.trades["event_id"].tolist() == ["event-first"]
    assert decisions.loc["event-first", "status"] == "accepted"
    assert decisions.loc["event-duplicate", "status"] == "rejected"
    assert decisions.loc["event-duplicate", "reason"] == "duplicate_order_id"
    assert result.stats["rejected_duplicate_order_id_count"] == 1.0


def test_portfolio_order_backtest_records_no_bars_when_market_data_is_empty() -> None:
    orders = pd.DataFrame(
        [
            _order(
                strategy_name="trend_strategy",
                symbol="000001.SZ",
                entry_price=10.4,
                stop_price=9.8,
                target_price=11.6,
            )
        ],
        columns=ORDER_COLUMNS,
    )
    empty_bars = pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"])

    result = run_portfolio_order_backtest(
        empty_bars,
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["order_id"] == "trend_strategy:000001.SZ"
    assert decision["status"] == "rejected"
    assert decision["reason"] == "no_bars"
    assert result.stats["rejected_no_bars_count"] == 1.0


def test_portfolio_order_backtest_prefers_invalid_order_over_no_bars_when_market_data_is_empty() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=11.6,
                ),
                "event_id": "",
            }
        ],
        columns=ORDER_COLUMNS,
    )
    empty_bars = pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"])

    result = run_portfolio_order_backtest(
        empty_bars,
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0
    assert result.stats["rejected_no_bars_count"] == 0.0


def test_portfolio_order_backtest_prefers_target_direction_rejection_over_no_bars_when_market_data_is_empty() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=10.0,
                ),
                "order_id": "bad-target-empty-market",
            }
        ],
        columns=ORDER_COLUMNS,
    )
    empty_bars = pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"])

    result = run_portfolio_order_backtest(
        empty_bars,
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "target_not_favorable"
    assert result.stats["rejected_target_not_favorable_count"] == 1.0
    assert result.stats["rejected_no_bars_count"] == 0.0


@pytest.mark.parametrize("order_id", ["", "   ", None])
def test_portfolio_order_backtest_rejects_orders_without_traceable_order_id(order_id: object) -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=11.6,
                ),
                "order_id": order_id,
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


@pytest.mark.parametrize("event_id", ["", "   ", None])
def test_portfolio_order_backtest_rejects_orders_without_traceable_event_id(event_id: object) -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=11.6,
                ),
                "event_id": event_id,
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_portfolio_order_backtest_reports_missing_required_order_columns_clearly() -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "missing-side",
                "event_id": "event-missing-side",
                "strategy_name": "trend_strategy",
                "stock_code": "000001.SZ",
                "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                "signal_bar_index": 0,
                "entry_price": 10.4,
                "stop_price": 9.8,
                "target_price": 11.6,
            }
        ]
    )

    with pytest.raises(ValueError, match="订单缺少必要字段：side"):
        run_portfolio_order_backtest(
            _portfolio_bars(),
            orders,
            BacktestConfig(max_holding_bars=3),
            PortfolioConfig(max_open_positions=1),
        )


def test_portfolio_order_backtest_records_invalid_order_instead_of_dropping_it() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=11.6,
                ),
                "order_id": "bad-date",
                "signal_date": "not-a-date",
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["order_id"] == "bad-date"
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_portfolio_order_backtest_records_non_numeric_signal_bar_index_as_invalid_order() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=11.6,
                ),
                "order_id": "bad-index-text",
                "signal_bar_index": "bad-index",
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["order_id"] == "bad-index-text"
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert decision["signal_bar_index"] == -1
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_portfolio_order_backtest_records_non_numeric_price_fields_as_invalid_order() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=11.6,
                ),
                "order_id": "bad-price",
                "entry_price": "bad-price",
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["order_id"] == "bad-price"
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert decision["planned_entry_price"] == 0.0
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_portfolio_order_backtest_rejects_zero_risk_orders_before_allocation() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=10.4,
                    target_price=11.6,
                ),
                "order_id": "zero-risk",
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(max_open_positions=1, risk_per_trade=0.01),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["order_id"] == "zero-risk"
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_portfolio_order_backtest_uses_config_max_holding_when_order_value_is_missing() -> None:
    order = _order(
        strategy_name="trend_strategy",
        symbol="000001.SZ",
        entry_price=10.4,
        stop_price=9.8,
        target_price=12.0,
    )
    order.pop("max_holding_bars")
    orders = pd.DataFrame([order], columns=ORDER_COLUMNS)

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=2),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.order_decisions.iloc[0]["status"] == "accepted"
    assert result.trades.loc[0, "order_id"] == "trend_strategy:000001.SZ"
    assert result.trades.loc[0, "holding_bars"] <= 2


def test_portfolio_order_backtest_rejects_non_numeric_max_holding_bars() -> None:
    orders = pd.DataFrame(
        [
            {
                **_order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.8,
                    target_price=12.0,
                ),
                "order_id": "bad-max-holding",
                "max_holding_bars": "bad",
            }
        ],
        columns=ORDER_COLUMNS,
    )

    result = run_portfolio_order_backtest(
        _portfolio_bars(),
        orders,
        BacktestConfig(max_holding_bars=2),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["order_id"] == "bad-max-holding"
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_portfolio_backtest_allocates_capacity_by_actual_entry_time_not_signal_time() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [
                _order(strategy_name="trend_strategy", symbol="000001.SZ", entry_price=10.4, stop_price=9.8, target_price=11.0)
            ],
        ),
        FixedOrderStrategy(
            "range_strategy",
            [
                {
                    **_order(
                        strategy_name="range_strategy",
                        symbol="000002.SZ",
                        entry_price=20.4,
                        stop_price=19.8,
                        target_price=21.0,
                    ),
                    "signal_date": pd.Timestamp("2026-05-26 09:30:00"),
                    "signal_bar_index": 0,
                }
            ],
        ),
    ]

    result = run_portfolio_backtest(
        _staggered_entry_bars(),
        strategies,
        BacktestConfig(max_holding_bars=1),
        PortfolioConfig(max_open_positions=1),
    )

    assert result.trades["order_id"].tolist() == ["range_strategy:000002.SZ", "trend_strategy:000001.SZ"]
    assert result.trades["entry_date"].tolist() == sorted(result.trades["entry_date"].tolist())
    assert result.stats["trade_count"] == 2.0


def test_portfolio_risk_budget_uses_actual_fill_price_after_gap_entry() -> None:
    result = run_portfolio_backtest(
        _gap_fill_bars(),
        [
            FixedOrderStrategy(
                "trend_strategy",
                [
                    _order(
                        strategy_name="trend_strategy",
                        symbol="000001.SZ",
                        entry_price=10.4,
                        stop_price=9.8,
                        target_price=12.0,
                    )
                ],
            )
        ],
        BacktestConfig(max_holding_bars=1),
        PortfolioConfig(max_open_positions=1, risk_per_trade=0.01),
    )

    trade = result.trades.iloc[0]
    expected_risk_fraction = (11.0 - 9.8) / 11.0
    assert trade["entry_price"] == pytest.approx(11.0)
    assert trade["risk_fraction"] == pytest.approx(expected_risk_fraction)
    assert trade["capital_fraction"] == pytest.approx(0.01 / expected_risk_fraction)


def test_portfolio_backtest_sizes_positions_by_trade_risk_and_sector_limit() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [
                _order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.88,
                    target_price=11.6,
                    metadata={"sector": "新能源"},
                ),
                _order(
                    strategy_name="trend_strategy",
                    symbol="000002.SZ",
                    entry_price=20.4,
                    stop_price=19.38,
                    target_price=21.6,
                    metadata={"sector": "新能源"},
                ),
            ],
        )
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(
            max_open_positions=3,
            risk_per_trade=0.01,
            max_capital_per_trade=0.5,
            sector_capital_limit={"新能源": 0.25},
        ),
    )

    assert result.trades["sector"].tolist() == ["新能源", "新能源"]
    expected_risk = (
        (result.trades["entry_price"] - result.trades["stop_price"]).abs() / result.trades["entry_price"]
    ).round(4)
    assert result.trades["risk_fraction"].round(4).tolist() == expected_risk.tolist()
    assert result.trades["capital_fraction"].round(4).tolist() == [0.1472, 0.1028]


def test_portfolio_backtest_can_apply_sector_limits_from_symbol_map() -> None:
    strategies = [
        FixedOrderStrategy(
            "trend_strategy",
            [
                _order(
                    strategy_name="trend_strategy",
                    symbol="000001.SZ",
                    entry_price=10.4,
                    stop_price=9.88,
                    target_price=11.6,
                ),
                _order(
                    strategy_name="trend_strategy",
                    symbol="000002.SZ",
                    entry_price=20.4,
                    stop_price=19.38,
                    target_price=21.6,
                ),
            ],
        )
    ]

    result = run_portfolio_backtest(
        _portfolio_bars(),
        strategies,
        BacktestConfig(max_holding_bars=3),
        PortfolioConfig(
            max_open_positions=3,
            risk_per_trade=0.01,
            max_capital_per_trade=0.5,
            sector_capital_limit={"新能源": 0.25},
            symbol_sector_map={"000001.SZ": "新能源", "000002.SZ": "新能源"},
        ),
    )

    assert result.trades["sector"].tolist() == ["新能源", "新能源"]
    assert result.trades["capital_fraction"].round(4).tolist() == [0.1472, 0.1028]


def test_portfolio_candidate_generation_uses_record_iteration_not_dataframe_iterrows() -> None:
    source = getsource(_candidate_trades_by_entry_time)

    assert ".iterrows(" not in source
