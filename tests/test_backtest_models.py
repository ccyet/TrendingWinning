from __future__ import annotations

import pandas as pd

from trending_winning.backtest.models import (
    BacktestConfig,
    BacktestResult,
    ORDER_DECISION_COLUMNS,
    ORDER_REQUIRED_COLUMNS,
    TRADE_COLUMNS,
)


def test_backtest_models_are_importable_without_engine() -> None:
    config = BacktestConfig(max_holding_bars=7, trailing_take_profit_ma_period=20)
    result = BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={})

    assert config.max_holding_bars == 7
    assert config.trailing_take_profit_ma_period == 20
    assert result.order_decisions.columns.tolist() == ORDER_DECISION_COLUMNS
    assert "strategy_name" in result.strategy_filter_decisions.columns
    assert "return_pct" in TRADE_COLUMNS
    assert ORDER_REQUIRED_COLUMNS.issubset(set(TRADE_COLUMNS))


def test_engine_reexports_backtest_models_for_compatibility() -> None:
    from trending_winning.backtest import engine

    assert engine.BacktestConfig is BacktestConfig
    assert engine.BacktestResult is BacktestResult
    assert engine.ORDER_DECISION_COLUMNS is ORDER_DECISION_COLUMNS
    assert engine.ORDER_REQUIRED_COLUMNS is ORDER_REQUIRED_COLUMNS
    assert engine.TRADE_COLUMNS is TRADE_COLUMNS
