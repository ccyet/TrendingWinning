from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.allocation import (
    PortfolioAllocationConfig,
    next_capital_fraction,
    order_margin_fraction,
)


def test_next_capital_fraction_respects_risk_margin_strategy_and_sector_limits() -> None:
    open_positions = [
        {
            "strategy_name": "trend",
            "sector": "新能源",
            "margin_fraction": 0.18,
        },
        {
            "strategy_name": "range",
            "sector": "银行",
            "margin_fraction": 0.10,
        },
    ]
    order = pd.Series({"strategy_name": "trend", "side": "short"})
    config = PortfolioAllocationConfig(
        max_open_positions=3,
        risk_per_trade=0.02,
        max_capital_per_trade=0.60,
        short_margin_rate=2.0,
        reserve_cash=0.10,
        strategy_capital_limit={"trend": 0.50},
        sector_capital_limit={"新能源": 0.40},
    )

    capital_fraction = next_capital_fraction(
        open_positions,
        order,
        sector="新能源",
        risk_fraction=0.05,
        config=config,
    )

    assert capital_fraction == pytest.approx(0.11)
    assert order_margin_fraction(order, capital_fraction, config) == pytest.approx(0.22)
