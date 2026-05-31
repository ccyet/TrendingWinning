from __future__ import annotations

from trending_winning.backtest.models import TRADE_COLUMNS
from trending_winning.backtest.portfolio_models import (
    PORTFOLIO_COLUMNS,
    PortfolioCandidateSet,
    PortfolioConfig,
)


def test_portfolio_models_are_importable_without_portfolio_runner() -> None:
    config = PortfolioConfig(max_open_positions=3, strategy_priority={"trend": 0})
    candidates = PortfolioCandidateSet(candidates=({"order_id": "one"},), rejections=({"reason": "no_fill"},))

    assert config.max_open_positions == 3
    assert config.strategy_priority == {"trend": 0}
    assert candidates.candidates[0]["order_id"] == "one"
    assert candidates.rejections[0]["reason"] == "no_fill"
    assert PORTFOLIO_COLUMNS[: len(TRADE_COLUMNS)] == TRADE_COLUMNS
    assert {"capital_fraction", "margin_fraction", "portfolio_priority"}.issubset(PORTFOLIO_COLUMNS)


def test_portfolio_runner_reexports_models_for_compatibility() -> None:
    from trending_winning.backtest import portfolio

    assert portfolio.PortfolioConfig is PortfolioConfig
    assert portfolio.PortfolioCandidateSet is PortfolioCandidateSet
    assert portfolio.PORTFOLIO_COLUMNS is PORTFOLIO_COLUMNS
