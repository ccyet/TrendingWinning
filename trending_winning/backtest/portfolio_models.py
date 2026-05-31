from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from trending_winning.backtest.models import TRADE_COLUMNS


PORTFOLIO_COLUMNS = TRADE_COLUMNS + [
    "raw_return_pct",
    "capital_fraction",
    "margin_fraction",
    "risk_fraction",
    "sector",
    "portfolio_priority",
]


@dataclass(frozen=True)
class PortfolioConfig:
    """组合回测参数；capital_fraction 是名义仓位，margin_fraction 是组合资金占用。"""

    max_open_positions: int = 5
    capital_per_trade: float | None = None
    risk_per_trade: float | None = None
    max_capital_per_trade: float = 1.0
    short_margin_rate: float = 1.0
    reserve_cash: float = 0.0
    allow_same_symbol_overlap: bool = False
    strategy_priority: Mapping[str, int] = field(default_factory=dict)
    strategy_capital_limit: Mapping[str, float] = field(default_factory=dict)
    sector_capital_limit: Mapping[str, float] = field(default_factory=dict)
    symbol_sector_map: Mapping[str, str] = field(default_factory=dict)
    sector_metadata_key: str = "sector"
    default_sector: str = "UNKNOWN"


@dataclass(frozen=True)
class PortfolioCandidateSet:
    """已完成单笔撮合的候选成交；组合参数遍历可复用，避免重复模拟成交路径。"""

    candidates: tuple[dict[str, object], ...] = field(default_factory=tuple)
    rejections: tuple[dict[str, object], ...] = field(default_factory=tuple)
