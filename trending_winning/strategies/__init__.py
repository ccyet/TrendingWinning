"""Strategy adapters that consume one detector event stream."""

from trending_winning.strategies.base import ORDER_COLUMNS, Strategy, empty_orders
from trending_winning.strategies.multitimeframe import HigherTimeframeAlignmentStrategy, TimeframeAlignmentConfig
from trending_winning.strategies.runtime import StrategyBatchRunResult, StrategyRunResult, execute_strategies, execute_strategy
from trending_winning.strategies.signal_bar import SignalBarStopStrategy, SignalBarStopStrategyConfig
from trending_winning.strategies.suite import StrategySuiteConfig, create_default_strategy_suite, create_strategy_for_detector

__all__ = [
    "HigherTimeframeAlignmentStrategy",
    "ORDER_COLUMNS",
    "SignalBarStopStrategy",
    "SignalBarStopStrategyConfig",
    "Strategy",
    "StrategyBatchRunResult",
    "StrategyRunResult",
    "StrategySuiteConfig",
    "TimeframeAlignmentConfig",
    "create_default_strategy_suite",
    "create_strategy_for_detector",
    "execute_strategies",
    "execute_strategy",
    "empty_orders",
]
