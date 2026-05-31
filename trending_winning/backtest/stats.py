from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from trending_winning.backtest.confidence import sample_confidence_statistics
from trending_winning.backtest.decision_stats import (
    compute_decision_reason_statistics as compute_decision_reason_statistics,
    summarize_order_decisions as summarize_order_decisions,
    summarize_strategy_filter_decisions as summarize_strategy_filter_decisions,
)
from trending_winning.backtest.drawdown import (
    empty_drawdown_statistics,
    equity_drawdown_statistics,
    max_drawdown_duration,
)
from trending_winning.backtest.equity_exposure import equity_exposure_statistics
from trending_winning.backtest.exposure import trade_exposure_statistics
from trending_winning.backtest.equity_metrics import equity_return_statistics
from trending_winning.backtest.exit_stats import (
    EXIT_REASONS as EXIT_REASONS,
    EXIT_REASON_STAT_KEYS as EXIT_REASON_STAT_KEYS,
    summarize_exit_reasons as summarize_exit_reasons,
)
from trending_winning.backtest.periods import (
    PERIOD_STAT_KEYS as PERIOD_STAT_KEYS,
    compute_period_return_statistics as compute_period_return_statistics,
    compute_period_returns as compute_period_returns,
)
from trending_winning.backtest.returns import (
    return_series_statistics,
)
from trending_winning.backtest.risk_metrics import trade_risk_quality_statistics


STAT_KEYS = [
    "trade_count",
    "win_rate",
    "win_rate_ci_lower",
    "win_rate_ci_upper",
    "total_return",
    "avg_return",
    "avg_return_standard_error",
    "avg_return_ci_lower",
    "avg_return_ci_upper",
    "positive_expectancy_probability",
    "max_drawdown",
    "profit_factor",
    "expectancy",
    "avg_win",
    "avg_loss",
    "payoff_ratio",
    "exposure_bars",
    "gross_profit",
    "gross_loss",
    "return_std",
    "sharpe_per_trade",
    "sortino_per_trade",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "avg_holding_bars",
    "best_trade",
    "worst_trade",
    "return_p05",
    "return_p25",
    "return_p50",
    "return_p75",
    "return_p95",
    "cvar_95",
    "max_drawdown_duration",
    "recovery_factor",
    "avg_r_multiple",
    "median_r_multiple",
    "best_r_multiple",
    "worst_r_multiple",
    "r_profit_factor",
    "system_quality_number",
    "avg_mae_pct",
    "avg_mfe_pct",
    "avg_mae_r",
    "avg_mfe_r",
    "return_contribution",
    "return_per_exposure_bar",
    "capital_turnover",
    "avg_capital_fraction",
    "max_capital_fraction",
    "margin_turnover",
    "avg_margin_fraction",
    "max_margin_fraction",
    "capital_exposure_bars",
    "margin_exposure_bars",
    "avg_capital_exposure_per_trade",
    "avg_margin_exposure_per_trade",
    "return_per_capital_exposure_bar",
    "return_per_margin_exposure_bar",
    "capital_weighted_raw_return",
]


EQUITY_STAT_KEYS = [
    "total_return",
    "max_drawdown",
    "max_drawdown_duration",
    "max_drawdown_start_at",
    "max_drawdown_trough_at",
    "max_drawdown_recovery_at",
    "current_drawdown",
    "current_underwater_bars",
    "equity_return_std",
    "equity_sharpe",
    "equity_sortino",
    "annualized_return",
    "annualized_volatility",
    "annualized_sharpe",
    "annualized_sortino",
    "calmar_ratio",
    "avg_drawdown",
    "ulcer_index",
    "time_under_water_ratio",
    "avg_gross_exposure",
    "max_gross_exposure",
    "avg_margin_exposure",
    "max_margin_exposure",
    "exposure_bar_ratio",
    "avg_open_positions",
    "max_open_positions",
    "avg_cash_ratio",
    "min_cash_ratio",
    "max_cash_ratio",
    "avg_net_exposure",
    "min_net_exposure",
    "max_net_exposure",
]

def build_equity_curve(trades: pd.DataFrame, initial_equity: float = 1.0) -> pd.DataFrame:
    """把逐笔收益转成净值曲线；有成交日期时同步保留时间轴。"""
    if trades.empty:
        return pd.DataFrame({"trade_no": [0], "net_value": [float(initial_equity)]})
    ordered = _trades_for_path_statistics(trades)
    returns = _returns_as_decimal(ordered)
    net_values = pd.concat(
        [pd.Series([float(initial_equity)]), initial_equity * (1.0 + returns).cumprod()],
        ignore_index=True,
    )
    result = pd.DataFrame(
        {
            "trade_no": range(0, len(net_values)),
            "net_value": net_values,
        }
    )
    dates = _trade_equity_dates(ordered)
    if dates is not None:
        result.insert(1, "date", dates)
    return result


def compute_trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """计算单策略回测绩效；输入只需要逐笔交易，避免和任何识别模块耦合。"""
    if trades.empty:
        return {key: 0.0 for key in STAT_KEYS}

    trades = _trades_for_path_statistics(trades)
    returns = _returns_as_decimal(trades)
    if returns.empty:
        return {key: 0.0 for key in STAT_KEYS}

    equity = _equity_with_initial_point(returns)
    drawdown = equity / equity.cummax() - 1.0
    return_stats = return_series_statistics(returns)
    wins = returns.loc[returns > 0]
    losses = returns.loc[returns < 0]
    gross_profit = _round_float(wins.sum())
    gross_loss = _round_float(abs(losses.sum()))
    avg_win = _mean_or_zero(wins)
    avg_loss = _mean_or_zero(losses)
    total_return = _round_float(equity.iloc[-1] - 1.0)
    max_drawdown = _round_float(drawdown.min())
    return_std = return_stats["return_std"]
    downside_deviation = return_stats["downside_deviation"]
    exposure_bars = pd.to_numeric(trades.get("holding_bars", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    risk_quality_stats = trade_risk_quality_statistics(trades)
    exposure_stats = trade_exposure_statistics(trades, returns)
    confidence_stats = sample_confidence_statistics(returns)

    return {
        "trade_count": float(len(returns)),
        "win_rate": _round_float((returns > 0).mean()),
        "win_rate_ci_lower": confidence_stats["win_rate_ci_lower"],
        "win_rate_ci_upper": confidence_stats["win_rate_ci_upper"],
        "total_return": total_return,
        "avg_return": return_stats["avg_return"],
        "avg_return_standard_error": confidence_stats["avg_return_standard_error"],
        "avg_return_ci_lower": confidence_stats["avg_return_ci_lower"],
        "avg_return_ci_upper": confidence_stats["avg_return_ci_upper"],
        "positive_expectancy_probability": confidence_stats["positive_expectancy_probability"],
        "max_drawdown": max_drawdown,
        "profit_factor": _ratio_or_inf(gross_profit, gross_loss),
        "expectancy": return_stats["avg_return"],
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": _ratio_or_zero(avg_win, abs(avg_loss)),
        "exposure_bars": exposure_stats["exposure_bars"],
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "return_std": return_std,
        "sharpe_per_trade": _ratio_or_zero(return_stats["avg_return"], return_std),
        "sortino_per_trade": _ratio_or_zero(return_stats["avg_return"], downside_deviation),
        "max_consecutive_wins": return_stats["max_consecutive_wins"],
        "max_consecutive_losses": return_stats["max_consecutive_losses"],
        "avg_holding_bars": _round_float(exposure_bars.mean()),
        "best_trade": return_stats["best_trade"],
        "worst_trade": return_stats["worst_trade"],
        "return_p05": return_stats["return_p05"],
        "return_p25": return_stats["return_p25"],
        "return_p50": return_stats["return_p50"],
        "return_p75": return_stats["return_p75"],
        "return_p95": return_stats["return_p95"],
        "cvar_95": return_stats["cvar_95"],
        "max_drawdown_duration": float(max_drawdown_duration(equity)),
        "recovery_factor": _ratio_or_zero(total_return, abs(max_drawdown)),
        "avg_r_multiple": risk_quality_stats["avg_r_multiple"],
        "median_r_multiple": risk_quality_stats["median_r_multiple"],
        "best_r_multiple": risk_quality_stats["best_r_multiple"],
        "worst_r_multiple": risk_quality_stats["worst_r_multiple"],
        "r_profit_factor": risk_quality_stats["r_profit_factor"],
        "system_quality_number": risk_quality_stats["system_quality_number"],
        "avg_mae_pct": risk_quality_stats["avg_mae_pct"],
        "avg_mfe_pct": risk_quality_stats["avg_mfe_pct"],
        "avg_mae_r": risk_quality_stats["avg_mae_r"],
        "avg_mfe_r": risk_quality_stats["avg_mfe_r"],
        "return_contribution": exposure_stats["return_contribution"],
        "return_per_exposure_bar": exposure_stats["return_per_exposure_bar"],
        "capital_turnover": exposure_stats["capital_turnover"],
        "avg_capital_fraction": exposure_stats["avg_capital_fraction"],
        "max_capital_fraction": exposure_stats["max_capital_fraction"],
        "margin_turnover": exposure_stats["margin_turnover"],
        "avg_margin_fraction": exposure_stats["avg_margin_fraction"],
        "max_margin_fraction": exposure_stats["max_margin_fraction"],
        "capital_exposure_bars": exposure_stats["capital_exposure_bars"],
        "margin_exposure_bars": exposure_stats["margin_exposure_bars"],
        "avg_capital_exposure_per_trade": exposure_stats["avg_capital_exposure_per_trade"],
        "avg_margin_exposure_per_trade": exposure_stats["avg_margin_exposure_per_trade"],
        "return_per_capital_exposure_bar": exposure_stats["return_per_capital_exposure_bar"],
        "return_per_margin_exposure_bar": exposure_stats["return_per_margin_exposure_bar"],
        "capital_weighted_raw_return": exposure_stats["capital_weighted_raw_return"],
    }


def compute_equity_statistics(equity_curve: pd.DataFrame, *, periods_per_year: float | None = None) -> dict[str, object]:
    """从时间净值曲线计算组合层收益和回撤，适合重叠持仓。"""
    if equity_curve.empty or "net_value" not in equity_curve.columns:
        return _empty_equity_statistics()
    data = equity_curve.copy()
    data["_row_order"] = range(len(data))
    data["net_value"] = pd.to_numeric(data["net_value"], errors="coerce")
    data = data.dropna(subset=["net_value"])
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        sort_columns = ["date", "_row_order"]
        if "trade_no" in data.columns:
            data["trade_no"] = pd.to_numeric(data["trade_no"], errors="coerce")
            sort_columns = ["date", "trade_no", "_row_order"]
        data = data.sort_values(sort_columns, kind="mergesort")
    data = data.reset_index(drop=True)
    net_value = data["net_value"].reset_index(drop=True)
    if net_value.empty:
        return _empty_equity_statistics()
    drawdown_stats = equity_drawdown_statistics(data, net_value)
    max_drawdown = float(drawdown_stats["max_drawdown"])
    equity_return_stats = equity_return_statistics(data, net_value, max_drawdown=max_drawdown, periods_per_year=periods_per_year)
    equity_exposure_stats = equity_exposure_statistics(data, net_value)
    return {
        "total_return": equity_return_stats["total_return"],
        **drawdown_stats,
        "equity_return_std": equity_return_stats["equity_return_std"],
        "equity_sharpe": equity_return_stats["equity_sharpe"],
        "equity_sortino": equity_return_stats["equity_sortino"],
        "annualized_return": equity_return_stats["annualized_return"],
        "annualized_volatility": equity_return_stats["annualized_volatility"],
        "annualized_sharpe": equity_return_stats["annualized_sharpe"],
        "annualized_sortino": equity_return_stats["annualized_sortino"],
        "calmar_ratio": equity_return_stats["calmar_ratio"],
        "avg_gross_exposure": equity_exposure_stats["avg_gross_exposure"],
        "max_gross_exposure": equity_exposure_stats["max_gross_exposure"],
        "avg_margin_exposure": equity_exposure_stats["avg_margin_exposure"],
        "max_margin_exposure": equity_exposure_stats["max_margin_exposure"],
        "exposure_bar_ratio": equity_exposure_stats["exposure_bar_ratio"],
        "avg_open_positions": equity_exposure_stats["avg_open_positions"],
        "max_open_positions": equity_exposure_stats["max_open_positions"],
        "avg_cash_ratio": equity_exposure_stats["avg_cash_ratio"],
        "min_cash_ratio": equity_exposure_stats["min_cash_ratio"],
        "max_cash_ratio": equity_exposure_stats["max_cash_ratio"],
        "avg_net_exposure": equity_exposure_stats["avg_net_exposure"],
        "min_net_exposure": equity_exposure_stats["min_net_exposure"],
        "max_net_exposure": equity_exposure_stats["max_net_exposure"],
    }


def compute_grouped_trade_statistics(trades: pd.DataFrame, *, by: str | Sequence[str]) -> pd.DataFrame:
    """按一个或多个字段拆分逐笔绩效；统计层不反向依赖任何策略实现。"""
    group_fields = _group_stat_fields(by)
    missing = [field for field in group_fields if field not in trades.columns]
    if missing:
        raise ValueError(f"trades 缺少分组字段：{', '.join(missing)}")
    if trades.empty:
        return pd.DataFrame(columns=pd.Index([*group_fields, *STAT_KEYS]))

    rows: list[dict[str, float | str]] = []
    grouped = trades.groupby(list(group_fields), sort=True, dropna=False)
    for values, group in grouped:
        value_tuple = values if isinstance(values, tuple) else (values,)
        row: dict[str, float | str] = {
            field: str(value)
            for field, value in zip(group_fields, value_tuple, strict=True)
        }
        row.update(compute_trade_statistics(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index([*group_fields, *STAT_KEYS]))


def _group_stat_fields(by: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(by, str):
        return (by,)
    fields = tuple(str(field) for field in by)
    if not fields:
        raise ValueError("至少需要一个分组字段。")
    return fields


def _trade_equity_dates(trades: pd.DataFrame) -> pd.Series | None:
    """用入场日作为初始点、出场日作为成交点，生成单策略净值时间轴。"""
    trade_dates = _coalesced_datetime_columns(trades, ("exit_date", "entry_date", "signal_date"))
    if trade_dates is None:
        return None
    trade_dates = trade_dates.ffill().bfill()
    if trade_dates.isna().all():
        return None
    start_candidates: list[pd.Series] = []
    for column in ("entry_date", "signal_date", "exit_date"):
        if column in trades.columns:
            start_candidates.append(pd.to_datetime(trades[column], errors="coerce").dropna())
    if not start_candidates:
        return None
    start_pool = pd.concat(start_candidates, ignore_index=True)
    if start_pool.empty:
        return None
    return pd.Series([start_pool.min(), *trade_dates.tolist()])


def _trades_for_path_statistics(trades: pd.DataFrame) -> pd.DataFrame:
    """路径类统计按真实平仓时间排序；没有时间字段时保留调用方给定顺序。"""
    realized_at = _coalesced_datetime_columns(trades, ("exit_date", "entry_date", "signal_date"))
    if realized_at is None or realized_at.isna().all():
        return trades
    result = trades.copy()
    result["_realized_at"] = realized_at.to_numpy()
    result["_row_order"] = range(len(result))
    sort_columns = ["_realized_at", "_row_order"]
    if "trade_no" in result.columns:
        result["_trade_no_sort"] = pd.to_numeric(result["trade_no"], errors="coerce")
        sort_columns = ["_realized_at", "_trade_no_sort", "_row_order"]
    return (
        result.sort_values(sort_columns, kind="mergesort", na_position="last")
        .drop(columns=[column for column in ("_realized_at", "_trade_no_sort", "_row_order") if column in result.columns])
        .reset_index(drop=True)
    )


def _coalesced_datetime_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series | None:
    """按优先级合并多个日期列，避免单列缺失导致时间轴丢失。"""
    result: pd.Series | None = None
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").reset_index(drop=True)
        result = values if result is None else result.fillna(values)
    return result


def _returns_as_decimal(trades: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(trades.get("return_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) / 100.0


def _equity_with_initial_point(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return pd.concat([pd.Series([1.0]), equity], ignore_index=True)


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


def _std_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.std(ddof=0))


def _ratio_or_inf(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return float("inf") if numerator > 0 else 0.0


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return 0.0


def _empty_equity_statistics() -> dict[str, object]:
    result: dict[str, object] = {key: 0.0 for key in EQUITY_STAT_KEYS}
    result.update(empty_drawdown_statistics())
    return result


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.max())


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
