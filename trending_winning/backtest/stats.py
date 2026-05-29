from __future__ import annotations

import math
import re

import pandas as pd


STAT_KEYS = [
    "trade_count",
    "win_rate",
    "total_return",
    "avg_return",
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
    "capital_weighted_raw_return",
]


EQUITY_STAT_KEYS = [
    "total_return",
    "max_drawdown",
    "max_drawdown_duration",
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
    "exposure_bar_ratio",
    "avg_open_positions",
    "max_open_positions",
]

PERIOD_STAT_KEYS = [
    "count",
    "win_rate",
    "positive_count",
    "negative_count",
    "avg_return",
    "return_std",
    "best_return",
    "worst_return",
    "avg_drawdown",
    "worst_drawdown",
    "avg_observation_count",
    "max_consecutive_gains",
    "max_consecutive_losses",
    "max_recovery_periods",
    "underwater_ratio",
]

_PERIOD_RETURN_COLUMNS = pd.Index(
    ["period", "start", "end", "start_net_value", "end_net_value", "return", "max_drawdown", "observation_count"]
)

DECISION_METRIC_COLUMNS = {
    "actual_risk_pct": ("avg_actual_risk_pct", "max_actual_risk_pct"),
    "actual_chase_pct": ("avg_actual_chase_pct", "max_actual_chase_pct"),
    "actual_reward_to_risk": ("avg_actual_reward_to_risk", "min_actual_reward_to_risk"),
}


def compute_decision_reason_statistics(
    decisions: pd.DataFrame,
    *,
    group_fields: tuple[str, ...] = ("strategy_name", "detector_name"),
) -> pd.DataFrame:
    """按策略、状态和原因汇总决策分布；输入只需要决策表，不依赖策略实现。"""
    metric_columns = _decision_metric_output_columns()
    columns = pd.Index(
        [
            *group_fields,
            "status",
            "reason",
            "decision_count",
            "decision_rate",
            "group_decision_count",
            "group_decision_rate",
            *metric_columns,
        ]
    )
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    required = {*group_fields, "status"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"decisions 缺少分组字段：{', '.join(sorted(missing))}")

    data = decisions.copy()
    for column in group_fields:
        data[column] = data[column].fillna("").astype(str)
    data["status"] = data["status"].fillna("").astype(str)
    if "reason" in data.columns:
        data["reason"] = data["reason"].fillna("").astype(str)
    else:
        data["reason"] = ""
    for source_column in DECISION_METRIC_COLUMNS:
        data[source_column] = _decision_metric_series(data, source_column)

    total = len(data)
    group_keys = [*group_fields, "status", "reason"]
    grouped = data.groupby(group_keys, sort=True, dropna=False).size().reset_index(name="decision_count")
    grouped["decision_count"] = grouped["decision_count"].astype(int)
    grouped["decision_rate"] = grouped["decision_count"].map(lambda value: _ratio_or_zero(float(value), float(total)))
    grouped = _attach_group_decision_rates(data, grouped, group_fields=group_fields)
    metric_stats = _decision_metric_statistics(data, group_keys)
    grouped = grouped.merge(metric_stats, on=group_keys, how="left")
    grouped[metric_columns] = grouped[metric_columns].fillna(0.0)
    return grouped[columns]


def _attach_group_decision_rates(
    data: pd.DataFrame,
    grouped: pd.DataFrame,
    *,
    group_fields: tuple[str, ...],
) -> pd.DataFrame:
    """同时给出全局占比和组内占比；组内占比用于比较单个策略或过滤器内部结构。"""
    if not group_fields:
        result = grouped.copy()
        result["group_decision_count"] = int(len(data))
    else:
        group_totals = (
            data.groupby(list(group_fields), sort=True, dropna=False)
            .size()
            .reset_index(name="group_decision_count")
        )
        result = grouped.merge(group_totals, on=list(group_fields), how="left")
    result["group_decision_count"] = result["group_decision_count"].fillna(0).astype(int)
    result["group_decision_rate"] = [
        _ratio_or_zero(float(decision_count), float(group_count))
        for decision_count, group_count in zip(
            result["decision_count"],
            result["group_decision_count"],
            strict=False,
        )
    ]
    return result


def _decision_metric_output_columns() -> list[str]:
    columns: list[str] = []
    for output_columns in DECISION_METRIC_COLUMNS.values():
        columns.extend(output_columns)
    return columns


def _decision_metric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series([0.0] * len(data), index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce").fillna(0.0)


def _decision_metric_statistics(data: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    grouped = data.groupby(group_keys, sort=True, dropna=False)
    stats = grouped.agg(
        avg_actual_risk_pct=("actual_risk_pct", "mean"),
        max_actual_risk_pct=("actual_risk_pct", "max"),
        avg_actual_chase_pct=("actual_chase_pct", "mean"),
        max_actual_chase_pct=("actual_chase_pct", "max"),
        avg_actual_reward_to_risk=("actual_reward_to_risk", "mean"),
        min_actual_reward_to_risk=("actual_reward_to_risk", "min"),
    ).reset_index()
    for column in _decision_metric_output_columns():
        stats[column] = stats[column].map(_round_float)
    return stats


def build_equity_curve(trades: pd.DataFrame, initial_equity: float = 1.0) -> pd.DataFrame:
    """把逐笔收益转成净值曲线；有成交日期时同步保留时间轴。"""
    if trades.empty:
        return pd.DataFrame({"trade_no": [0], "net_value": [float(initial_equity)]})
    returns = _returns_as_decimal(trades)
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
    dates = _trade_equity_dates(trades)
    if dates is not None:
        result.insert(1, "date", dates)
    return result


def compute_trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """计算单策略回测绩效；输入只需要逐笔交易，避免和任何识别模块耦合。"""
    if trades.empty:
        return {key: 0.0 for key in STAT_KEYS}

    returns = _returns_as_decimal(trades)
    if returns.empty:
        return {key: 0.0 for key in STAT_KEYS}

    equity = _equity_with_initial_point(returns)
    drawdown = equity / equity.cummax() - 1.0
    wins = returns.loc[returns > 0]
    losses = returns.loc[returns < 0]
    gross_profit = _round_float(wins.sum())
    gross_loss = _round_float(abs(losses.sum()))
    avg_win = _mean_or_zero(wins)
    avg_loss = _mean_or_zero(losses)
    total_return = _round_float(equity.iloc[-1] - 1.0)
    max_drawdown = _round_float(drawdown.min())
    return_std = _round_float(returns.std(ddof=0))
    downside_std = _round_float(losses.std(ddof=0)) if not losses.empty else 0.0
    exposure_bars = pd.to_numeric(trades.get("holding_bars", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    r_multiple = _numeric_column(trades, "r_multiple")
    positive_r = r_multiple.loc[r_multiple > 0]
    negative_r = r_multiple.loc[r_multiple < 0]
    r_std = _std_or_zero(r_multiple)
    mae_pct = _numeric_column(trades, "mae_pct")
    mfe_pct = _numeric_column(trades, "mfe_pct")
    mae_r = _numeric_column(trades, "mae_r")
    mfe_r = _numeric_column(trades, "mfe_r")
    capital_fraction = _optional_numeric_column(trades, "capital_fraction")
    margin_fraction = _optional_numeric_column(trades, "margin_fraction")
    capital_exposure = _weighted_exposure_bars(exposure_bars, capital_fraction)
    margin_exposure = _weighted_exposure_bars(exposure_bars, margin_fraction)
    raw_returns = (
        _optional_numeric_column(trades, "raw_return_pct") / 100.0
        if "raw_return_pct" in trades.columns
        else returns
    )
    return_p05 = _quantile_or_zero(returns, 0.05)
    return_p25 = _quantile_or_zero(returns, 0.25)
    return_p50 = _quantile_or_zero(returns, 0.50)
    return_p75 = _quantile_or_zero(returns, 0.75)
    return_p95 = _quantile_or_zero(returns, 0.95)
    capital_turnover = _round_float(capital_fraction.sum()) if not capital_fraction.empty else 0.0

    return {
        "trade_count": float(len(returns)),
        "win_rate": _round_float((returns > 0).mean()),
        "total_return": total_return,
        "avg_return": _round_float(returns.mean()),
        "max_drawdown": max_drawdown,
        "profit_factor": _ratio_or_inf(gross_profit, gross_loss),
        "expectancy": _round_float(returns.mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": _ratio_or_zero(avg_win, abs(avg_loss)),
        "exposure_bars": _round_float(exposure_bars.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "return_std": return_std,
        "sharpe_per_trade": _ratio_or_zero(_round_float(returns.mean()), return_std),
        "sortino_per_trade": _ratio_or_zero(_round_float(returns.mean()), downside_std),
        "max_consecutive_wins": float(_max_streak(returns, positive=True)),
        "max_consecutive_losses": float(_max_streak(returns, positive=False)),
        "avg_holding_bars": _round_float(exposure_bars.mean()),
        "best_trade": _round_float(returns.max()),
        "worst_trade": _round_float(returns.min()),
        "return_p05": return_p05,
        "return_p25": return_p25,
        "return_p50": return_p50,
        "return_p75": return_p75,
        "return_p95": return_p95,
        "cvar_95": _mean_or_zero(returns.loc[returns <= return_p05]),
        "max_drawdown_duration": float(_max_drawdown_duration(equity)),
        "recovery_factor": _ratio_or_zero(total_return, abs(max_drawdown)),
        "avg_r_multiple": _mean_or_zero(r_multiple),
        "median_r_multiple": _median_or_zero(r_multiple),
        "best_r_multiple": _round_float(r_multiple.max()) if not r_multiple.empty else 0.0,
        "worst_r_multiple": _round_float(r_multiple.min()) if not r_multiple.empty else 0.0,
        "r_profit_factor": _ratio_or_inf(_round_float(positive_r.sum()), _round_float(abs(negative_r.sum()))),
        "system_quality_number": _system_quality_number(r_multiple, r_std),
        "avg_mae_pct": _mean_or_zero(mae_pct),
        "avg_mfe_pct": _mean_or_zero(mfe_pct),
        "avg_mae_r": _mean_or_zero(mae_r),
        "avg_mfe_r": _mean_or_zero(mfe_r),
        "return_contribution": _round_float(returns.sum()) if not capital_fraction.empty else 0.0,
        "capital_turnover": capital_turnover,
        "avg_capital_fraction": _mean_or_zero(capital_fraction),
        "max_capital_fraction": _max_or_zero(capital_fraction),
        "margin_turnover": _round_float(margin_fraction.sum()) if not margin_fraction.empty else 0.0,
        "avg_margin_fraction": _mean_or_zero(margin_fraction),
        "max_margin_fraction": _max_or_zero(margin_fraction),
        "capital_exposure_bars": _round_float(capital_exposure.sum()) if not capital_exposure.empty else 0.0,
        "margin_exposure_bars": _round_float(margin_exposure.sum()) if not margin_exposure.empty else 0.0,
        "avg_capital_exposure_per_trade": _mean_or_zero(capital_exposure),
        "avg_margin_exposure_per_trade": _mean_or_zero(margin_exposure),
        "capital_weighted_raw_return": _weighted_mean_or_zero(raw_returns, capital_fraction),
    }


def compute_equity_statistics(equity_curve: pd.DataFrame, *, periods_per_year: float | None = None) -> dict[str, float]:
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
    net_value = data["net_value"].reset_index(drop=True)
    if net_value.empty:
        return _empty_equity_statistics()
    returns = net_value.pct_change().dropna()
    losses = returns.loc[returns < 0]
    drawdown = net_value / net_value.cummax() - 1.0
    std = _std_or_zero(returns)
    downside_std = _std_or_zero(losses)
    total_return = _round_float(net_value.iloc[-1] / net_value.iloc[0] - 1.0)
    max_drawdown = _round_float(drawdown.min())
    avg_drawdown = _mean_or_zero(drawdown)
    ulcer_index = _round_float(math.sqrt(float(drawdown.pow(2).mean()))) if not drawdown.empty else 0.0
    time_under_water_ratio = _round_float(float(drawdown.lt(0).mean())) if not drawdown.empty else 0.0
    annual_periods = float(periods_per_year or _infer_periods_per_year(data))
    annualized_return = _annualized_return(net_value, annual_periods, len(returns))
    annualized_volatility = _round_float(std * math.sqrt(annual_periods))
    annualized_sharpe = _annualized_ratio(returns, std, annual_periods)
    annualized_sortino = _annualized_ratio(returns, downside_std, annual_periods)
    gross_exposure = _numeric_column(data, "gross_exposure")
    open_positions = _numeric_column(data, "open_positions")
    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": float(_max_drawdown_duration(net_value.reset_index(drop=True))),
        "equity_return_std": std,
        "equity_sharpe": _ratio_or_zero(_round_float(returns.mean()), std),
        "equity_sortino": _ratio_or_zero(_round_float(returns.mean()), downside_std),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": annualized_sharpe,
        "annualized_sortino": annualized_sortino,
        "calmar_ratio": _ratio_or_zero(annualized_return, abs(max_drawdown)),
        "avg_drawdown": avg_drawdown,
        "ulcer_index": ulcer_index,
        "time_under_water_ratio": time_under_water_ratio,
        "avg_gross_exposure": _mean_or_zero(gross_exposure),
        "max_gross_exposure": _round_float(gross_exposure.max()) if not gross_exposure.empty else 0.0,
        "exposure_bar_ratio": _round_float((gross_exposure > 0).mean()) if not gross_exposure.empty else 0.0,
        "avg_open_positions": _mean_or_zero(open_positions),
        "max_open_positions": _round_float(open_positions.max()) if not open_positions.empty else 0.0,
    }


def compute_grouped_trade_statistics(trades: pd.DataFrame, *, by: str) -> pd.DataFrame:
    """按策略或标的拆分逐笔绩效；统计层不反向依赖任何策略实现。"""
    if by not in trades.columns:
        raise ValueError(f"trades 缺少分组字段：{by}")
    if trades.empty:
        return pd.DataFrame(columns=pd.Index([by, *STAT_KEYS]))

    rows: list[dict[str, float | str]] = []
    for value, group in trades.groupby(by, sort=True, dropna=False):
        row: dict[str, float | str] = {by: str(value)}
        row.update(compute_trade_statistics(group))
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index([by, *STAT_KEYS]))


def compute_period_returns(equity_curve: pd.DataFrame, *, freq: str = "M") -> pd.DataFrame:
    """按自然周期拆分净值收益；用于月度/年度复盘和策略稳定性检查。"""
    if equity_curve.empty or not {"date", "net_value"}.issubset(equity_curve.columns):
        return pd.DataFrame(columns=_PERIOD_RETURN_COLUMNS)

    columns = ["date", "net_value", *(["trade_no"] if "trade_no" in equity_curve.columns else [])]
    data = equity_curve[columns].copy()
    data["_row_order"] = range(len(data))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["net_value"] = pd.to_numeric(data["net_value"], errors="coerce")
    data = data.dropna(subset=["date", "net_value"])
    sort_columns = ["date", "_row_order"]
    if "trade_no" in data.columns:
        data["trade_no"] = pd.to_numeric(data["trade_no"], errors="coerce")
        sort_columns = ["date", "trade_no", "_row_order"]
    data = data.sort_values(sort_columns, kind="mergesort")
    if data.empty:
        return pd.DataFrame(columns=_PERIOD_RETURN_COLUMNS)

    data["_period"] = data["date"].dt.to_period(freq)
    rows: list[dict[str, object]] = []
    previous_net_value: float | None = None
    for period, group in data.groupby("_period", sort=True):
        first_net_value = float(group.iloc[0]["net_value"])
        start_net_value = first_net_value if previous_net_value is None else float(previous_net_value)
        end_net_value = float(group.iloc[-1]["net_value"])
        period_equity = pd.concat(
            [pd.Series([start_net_value]), group["net_value"].reset_index(drop=True)],
            ignore_index=True,
        )
        period_drawdown = period_equity / period_equity.cummax() - 1.0
        rows.append(
            {
                "period": str(period),
                "start": group.iloc[0]["date"],
                "end": group.iloc[-1]["date"],
                "start_net_value": start_net_value,
                "end_net_value": end_net_value,
                "return": _ratio_or_zero(end_net_value - start_net_value, start_net_value),
                "max_drawdown": _round_float(period_drawdown.min()),
                "observation_count": int(len(group)),
            }
        )
        previous_net_value = end_net_value
    return pd.DataFrame(rows, columns=_PERIOD_RETURN_COLUMNS)


def compute_period_return_statistics(period_returns: pd.DataFrame, *, prefix: str = "period") -> dict[str, float]:
    """把月度/年度收益表压成稳定性摘要；只依赖周期收益表，和策略模块完全解耦。"""
    if period_returns.empty:
        return _empty_period_return_statistics(prefix)

    returns = _numeric_column(period_returns, "return")
    drawdown = _numeric_column(period_returns, "max_drawdown")
    observation_count = _numeric_column(period_returns, "observation_count")
    period_net_value = _period_net_value_series(period_returns, returns)
    period_underwater = period_net_value < period_net_value.cummax()
    period_count = float(len(returns))
    positive_count = float((returns > 0).sum())
    negative_count = float((returns < 0).sum())
    return {
        f"{prefix}_count": period_count,
        f"{prefix}_win_rate": _ratio_or_zero(positive_count, period_count),
        f"{prefix}_positive_count": positive_count,
        f"{prefix}_negative_count": negative_count,
        f"{prefix}_avg_return": _mean_or_zero(returns),
        f"{prefix}_return_std": _std_or_zero(returns),
        f"{prefix}_best_return": _max_or_zero(returns),
        f"{prefix}_worst_return": _min_or_zero(returns),
        f"{prefix}_avg_drawdown": _mean_or_zero(drawdown),
        f"{prefix}_worst_drawdown": _min_or_zero(drawdown),
        f"{prefix}_avg_observation_count": _mean_or_zero(observation_count),
        f"{prefix}_max_consecutive_gains": float(_max_streak(returns, positive=True)),
        f"{prefix}_max_consecutive_losses": float(_max_streak(returns, positive=False)),
        f"{prefix}_max_recovery_periods": float(_max_drawdown_duration(period_net_value)),
        f"{prefix}_underwater_ratio": _round_float(float(period_underwater.mean())) if not period_underwater.empty else 0.0,
    }


def summarize_order_decisions(order_decisions: pd.DataFrame) -> dict[str, float]:
    """汇总订单接受和拒绝原因；用于解释信号为什么没有变成成交。"""
    keys = [
        "order_count",
        "accepted_order_count",
        "rejected_order_count",
        "acceptance_rate",
        "rejection_rate",
        "rejected_no_fill_count",
        "rejected_no_liquidity_count",
        "rejected_no_bars_count",
        "rejected_invalid_order_count",
        "rejected_duplicate_order_id_count",
        "rejected_already_open_count",
        "rejected_max_open_positions_count",
        "rejected_same_symbol_overlap_count",
        "rejected_no_capital_count",
        "rejected_actual_risk_too_high_count",
        "rejected_chase_too_far_count",
        "rejected_target_not_favorable_count",
        "avg_accepted_capital_fraction",
        "max_accepted_capital_fraction",
        "avg_accepted_risk_fraction",
        "max_accepted_risk_fraction",
        "avg_accepted_margin_fraction",
        "max_accepted_margin_fraction",
        "avg_executed_actual_risk_pct",
        "max_executed_actual_risk_pct",
        "avg_executed_actual_chase_pct",
        "max_executed_actual_chase_pct",
        "avg_executed_actual_reward_to_risk",
        "min_executed_actual_reward_to_risk",
    ]
    if order_decisions.empty or "status" not in order_decisions.columns:
        return {key: 0.0 for key in keys}
    status = order_decisions["status"].astype(str)
    if "reason" in order_decisions.columns:
        reason = order_decisions["reason"].astype(str)
    else:
        reason = pd.Series([""] * len(order_decisions), index=order_decisions.index, dtype=str)
    order_count = float(len(order_decisions))
    accepted = status.eq("accepted")
    rejected = status.eq("rejected")
    accepted_count = float(accepted.sum())
    rejected_count = float(rejected.sum())
    executed = _executed_decisions(order_decisions)
    result = {
        "order_count": order_count,
        "accepted_order_count": accepted_count,
        "rejected_order_count": rejected_count,
        "acceptance_rate": _ratio_or_zero(accepted_count, order_count),
        "rejection_rate": _ratio_or_zero(rejected_count, order_count),
        "rejected_no_fill_count": float((rejected & reason.eq("no_fill")).sum()),
        "rejected_no_liquidity_count": float((rejected & reason.eq("no_liquidity")).sum()),
        "rejected_no_bars_count": float((rejected & reason.eq("no_bars")).sum()),
        "rejected_invalid_order_count": float((rejected & reason.eq("invalid_order")).sum()),
        "rejected_duplicate_order_id_count": float((rejected & reason.eq("duplicate_order_id")).sum()),
        "rejected_already_open_count": float((rejected & reason.eq("already_open")).sum()),
        "rejected_max_open_positions_count": float((rejected & reason.eq("max_open_positions")).sum()),
        "rejected_same_symbol_overlap_count": float((rejected & reason.eq("same_symbol_overlap")).sum()),
        "rejected_no_capital_count": float((rejected & reason.eq("no_capital")).sum()),
        "rejected_actual_risk_too_high_count": float((rejected & reason.eq("actual_risk_too_high")).sum()),
        "rejected_chase_too_far_count": float((rejected & reason.eq("chase_too_far")).sum()),
        "rejected_target_not_favorable_count": float((rejected & reason.eq("target_not_favorable")).sum()),
        "avg_accepted_capital_fraction": _accepted_mean(order_decisions, accepted, "capital_fraction"),
        "max_accepted_capital_fraction": _accepted_max(order_decisions, accepted, "capital_fraction"),
        "avg_accepted_risk_fraction": _accepted_mean(order_decisions, accepted, "risk_fraction"),
        "max_accepted_risk_fraction": _accepted_max(order_decisions, accepted, "risk_fraction"),
        "avg_accepted_margin_fraction": _accepted_mean(order_decisions, accepted, "margin_fraction"),
        "max_accepted_margin_fraction": _accepted_max(order_decisions, accepted, "margin_fraction"),
        "avg_executed_actual_risk_pct": _masked_mean(order_decisions, executed, "actual_risk_pct"),
        "max_executed_actual_risk_pct": _masked_max(order_decisions, executed, "actual_risk_pct"),
        "avg_executed_actual_chase_pct": _masked_mean(order_decisions, executed, "actual_chase_pct"),
        "max_executed_actual_chase_pct": _masked_max(order_decisions, executed, "actual_chase_pct"),
        "avg_executed_actual_reward_to_risk": _masked_mean(order_decisions, executed, "actual_reward_to_risk"),
        "min_executed_actual_reward_to_risk": _masked_min(order_decisions, executed, "actual_reward_to_risk"),
    }
    result.update(_rejected_reason_counts(reason, rejected, prefix="rejected"))
    return result


def summarize_strategy_filter_decisions(filter_decisions: pd.DataFrame) -> dict[str, float]:
    """汇总策略层过滤结果；用于解释信号为什么没有进入撮合层。"""
    keys = [
        "strategy_signal_count",
        "strategy_accepted_signal_count",
        "strategy_rejected_signal_count",
        "strategy_filter_acceptance_rate",
        "strategy_filter_rejection_rate",
        "strategy_rejected_higher_timeframe_mismatch_count",
        "strategy_rejected_higher_timeframe_no_context_count",
        "strategy_rejected_higher_timeframe_stale_count",
        "strategy_rejected_invalid_order_key_count",
        "strategy_rejected_signal_bar_no_liquidity_count",
    ]
    if filter_decisions.empty or "status" not in filter_decisions.columns:
        return {key: 0.0 for key in keys}
    status = filter_decisions["status"].astype(str)
    if "reason" in filter_decisions.columns:
        reason = filter_decisions["reason"].astype(str)
    else:
        reason = pd.Series([""] * len(filter_decisions), index=filter_decisions.index, dtype=str)
    signal_count = float(len(filter_decisions))
    accepted = status.eq("accepted")
    rejected = status.eq("rejected")
    accepted_count = float(accepted.sum())
    rejected_count = float(rejected.sum())
    result = {
        "strategy_signal_count": signal_count,
        "strategy_accepted_signal_count": accepted_count,
        "strategy_rejected_signal_count": rejected_count,
        "strategy_filter_acceptance_rate": _ratio_or_zero(accepted_count, signal_count),
        "strategy_filter_rejection_rate": _ratio_or_zero(rejected_count, signal_count),
        "strategy_rejected_higher_timeframe_mismatch_count": float(
            (rejected & reason.eq("higher_timeframe_mismatch")).sum()
        ),
        "strategy_rejected_higher_timeframe_no_context_count": float(
            (rejected & reason.eq("higher_timeframe_no_context")).sum()
        ),
        "strategy_rejected_higher_timeframe_stale_count": float((rejected & reason.eq("higher_timeframe_stale")).sum()),
        "strategy_rejected_invalid_order_key_count": float((rejected & reason.eq("invalid_order_key")).sum()),
        "strategy_rejected_signal_bar_no_liquidity_count": float(
            (rejected & reason.eq("signal_bar_no_liquidity")).sum()
        ),
    }
    result.update(_rejected_reason_counts(reason, rejected, prefix="strategy_rejected"))
    return result


def _rejected_reason_counts(reason: pd.Series, rejected: pd.Series, *, prefix: str) -> dict[str, float]:
    """动态汇总所有拒绝原因；新增风控或策略过滤原因无需修改统计白名单。"""
    rejected_reasons = reason.loc[rejected].fillna("").astype(str)
    rejected_reasons = rejected_reasons.loc[rejected_reasons.ne("")]
    if rejected_reasons.empty:
        return {}
    counts = rejected_reasons.value_counts(sort=True)
    return {
        f"{prefix}_{_metric_safe_reason(str(reason_value))}_count": float(count)
        for reason_value, count in counts.items()
        if _metric_safe_reason(str(reason_value))
    }


def _metric_safe_reason(reason: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", reason.strip().lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


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


def _median_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.median())


def _quantile_or_zero(values: pd.Series, quantile: float) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.quantile(quantile))


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


def _empty_equity_statistics() -> dict[str, float]:
    return {key: 0.0 for key in EQUITY_STAT_KEYS}


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.max())


def _min_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.min())


def _empty_period_return_statistics(prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": 0.0 for key in PERIOD_STAT_KEYS}


def _period_net_value_series(period_returns: pd.DataFrame, returns: pd.Series) -> pd.Series:
    """优先使用真实期末净值；没有期末净值时用周期收益复原一条相对净值。"""
    if "end_net_value" in period_returns.columns:
        return _numeric_column(period_returns, "end_net_value")
    if returns.empty:
        return pd.Series(dtype=float)
    return (1.0 + returns).cumprod().reset_index(drop=True)


def _weighted_mean_or_zero(values: pd.Series, weights: pd.Series) -> float:
    if values.empty or weights.empty:
        return 0.0
    length = min(len(values), len(weights))
    value_slice = values.iloc[:length]
    weight_slice = weights.iloc[:length]
    denominator = float(weight_slice.sum())
    if denominator <= 0:
        return 0.0
    return _round_float(float((value_slice * weight_slice).sum()) / denominator)


def _weighted_exposure_bars(holding_bars: pd.Series, fractions: pd.Series) -> pd.Series:
    """资金或保证金占用乘以持仓 K 数，用于衡量长期占资压力。"""
    if holding_bars.empty or fractions.empty:
        return pd.Series(dtype=float)
    length = min(len(holding_bars), len(fractions))
    return (
        holding_bars.iloc[:length].reset_index(drop=True)
        * fractions.iloc[:length].reset_index(drop=True)
    )


def _accepted_values(frame: pd.DataFrame, accepted: pd.Series, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame.loc[accepted, column], errors="coerce").dropna().astype(float).reset_index(drop=True)


def _accepted_mean(frame: pd.DataFrame, accepted: pd.Series, column: str) -> float:
    return _mean_or_zero(_accepted_values(frame, accepted, column))


def _accepted_max(frame: pd.DataFrame, accepted: pd.Series, column: str) -> float:
    values = _accepted_values(frame, accepted, column)
    return _round_float(values.max()) if not values.empty else 0.0


def _executed_decisions(frame: pd.DataFrame) -> pd.Series:
    if "actual_entry_price" not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index, dtype=bool)
    entry_price = pd.to_numeric(frame["actual_entry_price"], errors="coerce").fillna(0.0)
    return entry_price > 0


def _masked_values(frame: pd.DataFrame, mask: pd.Series, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna().astype(float).reset_index(drop=True)


def _masked_mean(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    return _mean_or_zero(_masked_values(frame, mask, column))


def _masked_max(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = _masked_values(frame, mask, column)
    return _round_float(values.max()) if not values.empty else 0.0


def _masked_min(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = _masked_values(frame, mask, column)
    return _round_float(values.min()) if not values.empty else 0.0


def _infer_periods_per_year(equity_curve: pd.DataFrame) -> float:
    if "date" not in equity_curve.columns:
        return 252.0
    dates = pd.to_datetime(equity_curve["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(dates) < 2:
        return 252.0
    delta_seconds = dates.diff().dropna().dt.total_seconds()
    if delta_seconds.empty:
        return 252.0
    median_seconds = float(delta_seconds.median())
    if median_seconds <= 0:
        return 252.0
    trading_day_seconds = 4.0 * 60.0 * 60.0
    if median_seconds < trading_day_seconds:
        return _round_float(252.0 * trading_day_seconds / median_seconds)
    return 252.0


def _annualized_return(net_value: pd.Series, periods_per_year: float, observed_periods: int) -> float:
    if observed_periods <= 0 or periods_per_year <= 0:
        return 0.0
    start = float(net_value.iloc[0])
    end = float(net_value.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return _round_float((end / start) ** (periods_per_year / observed_periods) - 1.0)


def _annualized_ratio(returns: pd.Series, denominator: float, periods_per_year: float) -> float:
    if returns.empty or denominator <= 0 or periods_per_year <= 0:
        return 0.0
    return _round_float(float(returns.mean()) / denominator * math.sqrt(periods_per_year))


def _system_quality_number(r_multiple: pd.Series, r_std: float) -> float:
    """按 Van Tharp SQN 口径衡量 R 倍数序列质量。"""
    if r_multiple.empty or r_std <= 0:
        return 0.0
    return _round_float(math.sqrt(len(r_multiple)) * float(r_multiple.mean()) / r_std)


def _max_streak(returns: pd.Series, *, positive: bool) -> int:
    best = 0
    current = 0
    for value in returns:
        is_match = value > 0 if positive else value < 0
        if is_match:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_drawdown_duration(equity: pd.Series) -> int:
    peak = -math.inf
    current = 0
    best = 0
    for value in equity:
        if value >= peak:
            peak = float(value)
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
