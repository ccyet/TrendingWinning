from __future__ import annotations

from collections.abc import Mapping
import math

import pandas as pd


EXPERIMENT_DIAGNOSTIC_COLUMNS = pd.Index(
    ["section", "check", "status", "severity", "metric", "value", "threshold", "detail"]
)
CASE_DIAGNOSTIC_COLUMNS = pd.Index(
    [
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "case_name",
        "case_config_hash",
        *EXPERIMENT_DIAGNOSTIC_COLUMNS,
    ]
)


def experiment_diagnostic_report(
    stats: Mapping[str, object],
    *,
    data_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """生成实验级诊断摘要；只读统计结果，不改变回测和策略行为。"""
    rows = [
        _data_coverage_row(stats, data_coverage),
        _trade_sample_row(stats),
        _order_acceptance_row(stats),
        _strategy_filter_row(stats),
        _drawdown_pressure_row(stats),
        _profit_quality_row(stats),
        _monthly_stability_row(stats),
        _path_risk_row(stats),
        _capital_exposure_row(stats),
    ]
    return pd.DataFrame(rows, columns=EXPERIMENT_DIAGNOSTIC_COLUMNS)


def diagnostic_summary_fields(stats: Mapping[str, object]) -> dict[str, object]:
    """把诊断明细压成 sweep.csv 可筛选字段。"""
    report = experiment_diagnostic_report(stats)
    if report.empty:
        return _empty_diagnostic_summary_fields()
    status = report["status"].fillna("").astype(str)
    failed = float(status.eq("失败").sum())
    attention = float(status.eq("关注").sum())
    passed = float(status.eq("通过").sum())
    severity = pd.to_numeric(report["severity"], errors="coerce").fillna(0.0)
    worst = report.loc[severity.eq(severity.max())].head(1)
    primary_issue = str(worst.iloc[0]["check"]) if not worst.empty and float(severity.max()) > 0 else ""
    return {
        "diagnostic_failed_count": failed,
        "diagnostic_attention_count": attention,
        "diagnostic_passed_count": passed,
        "diagnostic_max_severity": float(severity.max()) if not severity.empty else 0.0,
        "diagnostic_primary_issue": primary_issue,
    }


def case_diagnostic_statistics(table: pd.DataFrame) -> pd.DataFrame:
    """按参数遍历 case 输出完整诊断明细，保留排名和配置指纹。"""
    if table.empty:
        return pd.DataFrame(columns=CASE_DIAGNOSTIC_COLUMNS)
    rows: list[dict[str, object]] = []
    for record in table.to_dict("records"):
        case_report = experiment_diagnostic_report(record)
        for diagnostic in case_report.to_dict("records"):
            rows.append(
                {
                    "sweep_rank": record.get("sweep_rank", pd.NA),
                    "pareto_rank": record.get("pareto_rank", pd.NA),
                    "is_pareto_efficient": record.get("is_pareto_efficient", pd.NA),
                    "case_name": str(record.get("case_name", "")),
                    "case_config_hash": str(record.get("case_config_hash", "")),
                    **diagnostic,
                }
            )
    return pd.DataFrame(rows, columns=CASE_DIAGNOSTIC_COLUMNS)


def _empty_diagnostic_summary_fields() -> dict[str, object]:
    return {
        "diagnostic_failed_count": 0.0,
        "diagnostic_attention_count": 0.0,
        "diagnostic_passed_count": 0.0,
        "diagnostic_max_severity": 0.0,
        "diagnostic_primary_issue": "",
    }


def _data_coverage_row(stats: Mapping[str, object], data_coverage: pd.DataFrame | None) -> dict[str, object]:
    threshold = _number(stats.get("data_min_coverage_threshold"), default=0.95)
    below_min_count = _number(stats.get("data_coverage_below_min_count"), default=0.0)
    failed_count = _number(stats.get("data_audit_failed_count"), default=0.0)
    value = _number(stats.get("data_weighted_coverage_ratio"), default=None)
    primary_detail = _primary_reason_detail(
        stats,
        reason_key="primary_data_issue",
        count_key="primary_data_issue_count",
        rate_key="primary_data_issue_rate",
        unit="项",
        rate_label="占数据问题",
    )
    if value is None:
        value = _coverage_value_from_frame(data_coverage)
    if value is None:
        return _row(
            "数据质量",
            "数据覆盖",
            "通过",
            "data_weighted_coverage_ratio",
            1.0,
            threshold,
            _append_detail("未发现覆盖率异常。", primary_detail),
        )
    if below_min_count > 0 or failed_count > 0 or value < threshold:
        return _row(
            "数据质量",
            "数据覆盖",
            "失败",
            "data_weighted_coverage_ratio",
            value,
            threshold,
            _append_detail("存在低于最低覆盖率或质量失败的数据，回测结果需要先排查数据。", primary_detail),
        )
    if value < 0.98:
        return _row(
            "数据质量",
            "数据覆盖",
            "关注",
            "data_weighted_coverage_ratio",
            value,
            0.98,
            _append_detail("覆盖率不低，但仍有缺口。", primary_detail),
        )
    return _row(
        "数据质量",
        "数据覆盖",
        "通过",
        "data_weighted_coverage_ratio",
        value,
        threshold,
        _append_detail("样本覆盖率满足当前要求。", primary_detail),
    )


def _trade_sample_row(stats: Mapping[str, object]) -> dict[str, object]:
    value = _number(stats.get("trade_count"), default=0.0)
    if value <= 0:
        return _row("交易质量", "交易样本", "失败", "trade_count", value, 1.0, "没有成交，统计结果不能用于评估策略质量。")
    if value < 30:
        return _row("交易质量", "交易样本", "关注", "trade_count", value, 30.0, "成交样本偏少，胜率和均值容易受偶然波动影响。")
    return _row("交易质量", "交易样本", "通过", "trade_count", value, 30.0, "成交样本数量达到基础评估要求。")


def _order_acceptance_row(stats: Mapping[str, object]) -> dict[str, object]:
    order_count = _number(stats.get("order_count"), default=0.0)
    value = _number(stats.get("acceptance_rate"), default=0.0)
    primary_detail = _primary_reason_detail(
        stats,
        reason_key="primary_rejected_reason",
        count_key="primary_rejected_reason_count",
        rate_key="primary_rejected_reason_rate",
        unit="笔",
        rate_label="占拒单",
    )
    if order_count <= 0:
        return _row("交易质量", "订单接受率", "失败", "acceptance_rate", value, 0.2, "没有订单进入撮合层，需要先检查策略是否生成信号。")
    if value < 0.2:
        return _row(
            "交易质量",
            "订单接受率",
            "关注",
            "acceptance_rate",
            value,
            0.2,
            _append_detail("订单接受率偏低，优先查看未成交、追价过远和风控拒绝原因。", primary_detail),
        )
    return _row(
        "交易质量",
        "订单接受率",
        "通过",
        "acceptance_rate",
        value,
        0.2,
        _append_detail("订单接受率处于可复盘范围。", primary_detail),
    )


def _strategy_filter_row(stats: Mapping[str, object]) -> dict[str, object]:
    value = _number(stats.get("strategy_filter_rejection_rate"), default=0.0)
    primary_detail = _primary_reason_detail(
        stats,
        reason_key="primary_strategy_rejected_reason",
        count_key="primary_strategy_rejected_reason_count",
        rate_key="primary_strategy_rejected_reason_rate",
        unit="条",
        rate_label="占过滤拒绝",
    )
    if value >= 0.6:
        return _row(
            "交易质量",
            "策略过滤",
            "关注",
            "strategy_filter_rejection_rate",
            value,
            0.6,
            _append_detail("策略层过滤比例较高，需要确认过滤参数是否过严。", primary_detail),
        )
    return _row(
        "交易质量",
        "策略过滤",
        "通过",
        "strategy_filter_rejection_rate",
        value,
        0.6,
        _append_detail("策略层过滤比例未见异常。", primary_detail),
    )


def _drawdown_pressure_row(stats: Mapping[str, object]) -> dict[str, object]:
    value = _number(stats.get("max_drawdown"), default=0.0)
    if value <= -0.35:
        return _row("风险", "回撤压力", "失败", "max_drawdown", value, -0.35, "最大回撤过深，当前参数风险不可接受。")
    if value <= -0.2:
        return _row("风险", "回撤压力", "关注", "max_drawdown", value, -0.2, "最大回撤偏深，需要结合收益和回撤区间复核。")
    return _row("风险", "回撤压力", "通过", "max_drawdown", value, -0.2, "最大回撤处于常规观察区间。")


def _profit_quality_row(stats: Mapping[str, object]) -> dict[str, object]:
    trade_count = _number(stats.get("trade_count"), default=0.0)
    value = _number(stats.get("profit_factor"), default=0.0)
    if math.isinf(value):
        return _row("收益", "收益质量", "通过", "profit_factor", value, 1.2, "未出现亏损交易，仍需结合样本量判断。")
    if trade_count > 0 and value < 1.0:
        return _row("收益", "收益质量", "失败", "profit_factor", value, 1.0, "盈亏因子低于 1，总亏损大于总盈利。")
    if trade_count > 0 and value < 1.2:
        return _row("收益", "收益质量", "关注", "profit_factor", value, 1.2, "盈亏因子偏低，策略边际不够厚。")
    return _row("收益", "收益质量", "通过", "profit_factor", value, 1.2, "盈亏因子达到基础观察要求。")


def _monthly_stability_row(stats: Mapping[str, object]) -> dict[str, object]:
    value = _number(stats.get("monthly_worst_return"), default=0.0)
    if value <= -0.2:
        return _row("收益", "月度稳定性", "失败", "monthly_worst_return", value, -0.2, "最差月度收益过低，需要复核参数抗极端行情能力。")
    if value <= -0.1:
        return _row("收益", "月度稳定性", "关注", "monthly_worst_return", value, -0.1, "存在明显亏损月份，需结合月度收益明细复核。")
    return _row("收益", "月度稳定性", "通过", "monthly_worst_return", value, -0.1, "月度最差收益未触发风险提示。")


def _path_risk_row(stats: Mapping[str, object]) -> dict[str, object]:
    value = _number(stats.get("avg_mae_r"), default=0.0)
    if value <= -1.2:
        return _row("风险", "路径风险", "失败", "avg_mae_r", value, -1.2, "平均最大不利波动超过初始风险，止损或入场质量需要重查。")
    if value <= -0.8:
        return _row("风险", "路径风险", "关注", "avg_mae_r", value, -0.8, "持仓过程平均回撤接近 1R，需查看交易路径分布。")
    return _row("风险", "路径风险", "通过", "avg_mae_r", value, -0.8, "平均持仓路径风险未触发提示。")


def _capital_exposure_row(stats: Mapping[str, object]) -> dict[str, object]:
    value = _number(stats.get("max_margin_exposure"), default=0.0)
    if value >= 1.5:
        return _row("组合", "资金暴露", "失败", "max_margin_exposure", value, 1.5, "最大保证金暴露过高，组合风险可能失控。")
    if value > 1.0:
        return _row("组合", "资金暴露", "关注", "max_margin_exposure", value, 1.0, "最大保证金暴露超过净值，需要复核仓位上限。")
    return _row("组合", "资金暴露", "通过", "max_margin_exposure", value, 1.0, "资金暴露未触发提示。")


def _coverage_value_from_frame(data_coverage: pd.DataFrame | None) -> float | None:
    if data_coverage is None or data_coverage.empty or "coverage_ratio" not in data_coverage.columns:
        return None
    values = pd.to_numeric(data_coverage["coverage_ratio"], errors="coerce").dropna()
    if values.empty:
        return None
    return _round_float(float(values.min()))


def _row(
    section: str,
    check: str,
    status: str,
    metric: str,
    value: float,
    threshold: float,
    detail: str,
) -> dict[str, object]:
    return {
        "section": section,
        "check": check,
        "status": status,
        "severity": {"通过": 0, "关注": 1, "失败": 2}.get(status, 0),
        "metric": metric,
        "value": _round_float(value),
        "threshold": _round_float(threshold),
        "detail": detail,
    }


def _number(value: object, *, default: float | None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    if math.isinf(numeric):
        return numeric
    return _round_float(numeric)


def _round_float(value: float) -> float:
    if math.isinf(value):
        return value
    return float(round(float(value), 12))


def _primary_reason_detail(
    stats: Mapping[str, object],
    *,
    reason_key: str,
    count_key: str,
    rate_key: str,
    unit: str,
    rate_label: str,
) -> str:
    reason = str(stats.get(reason_key) or "").strip()
    count = _number(stats.get(count_key), default=0.0) or 0.0
    rate = _number(stats.get(rate_key), default=0.0) or 0.0
    if not reason or count <= 0:
        return ""
    return f"主要原因：{reason} {_format_count(count)} {unit}，{rate_label} {rate:.1%}。"


def _append_detail(base: str, extra: str) -> str:
    return f"{base}{extra}" if extra else base


def _format_count(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(_round_float(value))
