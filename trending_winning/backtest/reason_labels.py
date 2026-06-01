from __future__ import annotations

# 数据、策略过滤、撮合和退出原因统一在这里维护，避免统计诊断和 Web 展示出现中文口径漂移。
DATA_ISSUE_LABELS: dict[str, str] = {
    "data_coverage_below_min": "覆盖率低于门槛",
    "data_audit_missing_file": "审计缺文件",
    "data_audit_missing_columns": "审计缺字段",
    "data_audit_no_window_data": "窗口无数据",
    "data_audit_quality_error": "数据质量异常",
    "data_audit_read_error": "审计读取失败",
    "data_inventory_missing_file": "缓存缺文件",
    "data_inventory_read_error": "缓存读取失败",
    "data_inventory_missing_columns": "缓存缺字段",
    "data_inventory_no_valid_rows": "缓存无有效K线",
    "limit_filter_daily_missing": "日K缺失",
    "limit_filter_daily_read_error": "日K读取失败",
    "limit_filter_daily_missing_columns": "日K缺字段",
    "limit_filter_daily_quality_error": "日K质量异常",
}

ORDER_REJECT_REASON_LABELS: dict[str, str] = {
    "no_fill": "未成交",
    "no_bars": "无K线数据",
    "no_liquidity": "无有效成交区间",
    "invalid_order": "订单字段无效",
    "duplicate_order_id": "订单ID重复",
    "already_open": "已有持仓未平仓",
    "actual_risk_too_high": "止损风险过大",
    "risk_too_large": "止损风险过大",
    "chase_too_far": "追价过远",
    "chase_too_large": "追价过远",
    "target_not_favorable": "目标价无效",
    "same_symbol_overlap": "同票已有持仓",
    "max_open_positions": "达到最大持仓数",
    "no_capital": "资金不足",
    "capital_limit": "资金上限",
    "sector_limit": "行业上限",
}

STRATEGY_FILTER_REASON_LABELS: dict[str, str] = {
    "side_mode_filtered": "交易方向过滤",
    "signal_bar_no_liquidity": "信号K无流动性",
    "higher_timeframe_mismatch": "大周期方向不一致",
    "higher_timeframe_missing": "无可用大周期上下文",
    "higher_timeframe_stale": "大周期信号过旧",
    "same_timeframe_middle": "同级别中部不交易",
    "terminal_false_breakout_risk": "末端假突破风险",
}

EXIT_REASON_LABELS: dict[str, str] = {
    "take_profit": "止盈",
    "trailing_take_profit": "回撤止盈",
    "stop_loss": "止损",
    "max_holding": "持有到期",
    "end_of_data": "样本结束",
    "other": "其他",
}

REASON_LABELS: dict[str, str] = {
    **DATA_ISSUE_LABELS,
    **ORDER_REJECT_REASON_LABELS,
    **STRATEGY_FILTER_REASON_LABELS,
}


def reason_label(reason: object) -> str:
    """把拒单、过滤和数据问题原因码转成中文；未知原因保留原码，便于排障。"""
    code = str(reason or "").strip()
    if not code:
        return ""
    return REASON_LABELS.get(code, code)


def reason_label_with_code(reason: object) -> str:
    """诊断明细使用“中文（code）”格式，同时兼顾阅读和回查原始 CSV。"""
    code = str(reason or "").strip()
    if not code:
        return ""
    label = reason_label(code)
    if label == code:
        return code
    return f"{label}（{code}）"


def exit_reason_label(reason: object) -> str:
    """把平仓原因码转成中文；未知原因保留原码。"""
    code = str(reason or "").strip()
    if not code:
        return ""
    return EXIT_REASON_LABELS.get(code, code)
