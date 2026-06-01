from __future__ import annotations

import pandas as pd

from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig

STRATEGY_SPACE_COLUMNS = ["策略空间", "当前设置", "触发与信号", "可能性分类", "边界/输出"]

DETECTOR_LABELS = {
    "trend": "趋势",
    "range": "区间",
    "channel": "通道",
    "reversal": "反转",
}
SIDE_MODE_LABELS = {"both": "多/空", "long_only": "仅多", "short_only": "仅空"}


def strategy_space_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> pd.DataFrame:
    """把回测配置转成可保存的策略执行空间摘要，便于跨机器复盘参数边界。"""
    if isinstance(config, SingleStrategyExperimentConfig):
        return _single_strategy_space_summary(config)
    return _portfolio_strategy_space_summary(config)


def _single_strategy_space_summary(config: SingleStrategyExperimentConfig) -> pd.DataFrame:
    rows = [
        _row(
            "样本",
            f"{len(config.symbols)} 只标的，{config.timeframe}，{config.start} 至 {config.end}",
            "只使用当前周期 K 线生成信号；日 K 只负责涨停开盘过滤。",
            _data_quality_summary(config),
            "样本不过关会写入 data_coverage.csv 和 limit_filter_audit.csv，相关标的不进入有效信号统计。",
        ),
        _row(
            "识别形态",
            f"单策略只运行一个识别模块：{_detector_label(config.detector)}",
            _detector_trigger_summary((config.detector,)),
            "单策略不混用其他形态；未启用模块的参数不会参与信号，也不会参与过滤。",
            "用于单独评估一个形态的胜率、R倍数、退出结构、持仓时间和信号生命周期。",
        ),
        _row(
            "信号条件",
            "信号K是已收完的确认 K，不使用未来 K 线。",
            _signal_condition_summary(),
            "可能出现顺势突破、回调后二次突破、失败突破反向、二次反转或无有效信号。",
            "信号不等于成交；信号只给出方向、挂单价、结构止损价和信号类型。",
        ),
        _row(
            "触发成交",
            f"{_side_mode_label(config.side_mode)}；盈亏比 {config.risk_reward:.2f}R",
            "信号K完成后，多头在信号K高点上方挂单，空头在信号K低点下方挂单；触发后按实际入场价、滑点和费用入账。",
            "成交、未触发、方向禁用、追价超限、结构止损风险超限会分开记录。",
            "盈亏比只决定固定目标价：目标距离 = 入场风险距离 x risk_reward。",
        ),
        _row(
            "开仓过滤",
            _filter_summary(config),
            _higher_timeframe_summary(config),
            _terminal_false_breakout_summary(config),
            "过滤只拒绝开仓订单，并写入 strategy_filter_decisions.csv 或 order_decisions.csv。",
        ),
        _row(
            "退出条件",
            f"最多持有 {config.max_holding_bars} 根 K；{_intrabar_policy_label(config.intrabar_exit_policy)}",
            _exit_trigger_summary(config),
            _exit_possibility_summary(config),
            "退出原因进入逐笔交易、退出原因绩效和开平仓路径绩效。",
        ),
        _row(
            "仓位规则",
            "满仓进出",
            "一笔持仓未关闭前，不允许第二笔、第三笔开仓。",
            "同向或反向新信号都会先经过单仓位检测，冲突订单记为已有持仓未平仓。",
            "适合验证单一形态本身，不处理组合资金分配。",
        ),
        _row(
            "或然分支",
            "本次回测会把每个候选信号归入清晰路径。",
            "候选信号 -> 策略过滤 -> 订单触发 -> 仓位检查 -> 退出",
            "通过成交、过滤拒单、未触发、方向禁用、追价/风险拒单、已有持仓拒单、止损/目标/回撤止盈/到期退出。",
            "先用决策表定位分支，再用 K 线和交易明细复核具体价格。",
        ),
        _row(
            "复盘输出",
            "先看策略K线运行区间，再看核心绩效和决策分布。",
            "K 线标注开多、开空、平仓、止损和回撤止盈。",
            "订单决策概览解释未成交、追价、止损风险过大和过滤拒单。",
            "输出 K线运行区间、净值、回撤、逐笔交易、策略过滤和信号形态统计。",
        ),
    ]
    return pd.DataFrame(rows, columns=STRATEGY_SPACE_COLUMNS)


def _portfolio_strategy_space_summary(config: PortfolioExperimentConfig) -> pd.DataFrame:
    rows = [
        _row(
            "样本",
            f"{len(config.symbols)} 只标的，{config.timeframe}，{config.start} 至 {config.end}",
            "每个形态先独立生成订单，再进入组合层排序和分配。",
            _data_quality_summary(config),
            "组合净值按全市场时间轴逐 K 重估，回撤使用持仓方向不利价格。",
        ),
        _row(
            "识别形态",
            f"组合策略启用识别模块：{_detector_list_label(config.detectors)}",
            _detector_trigger_summary(config.detectors),
            "组合策略可以同时比较趋势、通道、区间、反转；未选择的形态不生成订单，也不占组合容量。",
            "策略绩效、识别模块绩效和信号形态绩效会分开输出。",
        ),
        _row(
            "信号条件",
            "每个识别模块独立输出信号K、方向、挂单价和结构止损价。",
            _signal_condition_summary(),
            "可能出现多个形态同 K 发信号、同一股票多信号、方向相反信号或无有效信号。",
            "信号不等于成交；组合层只处理冲突和仓位，不重写 detector 的信号。",
        ),
        _row(
            "触发成交",
            f"{_side_mode_label(config.side_mode)}；盈亏比 {config.risk_reward:.2f}R",
            "各 detector 仍按信号K上方/下方挂单入场；同 K 多个信号先按策略优先级进入组合检查。",
            "成交、未触发、方向禁用、追价超限、结构止损风险超限、资金不足、达到最大持仓数、同票冲突会分开记录。",
            "被资金或容量拒绝的订单仍保留实际触发价、止损风险、追价距离和拒绝原因。",
        ),
        _row(
            "开仓过滤",
            _filter_summary(config),
            _higher_timeframe_summary(config),
            _terminal_false_breakout_summary(config),
            "过滤只处理是否允许开仓，不改变 detector 事件和持仓结算。",
        ),
        _row(
            "退出条件",
            f"最多持有 {config.max_holding_bars} 根 K；{_intrabar_policy_label(config.intrabar_exit_policy)}",
            _exit_trigger_summary(config),
            _exit_possibility_summary(config),
            "退出原因会进入组合层开平仓路径绩效和退出原因绩效。",
        ),
        _row(
            "仓位规则",
            _portfolio_allocation_summary(config),
            "组合层只做资金分配、保证金、预留现金、策略/行业上限和持仓互斥。",
            "单策略信号之间不互相修改；组合层负责冲突取舍和仓位大小。",
            "输出现金比例、净暴露、总暴露、保证金暴露和持仓数。",
        ),
        _row(
            "或然分支",
            "组合回测会把候选信号和组合分配结果分开。",
            "候选信号 -> 策略过滤 -> 订单触发 -> 组合分配 -> 退出",
            "通过成交、过滤拒单、未触发、资金不足、容量已满、同票冲突、止损/目标/回撤止盈/到期退出。",
            "先看 order_decisions.csv 的组合拒绝原因，再看策略/股票/行业分组绩效。",
        ),
        _row(
            "复盘输出",
            "先看策略K线运行区间，再看组合净值、回撤和分组绩效。",
            "K 线标注各笔开仓、平仓、止损和回撤止盈。",
            "订单决策统计解释未成交、资金不足、达到最大持仓数、同票冲突和过滤拒单。",
            "输出策略绩效、识别模块绩效、信号形态绩效、股票绩效和月度收益。",
        ),
    ]
    return pd.DataFrame(rows, columns=STRATEGY_SPACE_COLUMNS)


def _row(space: str, current: str, trigger: str, scenarios: str, boundary: str) -> dict[str, str]:
    return {
        "策略空间": space,
        "当前设置": current,
        "触发与信号": trigger,
        "可能性分类": scenarios,
        "边界/输出": boundary,
    }


def _detector_label(detector: object) -> str:
    return DETECTOR_LABELS.get(str(detector), str(detector))


def _detector_list_label(detectors: tuple[str, ...]) -> str:
    if not detectors:
        return "未选择形态"
    return "、".join(_detector_label(detector) for detector in detectors)


def _side_mode_label(value: object) -> str:
    return SIDE_MODE_LABELS.get(str(value), str(value))


def _detector_trigger_summary(detectors: tuple[str, ...]) -> str:
    selected = set(detectors)
    parts: list[str] = []
    if "trend" in selected:
        parts.append("趋势：趋势评分达标后，识别 H1/H2/L1/L2；H1/H2 偏多头，L1/L2 偏空头，H2/L2 要满足最少回调腿数。")
    if "channel" in selected:
        parts.append("通道：先用回归或摆动点确认中轴和上下轨，价格收盘越过上一根已完成边界后才生成突破信号。")
    if "range" in selected:
        parts.append("区间：先确认上下沿和中部，只在上沿或下沿做失败突破，中部不交易。")
    if "reversal" in selected:
        parts.append("反转：第一次反转默认观察，旧极端测试失败并完成结构确认后，第二次信号才允许交易。")
    return " ".join(parts) if parts else "未启用形态识别。"


def _signal_condition_summary() -> str:
    return (
        "多头信号K要给出向上突破或下沿失败测试，挂单价在信号K高点上方；"
        "空头信号K要给出向下突破或上沿失败测试，挂单价在信号K低点下方；"
        "结构止损取信号K相反端或识别模块给出的保护价。"
    )


def _data_quality_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    mode = "严格数据质量检查" if config.strict_data_quality else "宽松数据质量检查"
    coverage = (
        "不设最低覆盖率"
        if config.min_coverage_ratio is None or float(config.min_coverage_ratio) <= 0
        else f"最低覆盖率 {config.min_coverage_ratio:.0%}"
    )
    return f"{mode}；{coverage}"


def _higher_timeframe_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    if not config.higher_timeframe:
        return "未启用大周期方向过滤。"
    age = (
        "不限信号年龄"
        if config.higher_timeframe_max_age_minutes is None
        else f"信号有效 {config.higher_timeframe_max_age_minutes} 分钟"
    )
    return f"大周期方向过滤：{config.higher_timeframe}，{age}。"


def _terminal_false_breakout_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    if not config.terminal_false_breakout_enabled:
        return "末端假突破过滤关闭。"
    detectors = _detector_list_label(config.terminal_false_breakout_detectors)
    return (
        f"末端假突破过滤开启，作用于{detectors}；"
        f"持续 {config.terminal_false_breakout_min_regime_bars} 根、"
        f"远离中轴 {config.terminal_false_breakout_extension_atr_multiple:.1f}ATR、"
        f"贴边 {config.terminal_false_breakout_edge_min_count} 次、"
        f"弱突破 {config.terminal_false_breakout_weak_progress_atr:.2f}ATR、"
        f"影线 {config.terminal_false_breakout_wick_ratio:.0%}，"
        f"命中 {config.terminal_false_breakout_min_score} 分拒单。"
    )


def _filter_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    pieces = [
        _risk_limit_summary(config.max_actual_risk_pct, config.max_chase_pct),
        f"交易方向：{_side_mode_label(config.side_mode)}",
    ]
    if config.higher_timeframe:
        pieces.append("大周期方向过滤开启")
    if config.terminal_false_breakout_enabled:
        pieces.append("末端假突破过滤开启")
    return "；".join(pieces)


def _risk_limit_summary(max_actual_risk_pct: float | None, max_chase_pct: float | None) -> str:
    risk = "不限制结构止损风险" if max_actual_risk_pct is None else f"结构止损最大风险 {max_actual_risk_pct:.1%}"
    chase = "不限制追价" if max_chase_pct is None else f"最大追价距离 {max_chase_pct:.1%}"
    return f"{risk}；{chase}"


def _trailing_take_profit_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    if config.trailing_take_profit_drawdown_pct <= 0 and config.trailing_take_profit_ma_period < 2:
        return "盈利通道回撤止盈关闭。"
    return (
        f"盈利通道回撤止盈开启：启动浮盈 {config.trailing_take_profit_activation_pct:.1%}，"
        f"最大盈利回撤 {config.trailing_take_profit_drawdown_pct:.1%}，"
        f"当前周期均线 {config.trailing_take_profit_ma_period} 根。"
    )


def _exit_trigger_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    return (
        "入场后同时监控结构止损、固定目标、盈利通道回撤止盈、最大持有K数和样本结束；"
        f"固定目标距离按 {config.risk_reward:.2f}R 计算。"
    )


def _exit_possibility_summary(config: SingleStrategyExperimentConfig | PortfolioExperimentConfig) -> str:
    trailing = _trailing_take_profit_summary(config)
    return f"先触止损、先触固定目标、同K冲突、持有到期、样本结束都单独归因；{trailing}"


def _intrabar_policy_label(value: str) -> str:
    return "同K冲突止损优先" if value == "conservative" else "同K冲突止盈优先"


def _portfolio_allocation_summary(config: PortfolioExperimentConfig) -> str:
    overlap = "允许同票重叠" if config.allow_same_symbol_overlap else "不允许同票重叠"
    capital = "自动分配仓位" if config.capital_per_trade is None else f"固定单笔仓位 {config.capital_per_trade:.0%}"
    risk = "不使用风险预算" if config.risk_per_trade is None else f"单笔风险预算 {config.risk_per_trade:.1%}"
    limits: list[str] = []
    if config.strategy_priority:
        limits.append("策略优先级")
    if config.strategy_capital_limit:
        limits.append("策略资金上限")
    if config.sector_capital_limit:
        limits.append("行业资金上限")
    limit_text = "，".join(limits) if limits else "无额外策略/行业上限"
    return (
        f"最大持仓 {config.max_open_positions}；{capital}；最大单笔仓位 {config.max_capital_per_trade:.0%}；"
        f"{risk}；预留现金 {config.reserve_cash:.0%}；空头保证金 {config.short_margin_rate:.1f} 倍；"
        f"{overlap}；{limit_text}"
    )
