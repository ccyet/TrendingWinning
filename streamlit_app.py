from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

from trending_winning.backtest.drawdown import drawdown_episodes, price_path_drawdown_inputs
from trending_winning.backtest.engine import BacktestConfig, run_backtest
from trending_winning.backtest.experiment import (
    run_portfolio_experiment,
    run_single_strategy_experiment,
)
from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig
from trending_winning.data.repository import BacktestDataBundle, MarketDataRepository, available_symbols
from trending_winning.data.schema import normalize_bars, normalize_symbol
from trending_winning.data.symbols import DEFAULT_STOCK_NAME_BY_CODE
from trending_winning.multitimeframe import scan_timeframes
from trending_winning.strategies.signal_bar import SUPPORTED_SIDE_MODES
from trending_winning.strategy import StrategyConfig, scan_bars

# 分钟级长样本必须完整画出，避免 Altair 默认 5000 行限制截断回测 K 线。
alt.data_transformers.disable_max_rows()

DEFAULT_DATA_ROOT = "/Users/a1234/Desktop/trend-backtest/data/market/daily"
DEFAULT_OUTPUT_ROOT = "runs"
# 数据管理要补日 K，策略执行仍只跑分钟级，避免日 K 误入同级别策略回测。
DATA_TIMEFRAMES = ["1d", "5m", "15m", "30m", "60m"]
INTRADAY_TIMEFRAMES = ["5m", "15m", "30m", "60m"]
SIDE_MODE_LABELS = {"both": "多/空", "long_only": "仅多", "short_only": "仅空"}
BACKTEST_HELP_TEXT = {
    "timeframe": "本次扫描或回测使用的 K 线周期。策略周期只支持分钟级，日 K 只用于涨停开盘过滤。",
    "symbols": "输入股票代码，多个代码用英文逗号分隔，例如 000001.SZ,600519.SH。",
    "start_date": "回测样本开始日期，系统只读取这个日期之后的本地 K 线。",
    "end_date": "回测样本结束日期，结束日之后的 K 线不会进入样本。",
    "scope_mode": "选择回测类型：旧突破用于兼容早期逻辑；单策略只测一个形态；组合策略处理多个形态的资金分配。",
    "take_profit": "旧突破回测使用的固定止盈比例，例如 0.060 表示 6%。单策略和组合策略的目标平仓价由结构止损距离和盈亏比计算。",
    "stop_loss": "旧突破回测使用的固定止损比例，例如 0.030 表示 3%。单策略和组合策略使用各形态输出的信号K结构止损价。",
    "structural_stop_loss": "现代策略的止损参数不是固定百分比；每笔订单使用信号K或形态模块输出的结构止损价，再用“结构止损最大风险”限制止损距离。",
    "enable_trailing_take_profit": "开启后，所有策略在实际成交后共用盈利通道回撤止盈；关闭时下方三个参数强制按 0 处理。比例止盈、均线回撤止盈和启动浮盈门槛互相独立，不依赖结构止损、固定止盈或盈亏比目标。",
    "trailing_take_profit_activation_pct": "可选的盈利通道启动条件。实际成交后，上一根已完成 K 的浮盈先达到该比例才开始跟踪；0 表示不设门槛，不会关闭比例止盈或均线回撤止盈。",
    "trailing_take_profit_drawdown_pct": "比例止盈参数，也就是最大盈利回撤幅度。按上一根已完成 K 的最大盈利价位计算平仓线：例，多头入场 100、最高浮盈到 108，参数 0.020 时回撤线约 105.84，跌破即平仓；空头按最低价后的反弹计算。",
    "trailing_take_profit_ma_period": "均线回撤止盈周期。由用户输入 K 数，用当前回测周期上一根已完成 K 的均线作为移动平仓线，独立于比例止盈；0 表示关闭。",
    "max_holding": "最多持有多少根当前周期 K 线，到期仍未止盈止损就平仓。",
    "fee_rate": "单边手续费率，0.0003 表示 0.03%。",
    "slippage_bps": "撮合滑点，1 bps 等于 0.01%。",
    "initial_equity": "净值曲线起始资金，默认 1.0，便于比较不同策略。",
    "intrabar_exit_policy": "同一根 K 线同时打到止盈和止损时的处理方式。",
    "strict_data_quality": "开启后会拒绝明显缺失、重复或异常的本地 K 线样本；关闭后仍会记录审计，无法读取或缺字段的文件会被跳过。",
    "min_coverage_ratio": "样本窗口内实际 K 线数量占理论数量的最低比例，0 表示不额外限制。",
    "higher_timeframe": "用更大周期判断主方向，只过滤逆大周期方向的订单，不改形态识别结果。",
    "higher_timeframe_max_age": "大周期信号允许滞后的最长分钟数，0 表示不限制信号年龄。",
    "single_detector": "只测试一种形态识别模块，便于单独评估趋势、区间、通道或反转。",
    "terminal_false_breakout_enabled": "开启后，只在开仓前过滤同级别趋势或通道末端的疑似假突破；不改变形态识别、撮合和仓位分配。",
    "terminal_false_breakout_detectors": "选择过滤器作用的形态模块。默认只作用于趋势和通道，区间、反转可单独开启。",
    "terminal_false_breakout_lookback": "计算通道上沿、下沿和中轴时使用的同级别 K 线数量。",
    "terminal_false_breakout_atr_period": "ATR 波动率周期，用来把远离中轴、突破推进不足等价格幅度标准化。",
    "terminal_false_breakout_min_regime_bars": "同向趋势或通道至少持续多少根 K 线后，才认为可能进入末端阶段。",
    "terminal_false_breakout_extension_atr_multiple": "价格距离通道中轴超过多少个 ATR 才算过度延伸。",
    "terminal_false_breakout_edge_lookback": "统计最近多少根 K 线是否反复贴近通道边缘。",
    "terminal_false_breakout_edge_pos": "贴近通道边缘的判定位置；0.90 表示上轨附近 10% 或下轨附近 10%。",
    "terminal_false_breakout_edge_min_count": "最近窗口内至少多少次贴近通道边缘，才计入末端风险。",
    "terminal_false_breakout_weak_progress_atr": "突破推进不足阈值；创新高或新低幅度低于该 ATR 倍数时，说明突破力度偏弱。",
    "terminal_false_breakout_wick_ratio": "上影线/下影线占整根 K 线振幅的比例；比例越大，越像冲高或杀跌失败。",
    "terminal_false_breakout_min_score": "末端风险命中分。持续、延伸、贴边、弱突破、影线五项中达到该分数才拒绝开仓。",
    "risk_reward": "盈亏比指向平仓信号，不决定开仓信号。开仓信号先给出开仓价和结构止损价；盈亏比只把这段风险距离换算成固定目标平仓价：多头目标平仓价 = 开仓价 + (开仓价 - 止损价) × 盈亏比，空头相反。",
    "trend_lookback": "趋势评分使用的回看 K 线数量，越大越重视较长结构。",
    "trend_min_score": "趋势形态入选的最低评分，越高越严格。",
    "trend_h2_min_pullback_legs": "H 是 High 1/High 2，指回调后第 1/2 次突破前一根 K 线高点；L 是 Low 1/Low 2，指反弹后第 1/2 次跌破前一根 K 线低点。这里限制 H2/L2 至少经历几段反向摆动，不是数单根 K 线。",
    "range_lookback": "判断交易区间上下沿时使用的回看 K 线数量。",
    "channel_lookback": "计算趋势通道或摆动点通道时使用的回看 K 线数量。",
    "channel_method": "回归通道适合连续斜率，摆动点通道更贴近人工画高低点连线。",
    "channel_sigma": "回归通道宽度倍数，越大通道越宽、突破越少。",
    "max_actual_risk_pct": "现代策略可调的止损参数。它限制开仓价到结构止损价的最大距离，0 表示不限制；超过该比例会拒单。",
    "max_chase_pct": "信号 K 突破价到实际成交价的最大追价距离，0 表示不限制。",
    "side_mode": "控制策略允许生成的订单方向。多/空表示多头和空头都做；仅多会过滤空头信号；仅空会过滤多头信号。",
    "reversal_lookback": "识别旧高/旧低、二次测试和结构确认时使用的回看 K 线数量。",
    "reversal_old_extreme_tolerance_pct": "价格接近旧高/旧低的容忍范围，例如 0.010 表示 1%。",
    "reversal_require_old_extreme_test": "开启后，反转必须先测试旧高/旧低并失败。",
    "reversal_require_structure_confirmation": "开启后，反转必须先出现结构确认，不做第一次反转。",
    "trend_strong_close_pos": "强收盘阈值，按收盘价在当根 K 线高低区间的位置计算。",
    "trend_min_body_ratio": "实体占整根 K 线振幅的最低比例，用来过滤影线过长的弱信号。",
    "trend_pullback_lookback": "统计 H/L 回撤腿时向前观察的 K 线数量。",
    "range_middle_low": "交易区间中部的下边界，区间策略避开中部，只看上下沿。",
    "range_middle_high": "交易区间中部的上边界，价格落在中部时不做同级别策略。",
    "range_false_break_buffer": "失败突破需要越过区间边界的最小幅度，0 表示只要破位即可。",
    "range_strong_close_pos": "区间反向信号的强收盘阈值。",
    "range_min_score": "区间形态入选的最低评分，越高越严格。",
    "channel_break_buffer": "通道突破需要超过通道边界的最小幅度。",
    "channel_swing_left_bars": "摆动高/低点左侧需要多少根 K 线确认。",
    "channel_swing_right_bars": "摆动高/低点右侧需要多少根 K 线确认。",
    "reversal_strong_close_pos": "反转信号 K 的强收盘阈值。",
    "reversal_min_body_ratio": "反转信号 K 的最小实体比例。",
    "max_open_positions": "组合最多同时持有的订单数量。",
    "risk_per_trade": "每笔交易最多亏损的资金比例，0 表示不用风险预算自动定仓。",
    "short_margin_rate": "做空时占用保证金的倍数。",
    "capital_per_trade": "固定每笔交易使用的资金比例，0 表示由其他规则决定。",
    "max_capital_per_trade": "单笔交易最多占用的资金比例。",
    "reserve_cash": "组合始终保留的现金比例，不参与开仓。",
    "allow_same_symbol_overlap": "关闭时，同一只股票同一时间只允许一笔持仓。",
    "strategy_priority": "多个策略同 K 竞争资金时的优先级，数字越小越优先。",
    "strategy_capital_limit": "单个策略最多可占用的组合资金比例。",
    "sector_capital_limit": "单个行业最多可占用的组合资金比例。",
    "symbol_sector_map": "股票到行业的映射，用于行业资金上限。",
    "landmark_lookback": "旧突破策略用于识别标志 K 的回看窗口。",
    "landmark_range_multiple": "标志 K 的振幅需要达到近期平均振幅的倍数。",
    "trigger_volume_multiple": "突破 K 的成交量需要达到近期均量的倍数。",
    "close_buffer": "突破收盘价需要超过关键价位的最小幅度。",
    "require_landmark": "开启后，突破 K 本身也必须满足标志 K 条件。",
    "save_outputs": "开启后保存 config、成交、净值、统计、拒单、过滤和数据审计文件，方便复盘和在 Windows 侧复现。",
    "output_parent": "选择实验产物保存的父目录，系统会按实验名自动生成子文件夹。",
}
DISPLAY_COLUMN_LABELS = {
    "stock_code": "股票名称",
    "symbol": "股票名称",
    "strategy_name": "策略",
    "detector_name": "形态模块",
    "event_type": "信号形态",
    "side": "方向",
    "side_mode": "交易方向",
    "exit_reason": "退出原因",
    "take_profit_exit_count": "止盈退出次数",
    "take_profit_exit_rate": "止盈退出比例",
    "trailing_take_profit_exit_count": "回撤止盈退出次数",
    "trailing_take_profit_exit_rate": "回撤止盈退出比例",
    "stop_loss_exit_count": "止损退出次数",
    "stop_loss_exit_rate": "止损退出比例",
    "max_holding_exit_count": "持有到期退出次数",
    "max_holding_exit_rate": "持有到期退出比例",
    "end_of_data_exit_count": "样本结束退出次数",
    "end_of_data_exit_rate": "样本结束退出比例",
    "other_exit_count": "其他退出次数",
    "other_exit_rate": "其他退出比例",
    "avg_take_profit_exit_rate": "平均止盈退出比例",
    "avg_trailing_take_profit_exit_rate": "平均回撤止盈退出比例",
    "avg_stop_loss_exit_rate": "平均止损退出比例",
    "avg_max_holding_exit_rate": "平均持有到期退出比例",
    "status": "状态",
    "reason": "原因",
    "section": "诊断模块",
    "check": "检查项",
    "severity": "严重度",
    "metric": "指标字段",
    "threshold": "阈值",
    "detail": "说明",
    "filter_name": "过滤模块",
    "parameter": "参数",
    "value": "取值",
    "case_count": "参数组数",
    "pareto_case_count": "Pareto候选数",
    "pareto_hit_rate": "Pareto命中率",
    "positive_return_case_count": "正收益组数",
    "positive_return_rate": "正收益率",
    "risk_adjusted_rank": "风险质量排名",
    "risk_adjusted_score": "风险质量评分",
    "avg_risk_adjusted_score": "平均风险质量评分",
    "median_risk_adjusted_score": "中位风险质量评分",
    "avg_total_return": "平均总收益",
    "median_total_return": "中位总收益",
    "std_total_return": "总收益标准差",
    "best_total_return": "最好总收益",
    "worst_total_return": "最差总收益",
    "avg_max_drawdown": "平均最大回撤",
    "avg_monthly_worst_return": "平均月度最差收益",
    "avg_monthly_return_std": "平均月度收益波动",
    "monthly_best_return_period": "月度最好收益周期",
    "monthly_worst_return_period": "月度最差收益周期",
    "monthly_worst_drawdown_period": "月度最深回撤周期",
    "monthly_current_underwater_periods": "当前连续水下月数",
    "best_sweep_rank": "最佳排名",
    "best_case_name": "最佳参数组",
    "best_case_config_hash": "最佳配置指纹",
    "best_risk_adjusted_case_name": "风险质量最佳参数组",
    "best_risk_adjusted_case_config_hash": "风险质量最佳配置指纹",
    "best_risk_adjusted_sweep_rank": "风险质量最佳原排名",
    "best_risk_adjusted_score": "最高风险质量评分",
    "worst_risk_adjusted_score": "最低风险质量评分",
    "timeframe": "周期",
    "date": "日期",
    "period": "周期",
    "episode_rank": "回撤排名",
    "episode_no": "回撤序号",
    "start_at": "开始时间",
    "trough_at": "触底时间",
    "recovery_at": "修复时间",
    "peak_net_value": "高点净值",
    "trough_net_value": "低点净值",
    "depth": "回撤幅度",
    "underwater_bars": "水下K数",
    "recovery_bars": "修复K数",
    "recovered": "已修复",
    "dimension": "分布维度",
    "bucket": "区间",
    "bucket_order": "区间顺序",
    "data_inventory_signature": "数据快照指纹",
    "data_inventory_row_count": "缓存检查项数",
    "data_inventory_cached_count": "缓存可用数",
    "data_inventory_unavailable_count": "缓存不可用数",
    "data_inventory_missing_file_count": "缓存缺文件数",
    "data_inventory_read_error_count": "缓存读取失败数",
    "data_inventory_missing_columns_count": "缓存缺字段数",
    "data_inventory_no_valid_rows_count": "缓存无有效K线数",
    "data_inventory_total_rows": "缓存总K线数",
    "data_inventory_total_file_size_bytes": "缓存文件总字节",
    "start": "开始",
    "end": "结束",
    "signal_date": "信号时间",
    "entry_date": "入场时间",
    "exit_date": "出场时间",
    "trade_no": "交易序号",
    "order_id": "订单ID",
    "event_id": "信号ID",
    "trade_count": "交易次数",
    "avg_holding_bars": "平均持有K数",
    "win_rate": "胜率",
    "win_rate_ci_lower": "胜率95%下限",
    "win_rate_ci_upper": "胜率95%上限",
    "total_return": "总收益",
    "avg_return": "平均收益",
    "avg_r_multiple": "平均R倍数",
    "avg_mae_r": "平均最大不利R",
    "avg_mfe_r": "平均最大有利R",
    "annualized_return": "年化收益",
    "annualized_sharpe": "年化Sharpe",
    "annualized_sortino": "年化Sortino",
    "avg_return_standard_error": "平均收益标准误",
    "avg_return_ci_lower": "平均收益95%下限",
    "avg_return_ci_upper": "平均收益95%上限",
    "positive_expectancy_probability": "正期望概率",
    "return_pct": "收益率",
    "raw_return_pct": "原始收益率",
    "return": "收益率",
    "max_drawdown": "最大回撤",
    "max_drawdown_start_at": "最大回撤开始",
    "max_drawdown_trough_at": "最大回撤触底",
    "max_drawdown_recovery_at": "最大回撤修复",
    "current_drawdown": "当前回撤",
    "current_underwater_bars": "当前水下K数",
    "calmar_ratio": "Calmar比率",
    "ulcer_index": "Ulcer指数",
    "profit_factor": "盈亏因子",
    "expectancy": "期望收益",
    "avg_win": "平均盈利",
    "avg_loss": "平均亏损",
    "payoff_ratio": "盈亏比",
    "holding_bars": "持有K数",
    "market_bar_count": "市场K数",
    "exposure_bars": "场内K数",
    "exposure_bar_ratio": "场内时间比例",
    "best_trade": "最好单笔",
    "worst_trade": "最差单笔",
    "net_value": "净值",
    "drawdown_net_value": "回撤估算净值",
    "start_net_value": "期初净值",
    "end_net_value": "期末净值",
    "observation_count": "观察点数",
    "mae_pct": "最大不利波动",
    "mfe_pct": "最大有利波动",
    "r_multiple": "R倍数",
    "system_quality_number": "SQN系统质量",
    "actual_risk_pct": "实际止损风险",
    "actual_chase_pct": "追价距离",
    "actual_reward_to_risk": "实际盈亏比",
    "executed_order_count": "触发成交候选数",
    "accepted_executed_order_count": "最终成交数",
    "avg_accepted_actual_risk_pct": "成交平均止损风险",
    "max_accepted_actual_risk_pct": "成交最大止损风险",
    "avg_accepted_actual_chase_pct": "成交平均追价距离",
    "max_accepted_actual_chase_pct": "成交最大追价距离",
    "avg_accepted_actual_reward_to_risk": "成交平均实际盈亏比",
    "min_accepted_actual_reward_to_risk": "成交最低实际盈亏比",
    "avg_executed_actual_risk_pct": "候选平均止损风险",
    "max_executed_actual_risk_pct": "候选最大止损风险",
    "avg_executed_actual_chase_pct": "候选平均追价距离",
    "max_executed_actual_chase_pct": "候选最大追价距离",
    "avg_executed_actual_reward_to_risk": "候选平均实际盈亏比",
    "min_executed_actual_reward_to_risk": "候选最低实际盈亏比",
    "capital_fraction": "资金占用",
    "risk_fraction": "风险占用",
    "margin_fraction": "保证金占用",
    "margin_exposure": "保证金暴露",
    "avg_cash_ratio": "平均现金比例",
    "avg_margin_exposure": "平均保证金暴露",
    "max_margin_exposure": "最大保证金暴露",
    "coverage_ratio": "K线覆盖率",
    "limit_pct": "涨跌停幅度",
    "limit_up_open": "涨停开盘",
    "limit_filter_daily_missing_count": "日K缺失数",
    "limit_filter_daily_read_error_count": "日K读取失败数",
    "limit_filter_daily_missing_columns_count": "日K缺字段数",
    "limit_filter_daily_quality_error_count": "日K质量异常数",
    "strategy_rejected_terminal_false_breakout_risk_count": "末端假突破过滤数",
    "terminal_false_breakout_score": "末端风险分",
    "terminal_false_breakout_context": "末端过滤上下文",
}
PERCENT_POINT_COLUMNS = {"return_pct", "raw_return_pct", "mae_pct", "mfe_pct", "avg_mae_pct", "avg_mfe_pct"}
DISPLAY_VALUE_MAP = {
    "side": {"long": "多头", "short": "空头"},
    "status": {
        "accepted": "已接受",
        "rejected": "已拒绝",
        "filled": "已成交",
        "no_fill": "未成交",
        "cached": "已有缓存",
        "missing": "缺失",
        "missing_file": "缓存缺文件",
        "read_error": "缓存读取失败",
        "missing_columns": "缓存缺字段",
        "no_valid_rows": "缓存无有效K线",
        "ok": "正常",
        "coverage_below_min": "覆盖率低于门槛",
        "quality_error": "质量异常",
        "no_window_data": "窗口无数据",
        "daily_missing": "日K缺失",
        "daily_read_error": "日K读取失败",
        "daily_missing_columns": "日K缺字段",
        "daily_quality_error": "日K质量异常",
    },
    "exit_reason": {
        "take_profit": "止盈",
        "trailing_take_profit": "回撤止盈",
        "stop_loss": "止损",
        "max_holding": "持有到期",
        "end_of_data": "样本结束",
    },
    "detector_name": {"trend": "趋势", "range": "区间", "channel": "通道", "reversal": "反转"},
    "strategy_name": {
        "trend_signal_bar": "趋势形态",
        "range_signal_bar": "区间形态",
        "channel_signal_bar": "通道形态",
        "reversal_signal_bar": "反转形态",
    },
    "event_type": {
        "trend_signal_bar": "趋势信号K",
        "range_signal_bar": "区间信号K",
        "channel_signal_bar": "通道信号K",
        "reversal_signal_bar": "反转信号K",
        "bull_h1_setup": "H1 多头第一次入场",
        "bull_h2_setup": "H2 多头二次入场",
        "bear_l1_setup": "L1 空头第一次入场",
        "bear_l2_setup": "L2 空头二次入场",
        "failed_breakdown": "跌破失败做多",
        "failed_breakout": "突破失败做空",
        "no_trade_middle": "区间中部不交易",
        "channel_overshoot_up": "通道上破",
        "channel_break_down": "通道下破",
        "first_reversal_watch_short": "第一次空头反转观察",
        "second_reversal_short": "第二次空头反转",
    },
    "reason": {
        "": "",
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
        "side_mode_filtered": "交易方向过滤",
        "trailing_take_profit": "回撤止盈",
        "terminal_false_breakout_risk": "末端假突破风险",
    },
    "filter_name": {
        "terminal_false_breakout_filter": "末端假突破过滤",
        "higher_timeframe_alignment": "大周期方向过滤",
    },
    "side_mode": {"both": "多/空", "long_only": "仅多", "short_only": "仅空"},
}


@dataclass(frozen=True)
class BacktestScopeInputs:
    """回测样本范围和模式选择。"""

    symbols: list[str]
    timeframe: str
    start: str
    end: str
    mode: str


@dataclass(frozen=True)
class BacktestRiskInputs:
    """回测基础风控、成本和撮合冲突规则。"""

    take_profit: float
    stop_loss: float
    max_holding: int
    fee_rate: float
    slippage_bps: float
    initial_equity: float
    intrabar_exit_policy: str
    trailing_take_profit_enabled: bool
    trailing_take_profit_activation_pct: float
    trailing_take_profit_drawdown_pct: float
    trailing_take_profit_ma_period: int


@dataclass(frozen=True)
class BacktestDataQualityInputs:
    """回测数据质量检查。"""

    strict_data_quality: bool
    min_coverage_ratio: float | None


@dataclass(frozen=True)
class HigherTimeframeInputs:
    """大周期方向过滤设置。"""

    higher_timeframe: str
    higher_timeframe_max_age_minutes: int | None


@dataclass(frozen=True)
class TerminalFalseBreakoutInputs:
    """末端假突破开仓过滤设置。"""

    enabled: bool
    detectors: tuple[str, ...]
    lookback: int
    atr_period: int
    min_regime_bars: int
    extension_atr_multiple: float
    edge_lookback: int
    edge_pos: float
    edge_min_count: int
    weak_progress_atr: float
    wick_ratio: float
    min_score: int


@dataclass(frozen=True)
class BacktestOutputInputs:
    """保存实验产物和运行按钮状态。"""

    save_outputs: bool
    output_dir: str
    run_clicked: bool


@dataclass(frozen=True)
class SingleStrategyInputs:
    """单策略形态识别参数。"""

    detector: str
    experiment_name: str
    risk_reward: float
    side_mode: str
    trend_lookback: int
    trend_min_score: float
    trend_h2_min_pullback_legs: int
    range_lookback: int
    channel_lookback: int
    channel_method: str
    channel_sigma: float
    max_actual_risk_pct: float | None
    max_chase_pct: float | None
    reversal_lookback: int
    reversal_old_extreme_tolerance_pct: float
    reversal_require_old_extreme_test: bool
    reversal_require_structure_confirmation: bool
    advanced_detector: dict[str, float | int]
    terminal_false_breakout: TerminalFalseBreakoutInputs


@dataclass(frozen=True)
class PortfolioAllocationInputs:
    """组合层资金、容量和暴露约束。"""

    detectors: tuple[str, ...]
    experiment_name: str
    risk_reward: float
    side_mode: str
    max_open_positions: int
    risk_per_trade: float | None
    short_margin_rate: float
    capital_per_trade: float | None
    max_capital_per_trade: float
    reserve_cash: float
    allow_same_symbol_overlap: bool
    strategy_priority_text: str
    strategy_capital_limit_text: str
    sector_capital_limit_text: str
    symbol_sector_map_text: str
    max_actual_risk_pct: float | None
    max_chase_pct: float | None


@dataclass(frozen=True)
class PortfolioDetectorInputs:
    """组合回测中各形态识别模块的参数。"""

    trend_lookback: int
    channel_lookback: int
    trend_min_score: float
    channel_sigma: float
    range_lookback: int
    reversal_lookback: int
    trend_h2_min_pullback_legs: int
    channel_method: str
    reversal_old_extreme_tolerance_pct: float
    reversal_require_old_extreme_test: bool
    reversal_require_structure_confirmation: bool
    advanced_detector: dict[str, float | int]
    terminal_false_breakout: TerminalFalseBreakoutInputs


def main() -> None:
    st.set_page_config(page_title="TrendingWinning", layout="wide")
    st.title("TrendingWinning")
    st.caption("TDX K 线、标志K、趋势通道、突破触发与回测工作台")

    with st.sidebar:
        st.subheader("数据目录")
        data_root = _directory_picker("行情根目录", DEFAULT_DATA_ROOT, key="data_root_picker")
        adjust = st.selectbox("复权", ["qfq", "hfq", ""], index=0)
        use_default_tdx_path = st.checkbox("使用系统默认 TDX 路径", value=True, key="tdx_default_path")
        tdx_path = (
            ""
            if use_default_tdx_path
            else str(_directory_picker("TDX PYPlugins/user", Path.home(), key="tdx_path_picker"))
        )
        if sys.platform == "darwin":
            st.caption("Mac 本机通达信不支持取数；真实 TDX 请求请用 CLI 的 Parallels runtime 或在 Win 侧运行页面。")

    fetch_tab, scan_tab, backtest_tab = st.tabs(["TDX K线", "策略扫描", "回测"])
    with fetch_tab:
        _fetch_panel(data_root, adjust, tdx_path)
    with scan_tab:
        _scan_panel(data_root, adjust)
    with backtest_tab:
        _backtest_panel(data_root, adjust)


def _directory_picker(
    label: str,
    default: str | Path,
    *,
    key: str,
    disabled: bool = False,
    help_text: str = "",
) -> Path:
    """本地文件夹选择器；优先弹出系统选择框，保留站内浏览作为补充。"""
    selected_key = f"{key}_selected_path"
    current_key = f"{key}_current_path"
    default_path = Path(default).expanduser()
    if selected_key not in st.session_state:
        st.session_state[selected_key] = str(default_path)
    if current_key not in st.session_state:
        st.session_state[current_key] = str(_initial_browse_directory(default_path))

    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)
    picker_cols = st.columns([1, 3])
    with picker_cols[0]:
        if st.button(
            "选择文件夹",
            key=f"{key}_native_select",
            disabled=disabled,
            help="打开系统文件夹选择框",
        ):
            try:
                selected = _open_native_directory_dialog(st.session_state[selected_key], f"选择{label}")
            except (ImportError, OSError, RuntimeError) as exc:
                st.warning(f"无法打开系统文件夹选择框：{exc}")
            else:
                if selected is not None:
                    st.session_state[selected_key] = str(selected)
                    st.session_state[current_key] = str(_existing_directory(selected))
    with picker_cols[1]:
        st.caption(f"已选文件夹：{_display_path(st.session_state[selected_key])}")

    quick_cols = st.columns([4, 1])
    with quick_cols[0]:
        quick_root = st.selectbox(
            f"{label}快速位置",
            _directory_option_strings(_quick_directory_roots(default_path)),
            key=f"{key}_quick_root",
            format_func=_display_path,
            disabled=disabled,
        )
    with quick_cols[1]:
        if st.button("→", key=f"{key}_open_quick", disabled=disabled, help="打开快速位置"):
            st.session_state[current_key] = quick_root

    current = _existing_directory(Path(st.session_state[current_key]).expanduser())
    st.caption(f"当前位置：{_display_path(str(current))}")
    child_choice = st.selectbox(
        f"{label}子文件夹",
        _folder_entry_options(current),
        key=f"{key}_child_dir",
        format_func=_display_path,
        disabled=disabled,
    )
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("进入", key=f"{key}_enter", disabled=disabled, help="进入选中的子文件夹"):
            st.session_state[current_key] = child_choice
    with action_cols[1]:
        if st.button("选中", key=f"{key}_select_current", disabled=disabled, help="选定当前文件夹"):
            st.session_state[selected_key] = str(current)
    with action_cols[2]:
        if st.button("默认", key=f"{key}_reset", disabled=disabled, help="恢复默认文件夹"):
            st.session_state[selected_key] = str(default_path)
            st.session_state[current_key] = str(_initial_browse_directory(default_path))

    selected = Path(st.session_state[selected_key]).expanduser()
    return selected


def _open_native_directory_dialog(initial_directory: str | Path, title: str) -> Path | None:
    """打开系统文件夹选择框；仅在本地桌面运行 Streamlit 时可用。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("当前 Python 环境缺少 tkinter，无法弹出系统选择框") from exc

    root: tk.Tk | None = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        return _resolve_native_directory_choice(initial_directory, title, filedialog.askdirectory)
    except tk.TclError as exc:
        raise RuntimeError("当前会话没有可用桌面窗口，无法弹出系统选择框") from exc
    finally:
        if root is not None:
            root.destroy()


def _resolve_native_directory_choice(
    initial_directory: str | Path,
    title: str,
    askdirectory: Callable[..., str],
) -> Path | None:
    """执行文件夹选择并归一化结果；空字符串表示用户取消选择。"""
    initial = _existing_directory(Path(initial_directory).expanduser())
    selected = askdirectory(title=title, initialdir=str(initial), mustexist=False)
    if not selected:
        return None
    return Path(selected).expanduser()


def _initial_browse_directory(path: Path) -> Path:
    if path.exists() and path.is_dir():
        return path
    parent = path.parent
    return parent if parent.exists() and parent.is_dir() else Path.home()


def _existing_directory(path: Path) -> Path:
    if path.exists() and path.is_dir():
        return path
    for parent in path.parents:
        if parent.exists() and parent.is_dir():
            return parent
    return Path.home()


def _quick_directory_roots(default_path: Path) -> list[Path]:
    candidates = [
        _initial_browse_directory(default_path),
        Path.cwd(),
        Path(DEFAULT_OUTPUT_ROOT),
        Path.home(),
        Path.home() / "Desktop",
        Path.home() / "Documents",
        Path.home() / "Downloads",
    ]
    for parent in default_path.parents:
        candidates.append(parent)
    for drive in ("C:/", "D:/", "E:/"):
        drive_path = Path(drive)
        if drive_path.exists():
            candidates.append(drive_path)
    return _unique_paths(candidates)


def _folder_entry_options(current: Path) -> list[str]:
    entries = [current.parent]
    entries.extend(_child_directories(current))
    return _directory_option_strings(entries)


def _child_directories(current: Path, *, limit: int = 120) -> list[Path]:
    try:
        children = [
            child
            for child in current.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]
    except (OSError, PermissionError):
        return []
    return sorted(children, key=lambda item: item.name.lower())[:limit]


def _directory_option_strings(paths: list[Path]) -> list[str]:
    return [str(path.expanduser()) for path in _unique_paths(paths)]


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _display_path(value: str) -> str:
    path = Path(value).expanduser()
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    return "~" if str(relative) == "." else f"~/{relative}"


def _side_mode_label(value: object) -> str:
    return SIDE_MODE_LABELS.get(str(value), str(value))


def _stock_name_label(value: object, stock_names: Mapping[str, str] | None = None) -> str:
    """把证券代码转成界面可读名称；已知代码在统计表中只展示股票名称。"""
    code = str(value).strip().upper()
    if not code:
        return ""
    name = (stock_names or DEFAULT_STOCK_NAME_BY_CODE).get(code)
    return name if name else f"未知名称（{code}）"


def _prepare_display_frame(frame: pd.DataFrame, *, stock_names: Mapping[str, str] | None = None) -> pd.DataFrame:
    """把回测展示表转成中文列名、中文枚举和已格式化字符串。"""
    if frame.empty:
        return frame.copy()
    used_labels: set[str] = set()
    display = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        label = _unique_display_label(_display_column_label(str(column)), used_labels)
        display[label] = [_format_display_value(str(column), value, stock_names=stock_names) for value in frame[column]]
    return display.reset_index(drop=True)


def _style_display_frame(frame: pd.DataFrame):
    """表格只做展示样式，不修改底层回测数据。"""
    return frame.style.set_properties(
        **{
            "text-align": "center",
            "border-bottom": "1px solid #e2e8f0",
            "padding": "8px 10px",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("background-color", "#eef2f7"),
                    ("font-weight", "700"),
                    ("color", "#0f172a"),
                    ("border-bottom", "1px solid #cbd5e1"),
                    ("padding", "9px 10px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("border-bottom", "1px solid #e2e8f0"),
                    ("padding", "8px 10px"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#f8fafc")],
            },
        ]
    )


def _render_display_table(
    title: str,
    frame: pd.DataFrame,
    *,
    tail: int | None = None,
    stock_names: Mapping[str, str] | None = None,
) -> None:
    if frame.empty:
        return
    data = frame.tail(tail) if tail is not None else frame
    st.markdown(f"##### {title}")
    st.dataframe(
        _style_display_frame(_prepare_display_frame(data, stock_names=stock_names)),
        use_container_width=True,
        hide_index=True,
    )


def _render_data_coverage_chart(
    data_coverage: pd.DataFrame,
    *,
    stock_names: Mapping[str, str] | None = None,
) -> None:
    chart_data = _data_coverage_chart_frame(data_coverage, stock_names=stock_names)
    if chart_data.empty:
        return
    st.markdown("##### 数据覆盖率概览")
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "K线覆盖率:Q",
                title="K线覆盖率",
                scale=alt.Scale(domain=[0.0, 1.0]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y("样本:N", title="", sort="-x"),
            color=alt.Color("状态:N", title="状态"),
            tooltip=[
                "股票名称:N",
                "周期:N",
                alt.Tooltip("K线覆盖率:Q", format=".2%"),
                "状态:N",
                alt.Tooltip("缺失K数:Q", format=".0f"),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _data_coverage_chart_frame(
    data_coverage: pd.DataFrame,
    *,
    stock_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """把数据审计结果转成覆盖率图表字段；数值保留原始比例，展示时再格式化。"""
    columns = ["样本", "股票名称", "周期", "K线覆盖率", "状态", "缺失K数"]
    if data_coverage.empty or "coverage_ratio" not in data_coverage.columns:
        return pd.DataFrame(columns=columns)
    frame = data_coverage.copy()
    frame["股票名称"] = frame.get("stock_code", pd.Series([""] * len(frame))).map(
        lambda value: _stock_name_label(value, stock_names)
    )
    frame["周期"] = frame.get("timeframe", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["K线覆盖率"] = pd.to_numeric(frame["coverage_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    frame["状态"] = frame.get("status", pd.Series([""] * len(frame))).fillna("").astype(str).map(
        lambda value: DISPLAY_VALUE_MAP.get("status", {}).get(value, value)
    )
    frame["缺失K数"] = pd.to_numeric(frame.get("missing_rows", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
    frame["样本"] = frame["股票名称"] + " · " + frame["周期"]
    return (
        frame.loc[:, columns]
        .sort_values(["K线覆盖率", "样本"], ascending=[True, True], kind="mergesort")
        .reset_index(drop=True)
    )


def _render_order_decision_charts(order_decisions: pd.DataFrame) -> None:
    funnel_data = _order_decision_funnel_frame(order_decisions)
    reason_data = _order_reject_reason_chart_frame(order_decisions)
    if funnel_data.empty and reason_data.empty:
        return
    st.markdown("##### 订单决策概览")
    funnel_col, reason_col = st.columns([1, 1])
    with funnel_col:
        if not funnel_data.empty:
            funnel_chart = (
                alt.Chart(funnel_data.reset_index(names="顺序"))
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("订单数:Q", title="订单数"),
                    y=alt.Y("阶段:N", title="", sort=alt.SortField(field="顺序", order="ascending")),
                    color=alt.Color("阶段:N", title="阶段", legend=None),
                    tooltip=[
                        "阶段:N",
                        alt.Tooltip("订单数:Q", format=".0f"),
                        alt.Tooltip("占全部订单:Q", format=".2%"),
                        "说明:N",
                    ],
                )
            )
            st.altair_chart(funnel_chart, use_container_width=True)
    with reason_col:
        if not reason_data.empty:
            st.markdown("###### 拒绝原因分布")
            reason_chart = (
                alt.Chart(reason_data)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("订单数:Q", title="订单数"),
                    y=alt.Y("拒绝原因:N", title="", sort="-x"),
                    color=alt.Color("拒绝原因:N", title="拒绝原因", legend=None),
                    tooltip=[
                        "拒绝原因:N",
                        "原因代码:N",
                        alt.Tooltip("订单数:Q", format=".0f"),
                        alt.Tooltip("占拒绝订单:Q", format=".2%"),
                    ],
                )
            )
            st.altair_chart(reason_chart, use_container_width=True)


def _render_diagnostic_status_chart(report: pd.DataFrame) -> None:
    chart_data = _diagnostic_status_chart_frame(report)
    if chart_data.empty:
        return
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("检查项数:Q", title="检查项数"),
            y=alt.Y("状态:N", title="", sort=alt.SortField(field="排序", order="ascending")),
            color=alt.Color(
                "状态:N",
                title="状态",
                scale=alt.Scale(domain=["失败", "关注", "通过"], range=["#dc2626", "#d97706", "#059669"]),
            ),
            tooltip=["状态:N", alt.Tooltip("检查项数:Q", format=".0f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _diagnostic_status_chart_frame(report: pd.DataFrame) -> pd.DataFrame:
    """汇总实验诊断状态，先展示失败和关注项。"""
    columns = ["状态", "检查项数", "排序"]
    if report.empty or "status" not in report.columns:
        return pd.DataFrame(columns=columns)
    order = {"失败": 0, "关注": 1, "通过": 2}
    status = report["status"].fillna("").astype(str).replace("", "通过")
    grouped = status.value_counts(sort=False).rename_axis("状态").reset_index(name="检查项数")
    grouped["排序"] = grouped["状态"].map(lambda value: order.get(str(value), 99))
    return grouped.sort_values(["排序", "状态"], kind="mergesort").loc[:, columns].reset_index(drop=True)


def _order_decision_funnel_frame(order_decisions: pd.DataFrame) -> pd.DataFrame:
    """把订单决策转成回测漏斗；只依赖标准决策字段，不绑定具体策略。"""
    columns = ["阶段", "订单数", "占全部订单", "说明"]
    if order_decisions.empty or "status" not in order_decisions.columns:
        return pd.DataFrame(columns=columns)
    total = int(len(order_decisions))
    if total <= 0:
        return pd.DataFrame(columns=columns)
    status = order_decisions["status"].fillna("").astype(str)
    actual_entry = pd.to_numeric(
        order_decisions.get("actual_entry_price", pd.Series([0.0] * total, index=order_decisions.index)),
        errors="coerce",
    ).fillna(0.0)
    rows = [
        ("全部订单", total, "策略层生成并进入撮合层的候选订单。"),
        ("触发入场价", int(actual_entry.gt(0).sum()), "价格到达挂单价，后续仍可能被风控、容量或资金拒绝。"),
        ("最终成交", int(status.eq("accepted").sum()), "通过撮合和风控，实际进入持仓。"),
        ("未成交/被拒", int(status.eq("rejected").sum()), "未触发挂单价，或被风控、容量、资金等规则拒绝。"),
    ]
    return pd.DataFrame(
        [{"阶段": stage, "订单数": count, "占全部订单": count / total, "说明": note} for stage, count, note in rows],
        columns=columns,
    )


def _order_reject_reason_chart_frame(order_decisions: pd.DataFrame) -> pd.DataFrame:
    """汇总订单拒绝原因；保留原因代码，展示字段使用中文说明。"""
    columns = ["拒绝原因", "订单数", "占拒绝订单", "原因代码"]
    if order_decisions.empty or "status" not in order_decisions.columns:
        return pd.DataFrame(columns=columns)
    status = order_decisions["status"].fillna("").astype(str)
    rejected = order_decisions.loc[status.eq("rejected")].copy()
    if rejected.empty:
        return pd.DataFrame(columns=columns)
    reason = rejected.get("reason", pd.Series([""] * len(rejected), index=rejected.index)).fillna("").astype(str)
    reason = reason.replace("", "unknown")
    total = int(len(rejected))
    grouped = reason.value_counts(sort=False).rename_axis("原因代码").reset_index(name="订单数")
    grouped["拒绝原因"] = grouped["原因代码"].map(lambda value: DISPLAY_VALUE_MAP.get("reason", {}).get(value, value))
    grouped["占拒绝订单"] = grouped["订单数"].map(lambda count: float(count) / total)
    grouped["排序"] = grouped["原因代码"].map(_reject_reason_sort_priority)
    return (
        grouped.sort_values(["订单数", "排序", "拒绝原因"], ascending=[False, True, True], kind="mergesort")
        .loc[:, columns]
        .reset_index(drop=True)
    )


def _reject_reason_sort_priority(reason: object) -> int:
    priority = {
        "no_fill": 10,
        "no_liquidity": 20,
        "no_bars": 30,
        "max_open_positions": 40,
        "same_symbol_overlap": 50,
        "no_capital": 60,
        "capital_limit": 70,
        "sector_limit": 80,
        "actual_risk_too_high": 90,
        "risk_too_large": 90,
        "chase_too_far": 100,
        "chase_too_large": 100,
        "target_not_favorable": 110,
        "invalid_order": 120,
        "duplicate_order_id": 130,
        "already_open": 140,
        "unknown": 999,
    }
    return priority.get(str(reason), 500)


def _performance_summary_frame(stats: Mapping[str, object]) -> pd.DataFrame:
    """把核心回测指标分组展示，避免用户在 stats.json 字段里逐项查找。"""
    columns = ["模块", "指标", "数值", "说明"]
    items = (
        ("收益", "total_return", "首尾净值累计收益。"),
        ("收益", "annualized_return", "按净值时间轴估算的复合年化收益。"),
        ("风险", "max_drawdown", "按逐 K 组合持仓市值估算的最大回撤；有 drawdown_net_value 时优先使用该口径。"),
        ("风险", "current_drawdown", "最新组合回撤估算净值相对历史高点的回撤。"),
        ("交易质量", "win_rate", "盈利交易占全部交易的比例。"),
        ("交易质量", "profit_factor", "总盈利除以总亏损，越高越好。"),
        ("交易质量", "avg_r_multiple", "平均每笔交易相对初始风险的 R 倍数。"),
        ("交易质量", "system_quality_number", "R 倍数均值、波动和样本数的综合质量。"),
        ("资金效率", "exposure_bar_ratio", "持仓 K 数占市场 K 数的比例。"),
        ("资金效率", "avg_cash_ratio", "组合逐 K 净值中的平均现金比例。"),
    )
    rows: list[dict[str, str]] = []
    for module, key, note in items:
        if key not in stats or _is_missing(stats[key]):
            continue
        rows.append(
            {
                "模块": module,
                "指标": _display_column_label(key),
                "数值": _format_display_value(key, stats[key]),
                "说明": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _display_column_label(column: str) -> str:
    if column in DISPLAY_COLUMN_LABELS:
        return DISPLAY_COLUMN_LABELS[column]
    label = column
    for raw, translated in [
        ("monthly_", "月度"),
        ("avg_", "平均"),
        ("max_", "最大"),
        ("min_", "最小"),
        ("p05", "5%分位"),
        ("p25", "25%分位"),
        ("p50", "中位数"),
        ("p75", "75%分位"),
        ("p95", "95%分位"),
        ("return", "收益"),
        ("drawdown", "回撤"),
        ("count", "次数"),
        ("rate", "比例"),
        ("ratio", "比例"),
        ("bars", "K数"),
        ("price", "价格"),
        ("capital", "资金"),
        ("margin", "保证金"),
    ]:
        label = label.replace(raw, translated)
    return label.replace("_", "")


def _unique_display_label(label: str, used_labels: set[str]) -> str:
    if label not in used_labels:
        used_labels.add(label)
        return label
    index = 2
    while f"{label}{index}" in used_labels:
        index += 1
    unique = f"{label}{index}"
    used_labels.add(unique)
    return unique


def _format_display_value(column: str, value: object, *, stock_names: Mapping[str, str] | None = None) -> str:
    if _is_missing(value):
        return ""
    if column in {"stock_code", "symbol"}:
        return _stock_name_label(value, stock_names=stock_names)
    if isinstance(value, bool):
        return "是" if value else "否"
    mapped = DISPLAY_VALUE_MAP.get(column, {}).get(str(value))
    if mapped is not None:
        return mapped
    if _is_date_column(column):
        formatted_date = _format_date_value(value)
        if formatted_date is not None:
            return formatted_date
    numeric = _coerce_float(value)
    if numeric is None:
        return str(value)
    if _is_percent_column(column):
        return _format_percent_value(column, numeric)
    if _is_integer_column(column):
        return f"{numeric:.0f}"
    if column in {"planned_entry_price", "actual_entry_price", "entry_price", "stop_price", "target_price", "exit_price"}:
        return f"{numeric:.4f}"
    if column in {"net_value", "drawdown_net_value", "start_net_value", "end_net_value"}:
        return f"{numeric:.4f}"
    if math.isinf(numeric):
        return "∞" if numeric > 0 else "-∞"
    return f"{numeric:.2f}"


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _coerce_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _is_date_column(column: str) -> bool:
    return (
        column
        in {
            "date",
            "period",
            "start",
            "end",
            "signal_date",
            "entry_date",
            "exit_date",
            "max_drawdown_start_at",
            "max_drawdown_trough_at",
            "max_drawdown_recovery_at",
        }
        or column.endswith("_date")
    )


def _format_date_value(value: object) -> str | None:
    if isinstance(value, pd.Period):
        return str(value)
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if timestamp.hour or timestamp.minute or timestamp.second:
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return timestamp.strftime("%Y-%m-%d")


def _is_percent_column(column: str) -> bool:
    name = column.lower()
    if name.endswith("_pct") or name.endswith("_rate") or name.endswith("_return") or name.endswith("_drawdown"):
        return True
    return name in {
        "return",
        "win_rate",
        "win_rate_ci_lower",
        "win_rate_ci_upper",
        "avg_return_standard_error",
        "avg_return_ci_lower",
        "avg_return_ci_upper",
        "positive_expectancy_probability",
        "acceptance_rate",
        "rejection_rate",
        "decision_rate",
        "group_decision_rate",
        "strategy_filter_acceptance_rate",
        "strategy_filter_rejection_rate",
        "total_return",
        "avg_return",
        "max_drawdown",
        "avg_drawdown",
        "best_trade",
        "worst_trade",
        "return_std",
        "best_return",
        "worst_return",
        "underwater_ratio",
        "coverage_ratio",
        "depth",
        "avg_cash_ratio",
        "min_cash_ratio",
        "max_cash_ratio",
        "avg_capital_fraction",
        "max_capital_fraction",
        "avg_margin_fraction",
        "max_margin_fraction",
        "margin_exposure",
        "avg_margin_exposure",
        "max_margin_exposure",
        "capital_fraction",
        "risk_fraction",
        "margin_fraction",
        "capital_turnover",
        "margin_turnover",
        "avg_gross_exposure",
        "max_gross_exposure",
        "avg_net_exposure",
        "min_net_exposure",
        "max_net_exposure",
        "exposure_bar_ratio",
        "time_under_water_ratio",
        "annualized_return",
        "annualized_volatility",
    }


def _format_percent_value(column: str, numeric: float) -> str:
    if math.isinf(numeric):
        return "∞" if numeric > 0 else "-∞"
    percent = numeric if column.lower() in PERCENT_POINT_COLUMNS else numeric * 100
    return f"{percent:.2f}%"


def _is_integer_column(column: str) -> bool:
    name = column.lower()
    if name.endswith("_id") or name.endswith("_count"):
        return True
    return name in {
        "trade_no",
        "episode_rank",
        "risk_adjusted_rank",
        "best_risk_adjusted_sweep_rank",
        "episode_no",
        "underwater_bars",
        "recovery_bars",
        "order_id",
        "event_id",
        "trade_count",
        "count",
        "positive_count",
        "negative_count",
        "decision_count",
        "group_decision_count",
        "holding_bars",
        "exposure_bars",
        "max_holding_bars",
        "signal_bar_index",
        "observation_count",
        "max_consecutive_wins",
        "max_consecutive_losses",
        "max_consecutive_gains",
        "monthly_max_consecutive_losses",
        "monthly_max_recovery_periods",
        "monthly_current_underwater_periods",
        "current_underwater_bars",
        "filtered_limit_open_count",
        "rejected_no_fill_count",
    }


def _equity_y_domain(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 1.0, 1.02
    lower_value = min(1.0, float(numeric.min()))
    upper_value = max(1.0, float(numeric.max()))
    spread = max(upper_value - lower_value, 0.02)
    lower = 1.0 if lower_value >= 1.0 else lower_value - spread * 0.08
    upper = upper_value + spread * 0.08
    if upper <= lower:
        upper = lower + 0.02
    return float(lower), float(upper)


def _render_equity_chart(equity_curve: pd.DataFrame) -> None:
    chart_data = _equity_chart_frame(equity_curve)
    if chart_data.empty:
        return
    x_label = "时间" if "时间" in chart_data.columns else "交易序号"
    lower, upper = _equity_y_domain(chart_data["净值比例"])
    line = (
        alt.Chart(chart_data)
        .mark_line(color="#0f766e", strokeWidth=2)
        .encode(
            x=alt.X(f"{x_label}:T" if x_label == "时间" else f"{x_label}:Q", title=x_label),
            y=alt.Y("净值比例:Q", title="净值比例", scale=alt.Scale(domain=[lower, upper]), axis=alt.Axis(format=".2f")),
            tooltip=[x_label, alt.Tooltip("净值比例:Q", format=".4f")],
        )
    )
    baseline = (
        alt.Chart(pd.DataFrame({"净值比例": [1.0]}))
        .mark_rule(color="#64748b", strokeDash=[4, 4])
        .encode(y="净值比例:Q")
    )
    st.altair_chart(line + baseline, use_container_width=True)


def _render_equity_drawdown_chart(equity_curve: pd.DataFrame) -> None:
    chart_data = _equity_drawdown_chart_frame(equity_curve)
    if chart_data.empty:
        return
    x_label = "时间" if "时间" in chart_data.columns else "交易序号"
    lower = min(-0.02, float(chart_data["回撤"].min()) * 1.08)
    area = (
        alt.Chart(chart_data)
        .mark_area(color="#dc2626", opacity=0.24)
        .encode(
            x=alt.X(f"{x_label}:T" if x_label == "时间" else f"{x_label}:Q", title=x_label),
            y=alt.Y(
                "回撤:Q",
                title="回撤",
                scale=alt.Scale(domain=[lower, 0.0]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[x_label, alt.Tooltip("回撤:Q", format=".2%")],
        )
    )
    baseline = alt.Chart(pd.DataFrame({"回撤": [0.0]})).mark_rule(color="#64748b").encode(y="回撤:Q")
    st.altair_chart(area + baseline, use_container_width=True)


def _equity_drawdown_episodes_frame(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """生成最大回撤区间明细；回撤口径与回撤曲线保持一致。"""
    if equity_curve.empty or "net_value" not in equity_curve.columns:
        return pd.DataFrame()
    drawdown_data, drawdown_value = price_path_drawdown_inputs(equity_curve, equity_curve["net_value"])
    return drawdown_episodes(drawdown_data, drawdown_value, limit=10)


def _render_trade_path_distribution_chart(frame: pd.DataFrame) -> None:
    chart_data = _trade_path_distribution_chart_frame(frame)
    if chart_data.empty:
        return
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("交易次数:Q", title="交易次数"),
            y=alt.Y("区间:N", title="", sort=alt.SortField(field="区间顺序", order="ascending")),
            color=alt.Color("分布维度:N", title="分布维度"),
            row=alt.Row("分布维度:N", title="", header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=[
                "分布维度:N",
                "区间:N",
                alt.Tooltip("交易次数:Q", format=".0f"),
                alt.Tooltip("胜率:Q", format=".1%"),
                alt.Tooltip("平均收益:Q", format=".2%"),
            ],
        )
        .properties(height=96)
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart, use_container_width=True)


def _trade_path_distribution_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """把交易路径分布表转为中文图表字段，页面和测试共用同一口径。"""
    required = {"dimension", "bucket", "bucket_order", "trade_count", "win_rate", "avg_return"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["分布维度", "区间", "区间顺序", "交易次数", "胜率", "平均收益"])
    chart_data = frame.loc[:, ["dimension", "bucket", "bucket_order", "trade_count", "win_rate", "avg_return"]].copy()
    for column in ("bucket_order", "trade_count", "win_rate", "avg_return"):
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(subset=["bucket_order", "trade_count"])
    return chart_data.rename(
        columns={
            "dimension": "分布维度",
            "bucket": "区间",
            "bucket_order": "区间顺序",
            "trade_count": "交易次数",
            "win_rate": "胜率",
            "avg_return": "平均收益",
        }
    ).reset_index(drop=True)


def _equity_chart_frame(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """把净值曲线转换成从 1.0 起步的比例曲线，避免初始资金大小影响视觉比例。"""
    if equity_curve.empty or "net_value" not in equity_curve.columns:
        return pd.DataFrame()
    x_column = "date" if "date" in equity_curve.columns else "trade_no"
    chart_data = equity_curve[[x_column, "net_value"]].copy()
    chart_data["net_value"] = pd.to_numeric(chart_data["net_value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["net_value"])
    if chart_data.empty:
        return pd.DataFrame()
    start_value = float(chart_data["net_value"].iloc[0])
    if start_value <= 0:
        return pd.DataFrame()
    chart_data["净值比例"] = chart_data["net_value"] / start_value
    if x_column == "date":
        chart_data[x_column] = pd.to_datetime(chart_data[x_column], errors="coerce")
        chart_data = chart_data.dropna(subset=[x_column])
    x_label = "时间" if x_column == "date" else "交易序号"
    return chart_data.rename(columns={x_column: x_label})[[x_label, "净值比例"]].reset_index(drop=True)


def _equity_drawdown_chart_frame(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """从净值曲线生成水下回撤序列，0 表示历史新高，负值表示回撤。"""
    if equity_curve.empty or "net_value" not in equity_curve.columns:
        return pd.DataFrame()
    x_column = "date" if "date" in equity_curve.columns else "trade_no"
    drawdown_data, drawdown_value = price_path_drawdown_inputs(equity_curve, equity_curve["net_value"])
    chart_data = drawdown_data[[x_column]].copy()
    chart_data["_drawdown_value"] = pd.to_numeric(drawdown_value, errors="coerce").reset_index(drop=True)
    chart_data = chart_data.dropna(subset=["_drawdown_value"])
    if chart_data.empty:
        return pd.DataFrame()
    running_high = chart_data["_drawdown_value"].cummax()
    chart_data["回撤"] = chart_data["_drawdown_value"] / running_high - 1.0
    if x_column == "date":
        chart_data[x_column] = pd.to_datetime(chart_data[x_column], errors="coerce")
        chart_data = chart_data.dropna(subset=[x_column])
    x_label = "时间" if x_column == "date" else "交易序号"
    return chart_data.rename(columns={x_column: x_label})[[x_label, "回撤"]].reset_index(drop=True)


def _strategy_kline_symbol_options(bars: pd.DataFrame, trades: pd.DataFrame) -> list[str]:
    """K 线图股票列表：有交易的股票优先，其余按行情出现顺序补齐。"""
    normalized = normalize_bars(bars)
    if normalized.empty:
        return []
    bar_symbols = [str(symbol) for symbol in normalized["stock_code"].drop_duplicates().tolist()]
    trade_symbols: list[str] = []
    if not trades.empty and "stock_code" in trades.columns:
        for raw_symbol in trades["stock_code"].dropna().tolist():
            symbol = normalize_symbol(raw_symbol)
            if symbol and symbol not in trade_symbols and symbol in bar_symbols:
                trade_symbols.append(symbol)
    return [*trade_symbols, *[symbol for symbol in bar_symbols if symbol not in trade_symbols]]


def _strategy_kline_chart_frame(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把单只股票的完整回测窗口 K 线转成图表字段，不因交易区间裁剪样本。"""
    columns = ["K序号", "时间", "股票代码", "开盘", "最高", "最低", "收盘", "涨跌"]
    normalized = normalize_bars(bars)
    normalized_symbol = normalize_symbol(symbol)
    if normalized.empty or not normalized_symbol:
        return pd.DataFrame(columns=columns)
    frame = normalized.loc[normalized["stock_code"].eq(normalized_symbol)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.sort_values("date")
    chart = frame.rename(
        columns={
            "date": "时间",
            "stock_code": "股票代码",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
        }
    )[["时间", "股票代码", "开盘", "最高", "最低", "收盘"]]
    chart.insert(0, "K序号", range(len(chart)))
    chart["涨跌"] = chart["收盘"].ge(chart["开盘"]).map({True: "上涨", False: "下跌"})
    return chart[columns].reset_index(drop=True)


def _kline_index_for_time(chart_data: pd.DataFrame, value: object) -> int | None:
    """把真实时间映射到连续 K 序号；非精确时间落到不晚于该时间的最近 K。"""
    if chart_data.empty or not {"时间", "K序号"}.issubset(chart_data.columns):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    lookup = pd.DataFrame(
        {
            "时间": pd.to_datetime(chart_data["时间"], errors="coerce"),
            "K序号": pd.to_numeric(chart_data["K序号"], errors="coerce"),
        }
    ).dropna()
    if lookup.empty:
        return None
    lookup = lookup.sort_values("时间")
    exact = lookup.loc[lookup["时间"].eq(timestamp), "K序号"]
    if not exact.empty:
        return int(exact.iloc[-1])
    prior = lookup.loc[lookup["时间"].le(timestamp), "K序号"]
    if prior.empty:
        return None
    return int(prior.iloc[-1])


def _trade_direction_label(side: str) -> str:
    return "空头" if side == "short" else "多头"


def _trade_entry_label(side: str) -> str:
    return "开空" if side == "short" else "开多"


def _trade_exit_label(side: str, exit_reason: str) -> str:
    if exit_reason == "stop_loss":
        return "止损"
    if exit_reason == "trailing_take_profit":
        return "回撤止盈"
    return "平空" if side == "short" else "平多"


def _trade_entry_reason(row: Mapping[str, object]) -> str:
    event_type = str(row.get("event_type", "")).strip()
    if not event_type:
        return "开仓"
    return DISPLAY_VALUE_MAP.get("event_type", {}).get(event_type, event_type)


def _trade_exit_reason(row: Mapping[str, object]) -> str:
    exit_reason = str(row.get("exit_reason", "")).strip()
    if not exit_reason:
        return "平仓"
    return DISPLAY_VALUE_MAP.get("exit_reason", {}).get(exit_reason, exit_reason)


def _marker_time_text(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _strategy_trade_marker_frame(trades: pd.DataFrame, symbol: str, chart_data: pd.DataFrame) -> pd.DataFrame:
    """把成交表转成连续 K 线图上的开仓、平仓和止损标注。"""
    columns = ["K序号", "时间", "开仓/平仓时间", "价格", "标注", "方向", "开仓/平仓原因"]
    normalized_symbol = normalize_symbol(symbol)
    required = {"stock_code", "side", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason"}
    if trades.empty or chart_data.empty or not normalized_symbol or not required.issubset(trades.columns):
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    trade_rows = trades.loc[trades["stock_code"].map(normalize_symbol).eq(normalized_symbol)]
    for row in trade_rows.to_dict("records"):
        side = str(row.get("side", "")).lower()
        direction = _trade_direction_label(side)
        entry_date = pd.to_datetime(row.get("entry_date"), errors="coerce")
        entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
        entry_index = _kline_index_for_time(chart_data, entry_date)
        if entry_index is not None and pd.notna(entry_date) and pd.notna(entry_price):
            records.append(
                {
                    "K序号": entry_index,
                    "时间": entry_date,
                    "开仓/平仓时间": _marker_time_text(entry_date),
                    "价格": float(entry_price),
                    "标注": _trade_entry_label(side),
                    "方向": direction,
                    "开仓/平仓原因": _trade_entry_reason(row),
                    "_priority": 1,
                }
            )
        exit_date = pd.to_datetime(row.get("exit_date"), errors="coerce")
        exit_price = pd.to_numeric(row.get("exit_price"), errors="coerce")
        exit_index = _kline_index_for_time(chart_data, exit_date)
        if exit_index is not None and pd.notna(exit_date) and pd.notna(exit_price):
            records.append(
                {
                    "K序号": exit_index,
                    "时间": exit_date,
                    "开仓/平仓时间": _marker_time_text(exit_date),
                    "价格": float(exit_price),
                    "标注": _trade_exit_label(side, str(row.get("exit_reason", ""))),
                    "方向": direction,
                    "开仓/平仓原因": _trade_exit_reason(row),
                    "_priority": 0,
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records).sort_values(["K序号", "_priority", "价格"]).drop(columns=["_priority"])
    return frame[columns].reset_index(drop=True)


def _strategy_stop_segment_frame(trades: pd.DataFrame, symbol: str, chart_data: pd.DataFrame) -> pd.DataFrame:
    """每笔交易的止损价横线，从入场延伸到退出，辅助检查策略风险边界。"""
    columns = ["开始K序号", "结束K序号", "开始时间", "结束时间", "止损价", "方向", "开仓原因", "平仓原因"]
    normalized_symbol = normalize_symbol(symbol)
    required = {"stock_code", "side", "entry_date", "exit_date", "stop_price"}
    if trades.empty or chart_data.empty or not normalized_symbol or not required.issubset(trades.columns):
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    trade_rows = trades.loc[trades["stock_code"].map(normalize_symbol).eq(normalized_symbol)]
    for row in trade_rows.to_dict("records"):
        start = pd.to_datetime(row.get("entry_date"), errors="coerce")
        end = pd.to_datetime(row.get("exit_date"), errors="coerce")
        stop_price = pd.to_numeric(row.get("stop_price"), errors="coerce")
        start_index = _kline_index_for_time(chart_data, start)
        end_index = _kline_index_for_time(chart_data, end)
        if start_index is None or end_index is None or pd.isna(start) or pd.isna(end) or pd.isna(stop_price):
            continue
        side = str(row.get("side", "")).lower()
        records.append(
            {
                "开始K序号": start_index,
                "结束K序号": max(start_index, end_index),
                "开始时间": start,
                "结束时间": end,
                "止损价": float(stop_price),
                "方向": _trade_direction_label(side),
                "开仓原因": _trade_entry_reason(row),
                "平仓原因": _trade_exit_reason(row),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).sort_values(["开始K序号", "结束K序号"]).reset_index(drop=True)


def _strategy_holding_interval_frame(trades: pd.DataFrame, symbol: str, chart_data: pd.DataFrame) -> pd.DataFrame:
    """持仓区间阴影层：只展示实际开仓到实际平仓的完整占用窗口。"""
    columns = [
        "开始K序号",
        "结束K序号",
        "开始绘图K序号",
        "结束绘图K序号",
        "开始时间",
        "结束时间",
        "区间低价",
        "区间高价",
        "方向",
        "开仓原因",
        "平仓原因",
    ]
    normalized_symbol = normalize_symbol(symbol)
    required = {"stock_code", "side", "entry_date", "exit_date"}
    if trades.empty or chart_data.empty or not normalized_symbol or not required.issubset(trades.columns):
        return pd.DataFrame(columns=columns)

    low = pd.to_numeric(chart_data["最低"], errors="coerce").min()
    high = pd.to_numeric(chart_data["最高"], errors="coerce").max()
    if pd.isna(low) or pd.isna(high):
        return pd.DataFrame(columns=columns)
    padding = max((float(high) - float(low)) * 0.04, abs(float(high)) * 0.002, 0.01)
    y_low = float(low) - padding
    y_high = float(high) + padding

    records: list[dict[str, object]] = []
    trade_rows = trades.loc[trades["stock_code"].map(normalize_symbol).eq(normalized_symbol)]
    for row in trade_rows.to_dict("records"):
        start = pd.to_datetime(row.get("entry_date"), errors="coerce")
        end = pd.to_datetime(row.get("exit_date"), errors="coerce")
        start_index = _kline_index_for_time(chart_data, start)
        end_index = _kline_index_for_time(chart_data, end)
        if start_index is None or end_index is None or pd.isna(start) or pd.isna(end):
            continue
        end_index = max(start_index, end_index)
        side = str(row.get("side", "")).lower()
        records.append(
            {
                "开始K序号": start_index,
                "结束K序号": end_index,
                "开始绘图K序号": start_index - 0.45,
                "结束绘图K序号": end_index + 0.45,
                "开始时间": start,
                "结束时间": end,
                "区间低价": y_low,
                "区间高价": y_high,
                "方向": _trade_direction_label(side),
                "开仓原因": _trade_entry_reason(row),
                "平仓原因": _trade_exit_reason(row),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).sort_values(["开始K序号", "结束K序号"]).reset_index(drop=True)


def _build_strategy_kline_altair_chart(
    chart_data: pd.DataFrame,
    markers: pd.DataFrame,
    stops: pd.DataFrame,
    intervals: pd.DataFrame,
) -> alt.LayerChart | None:
    if chart_data.empty:
        return None

    candle_size = max(2.0, min(8.0, 560.0 / max(len(chart_data), 1)))
    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"], name="kline_zoom")
    base = alt.Chart(chart_data).encode(
        x=alt.X(
            "K序号:Q",
            title="K线序号（连续压缩）",
            axis=alt.Axis(labelAngle=0, tickCount=8),
        ),
        tooltip=[
            alt.Tooltip("时间:T", title="时间"),
            alt.Tooltip("开盘:Q", format=".3f"),
            alt.Tooltip("最高:Q", format=".3f"),
            alt.Tooltip("最低:Q", format=".3f"),
            alt.Tooltip("收盘:Q", format=".3f"),
        ],
    )
    wick = base.mark_rule(color="#475569", opacity=0.75, clip=True).encode(
        y=alt.Y("最低:Q", title="价格", scale=alt.Scale(zero=False)),
        y2="最高:Q",
    )
    body = base.mark_bar(size=candle_size, clip=True).encode(
        y=alt.Y("开盘:Q", title="价格", scale=alt.Scale(zero=False)),
        y2="收盘:Q",
        color=alt.Color(
            "涨跌:N",
            title="K线",
            scale=alt.Scale(domain=["上涨", "下跌"], range=["#dc2626", "#16a34a"]),
        ),
    )
    layers: list[alt.Chart] = [wick]
    if not intervals.empty:
        layers.append(
            alt.Chart(intervals)
            .mark_rect(opacity=0.12, clip=True)
            .encode(
                x=alt.X("开始绘图K序号:Q", title="K线序号（连续压缩）"),
                x2="结束绘图K序号:Q",
                y=alt.Y("区间低价:Q", title="价格", scale=alt.Scale(zero=False)),
                y2="区间高价:Q",
                color=alt.Color(
                    "方向:N",
                    title="持仓区间",
                    scale=alt.Scale(domain=["多头", "空头"], range=["#ef4444", "#2563eb"]),
                ),
                tooltip=[
                    alt.Tooltip("方向:N", title="方向"),
                    alt.Tooltip("开始时间:T", title="开仓时间"),
                    alt.Tooltip("结束时间:T", title="平仓时间"),
                    alt.Tooltip("开仓原因:N", title="开仓原因"),
                    alt.Tooltip("平仓原因:N", title="平仓原因"),
                ],
            )
        )
    layers.append(body)

    if not stops.empty:
        layers.append(
            alt.Chart(stops)
            .mark_rule(color="#ea580c", strokeDash=[5, 4], strokeWidth=1.5, clip=True)
            .encode(
                x=alt.X("开始K序号:Q", title="K线序号（连续压缩）"),
                x2="结束K序号:Q",
                y=alt.Y("止损价:Q", title="价格", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("方向:N", title="方向"),
                    alt.Tooltip("止损价:Q", title="止损价", format=".3f"),
                    alt.Tooltip("开始时间:T", title="开仓时间"),
                    alt.Tooltip("结束时间:T", title="平仓时间"),
                    alt.Tooltip("开仓原因:N", title="开仓原因"),
                    alt.Tooltip("平仓原因:N", title="平仓原因"),
                ],
            )
        )

    if not markers.empty:
        marker_base = alt.Chart(markers).encode(
            x=alt.X("K序号:Q", title="K线序号（连续压缩）"),
            y=alt.Y("价格:Q", title="价格", scale=alt.Scale(zero=False)),
            shape=alt.Shape(
                "标注:N",
                title="标注",
                scale=alt.Scale(
                    domain=["开多", "开空", "止损", "回撤止盈", "平多", "平空"],
                    range=["triangle-up", "triangle-down", "cross", "diamond", "circle", "circle"],
                ),
            ),
            color=alt.Color(
                "标注:N",
                title="标注",
                scale=alt.Scale(
                    domain=["开多", "开空", "止损", "回撤止盈", "平多", "平空"],
                    range=["#dc2626", "#2563eb", "#ea580c", "#f59e0b", "#991b1b", "#1d4ed8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("标注:N"),
                alt.Tooltip("价格:Q", format=".3f"),
                alt.Tooltip("开仓/平仓时间:N"),
                alt.Tooltip("开仓/平仓原因:N"),
            ],
        )
        layers.append(
            alt.Chart(markers)
            .transform_filter(alt.FieldOneOfPredicate(field="标注", oneOf=["开多", "开空"]))
            .mark_rule(color="#0f172a", opacity=0.18, strokeDash=[2, 4], clip=True)
            .encode(x=alt.X("K序号:Q", title="K线序号（连续压缩）"))
        )
        layers.append(
            alt.Chart(markers)
            .transform_filter(alt.FieldOneOfPredicate(field="标注", oneOf=["止损", "回撤止盈", "平多", "平空"]))
            .mark_rule(color="#0f172a", opacity=0.1, strokeDash=[1, 5], clip=True)
            .encode(x=alt.X("K序号:Q", title="K线序号（连续压缩）"))
        )
        layers.append(marker_base.mark_point(filled=True, size=105, stroke="#111827", strokeWidth=0.6, clip=True))
        layers.append(
            alt.Chart(markers).mark_text(dy=-15, fontSize=11, fontWeight="bold", clip=True).encode(
                x=alt.X("K序号:Q", title="K线序号（连续压缩）"),
                y=alt.Y("价格:Q", title="价格", scale=alt.Scale(zero=False)),
                text="标注:N",
                color=alt.Color(
                    "标注:N",
                    title="标注",
                    scale=alt.Scale(
                        domain=["开多", "开空", "止损", "回撤止盈", "平多", "平空"],
                        range=["#dc2626", "#2563eb", "#ea580c", "#f59e0b", "#991b1b", "#1d4ed8"],
                    ),
                ),
            )
        )

    return (
        alt.layer(*layers)
        .resolve_scale(y="shared")
        .properties(
            width="container",
            height=640,
            autosize={"type": "fit-x", "contains": "padding"},
            usermeta={"embedOptions": {"renderer": "svg"}},
        )
        .add_params(zoom)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_view(continuousWidth=1100, continuousHeight=640, strokeWidth=0)
    )


def _render_strategy_kline_chart(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    stock_names: Mapping[str, str] | None = None,
) -> None:
    symbols = _strategy_kline_symbol_options(bars, trades)
    if not symbols:
        return

    st.markdown("##### 策略K线运行区间")
    if len(symbols) > 1:
        selected_symbol = st.selectbox(
            "K线图股票",
            symbols,
            key="backtest_kline_symbol",
            format_func=lambda value: _stock_name_label(value, stock_names),
        )
    else:
        selected_symbol = symbols[0]
        st.caption(f"K线图股票：{_stock_name_label(selected_symbol, stock_names)}")

    chart_data = _strategy_kline_chart_frame(bars, selected_symbol)
    if chart_data.empty:
        return
    markers = _strategy_trade_marker_frame(trades, selected_symbol, chart_data)
    stops = _strategy_stop_segment_frame(trades, selected_symbol, chart_data)
    intervals = _strategy_holding_interval_frame(trades, selected_symbol, chart_data)
    st.caption("横轴按连续 K 线序号压缩，午休、夜间和非交易日不留空；悬停可看真实时间。")
    chart = _build_strategy_kline_altair_chart(chart_data, markers, stops, intervals)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)


def _symbol_input(label: str, default: str = "000001.SZ", *, key: str, help_text: str | None = None) -> list[str]:
    raw = st.text_input(label, default, key=key, help=help_text)
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _date_inputs(key_prefix: str) -> tuple[str, str]:
    today = date.today()
    start = st.date_input(
        "开始日期",
        today - timedelta(days=20),
        key=f"{key_prefix}_start",
        help=BACKTEST_HELP_TEXT["start_date"],
    )
    end = st.date_input("结束日期", today, key=f"{key_prefix}_end", help=BACKTEST_HELP_TEXT["end_date"])
    return str(start), str(end)


def _parse_float_mapping(value: str) -> dict[str, float]:
    return {key: float(raw) for key, raw in _parse_key_value_pairs(value).items()}


def _parse_int_mapping(value: str) -> dict[str, int]:
    return {key: int(raw) for key, raw in _parse_key_value_pairs(value).items()}


def _parse_text_mapping(value: str) -> dict[str, str]:
    return _parse_key_value_pairs(value)


def _parse_key_value_pairs(value: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in value.replace("\n", ",").split(","):
        text = item.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"映射参数必须使用 key=value 格式：{text}")
        key, raw = text.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key or not raw:
            raise ValueError(f"映射参数不能为空：{text}")
        pairs[key] = raw
    return pairs


def _experiment_name(prefix: str, timeframe: str, start: str, end: str) -> str:
    return f"{prefix}-{timeframe}-{start}-{end}".replace("/", "-")


def _experiment_output_controls(prefix: str, default_name: str, *, compact: bool = False) -> tuple[bool, str]:
    if compact:
        save_outputs = st.checkbox(
            "保存实验产物",
            value=False,
            key=f"{prefix}_save_outputs",
            help=BACKTEST_HELP_TEXT["save_outputs"],
        )
        output_root = _directory_picker(
            "输出父目录",
            DEFAULT_OUTPUT_ROOT,
            key=f"{prefix}_output_root",
            disabled=not save_outputs,
            help_text=BACKTEST_HELP_TEXT["output_parent"],
        )
        output_dir = output_root / default_name
        if save_outputs:
            st.caption(f"本次保存到：{_display_path(str(output_dir))}")
        return bool(save_outputs), str(output_dir)

    c1, c2 = st.columns([1, 3])
    with c1:
        save_outputs = st.checkbox(
            "保存实验产物",
            value=False,
            key=f"{prefix}_save_outputs",
            help=BACKTEST_HELP_TEXT["save_outputs"],
        )
    with c2:
        output_root = _directory_picker(
            "输出父目录",
            DEFAULT_OUTPUT_ROOT,
            key=f"{prefix}_output_root",
            disabled=not save_outputs,
            help_text=BACKTEST_HELP_TEXT["output_parent"],
        )
        output_dir = output_root / default_name
        if save_outputs:
            st.caption(f"本次保存到：{_display_path(str(output_dir))}")
    return bool(save_outputs), str(output_dir)


def _detector_parameter_controls(prefix: str, label_prefix: str = "") -> dict[str, float | int]:
    """高级形态识别参数；单策略和组合回测共用，避免 Web 表单和配置层脱节。"""
    with st.expander(f"{label_prefix}高级形态识别参数", expanded=False):
        trend_c1, trend_c2, trend_c3 = st.columns(3)
        with trend_c1:
            trend_strong_close_pos = st.number_input(
                f"{label_prefix}趋势强收盘",
                min_value=0.01,
                max_value=0.99,
                value=0.65,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_trend_strong_close_pos",
                help=BACKTEST_HELP_TEXT["trend_strong_close_pos"],
            )
        with trend_c2:
            trend_min_body_ratio = st.number_input(
                f"{label_prefix}趋势最小实体",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_trend_min_body_ratio",
                help=BACKTEST_HELP_TEXT["trend_min_body_ratio"],
            )
        with trend_c3:
            trend_pullback_lookback = st.number_input(
                f"{label_prefix}趋势回撤窗口",
                min_value=1,
                value=5,
                key=f"{prefix}_trend_pullback_lookback",
                help=BACKTEST_HELP_TEXT["trend_pullback_lookback"],
            )
        range_c1, range_c2, range_c3, range_c4, range_c5 = st.columns(5)
        with range_c1:
            range_middle_low = st.number_input(
                f"{label_prefix}区间中部下沿",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_range_middle_low",
                help=BACKTEST_HELP_TEXT["range_middle_low"],
            )
        with range_c2:
            range_middle_high = st.number_input(
                f"{label_prefix}区间中部上沿",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_range_middle_high",
                help=BACKTEST_HELP_TEXT["range_middle_high"],
            )
        with range_c3:
            range_false_break_buffer = st.number_input(
                f"{label_prefix}区间失败突破缓冲",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key=f"{prefix}_range_false_break_buffer",
                help=BACKTEST_HELP_TEXT["range_false_break_buffer"],
            )
        with range_c4:
            range_strong_close_pos = st.number_input(
                f"{label_prefix}区间强收盘",
                min_value=0.01,
                max_value=0.99,
                value=0.65,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_range_strong_close_pos",
                help=BACKTEST_HELP_TEXT["range_strong_close_pos"],
            )
        with range_c5:
            range_min_score = st.number_input(
                f"{label_prefix}区间最低评分",
                min_value=0.0,
                value=0.8,
                step=0.1,
                format="%.2f",
                key=f"{prefix}_range_min_score",
                help=BACKTEST_HELP_TEXT["range_min_score"],
            )
        channel_c1, channel_c2, channel_c3 = st.columns(3)
        with channel_c1:
            channel_break_buffer = st.number_input(
                f"{label_prefix}通道突破缓冲",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key=f"{prefix}_channel_break_buffer",
                help=BACKTEST_HELP_TEXT["channel_break_buffer"],
            )
        with channel_c2:
            channel_swing_left_bars = st.number_input(
                f"{label_prefix}摆动左侧K数",
                min_value=1,
                value=2,
                key=f"{prefix}_channel_swing_left_bars",
                help=BACKTEST_HELP_TEXT["channel_swing_left_bars"],
            )
        with channel_c3:
            channel_swing_right_bars = st.number_input(
                f"{label_prefix}摆动右侧K数",
                min_value=1,
                value=2,
                key=f"{prefix}_channel_swing_right_bars",
                help=BACKTEST_HELP_TEXT["channel_swing_right_bars"],
            )
        reversal_c1, reversal_c2 = st.columns(2)
        with reversal_c1:
            reversal_strong_close_pos = st.number_input(
                f"{label_prefix}反转强收盘",
                min_value=0.01,
                max_value=0.99,
                value=0.65,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_reversal_strong_close_pos",
                help=BACKTEST_HELP_TEXT["reversal_strong_close_pos"],
            )
        with reversal_c2:
            reversal_min_body_ratio = st.number_input(
                f"{label_prefix}反转最小实体",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_reversal_min_body_ratio",
                help=BACKTEST_HELP_TEXT["reversal_min_body_ratio"],
            )
    return {
        "trend_strong_close_pos": float(trend_strong_close_pos),
        "trend_min_body_ratio": float(trend_min_body_ratio),
        "trend_pullback_lookback": int(trend_pullback_lookback),
        "range_middle_low": float(range_middle_low),
        "range_middle_high": float(range_middle_high),
        "range_false_break_buffer": float(range_false_break_buffer),
        "range_strong_close_pos": float(range_strong_close_pos),
        "range_min_score": float(range_min_score),
        "channel_break_buffer": float(channel_break_buffer),
        "channel_swing_left_bars": int(channel_swing_left_bars),
        "channel_swing_right_bars": int(channel_swing_right_bars),
        "reversal_strong_close_pos": float(reversal_strong_close_pos),
        "reversal_min_body_ratio": float(reversal_min_body_ratio),
    }


def _terminal_false_breakout_controls(prefix: str, label_prefix: str = "") -> TerminalFalseBreakoutInputs:
    """末端假突破过滤参数；只在现代策略表单出现，旧突破回测不使用。"""
    title = f"{label_prefix}末端假突破过滤（可选）"
    with st.expander(title, expanded=False):
        top1, top2, top3 = st.columns([1, 2, 2])
        with top1:
            enabled = st.checkbox(
                f"{label_prefix}启用末端假突破过滤",
                value=False,
                key=f"{prefix}_terminal_false_breakout_enabled",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_enabled"],
            )
        with top2:
            detectors = st.multiselect(
                f"{label_prefix}适用模块",
                ["trend", "channel", "range", "reversal"],
                default=["trend", "channel"],
                key=f"{prefix}_terminal_false_breakout_detectors",
                format_func=lambda value: DISPLAY_VALUE_MAP["detector_name"].get(value, value),
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_detectors"],
                disabled=not enabled,
            )
        with top3:
            st.caption("多/空方向沿用上方交易方向；这里只决定哪些形态模块会被过滤。")
        row2 = st.columns(5)
        with row2[0]:
            lookback = st.number_input(
                f"{label_prefix}通道计算窗口",
                min_value=3,
                value=40,
                key=f"{prefix}_terminal_false_breakout_lookback",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_lookback"],
                disabled=not enabled,
            )
        with row2[1]:
            min_regime_bars = st.number_input(
                f"{label_prefix}持续K数",
                min_value=1,
                value=18,
                key=f"{prefix}_terminal_false_breakout_min_regime_bars",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_min_regime_bars"],
                disabled=not enabled,
            )
        with row2[2]:
            atr_period = st.number_input(
                f"{label_prefix}ATR周期",
                min_value=1,
                value=14,
                key=f"{prefix}_terminal_false_breakout_atr_period",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_atr_period"],
                disabled=not enabled,
            )
        with row2[3]:
            extension_atr_multiple = st.number_input(
                f"{label_prefix}过度延伸倍数",
                min_value=0.0,
                value=2.0,
                step=0.1,
                format="%.2f",
                key=f"{prefix}_terminal_false_breakout_extension_atr_multiple",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_extension_atr_multiple"],
                disabled=not enabled,
            )
        with row2[4]:
            min_score = st.number_input(
                f"{label_prefix}最低命中分",
                min_value=1,
                max_value=5,
                value=3,
                key=f"{prefix}_terminal_false_breakout_min_score",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_min_score"],
                disabled=not enabled,
            )
        row3 = st.columns(5)
        with row3[0]:
            edge_lookback = st.number_input(
                f"{label_prefix}贴边窗口",
                min_value=1,
                value=8,
                key=f"{prefix}_terminal_false_breakout_edge_lookback",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_edge_lookback"],
                disabled=not enabled,
            )
        with row3[1]:
            edge_pos = st.number_input(
                f"{label_prefix}贴边阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.90,
                step=0.01,
                format="%.2f",
                key=f"{prefix}_terminal_false_breakout_edge_pos",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_edge_pos"],
                disabled=not enabled,
            )
        with row3[2]:
            edge_min_count = st.number_input(
                f"{label_prefix}贴边次数",
                min_value=1,
                value=3,
                key=f"{prefix}_terminal_false_breakout_edge_min_count",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_edge_min_count"],
                disabled=not enabled,
            )
        with row3[3]:
            weak_progress_atr = st.number_input(
                f"{label_prefix}弱突破幅度",
                min_value=0.0,
                value=0.35,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_terminal_false_breakout_weak_progress_atr",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_weak_progress_atr"],
                disabled=not enabled,
            )
        with row3[4]:
            wick_ratio = st.number_input(
                f"{label_prefix}影线比例",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.05,
                format="%.2f",
                key=f"{prefix}_terminal_false_breakout_wick_ratio",
                help=BACKTEST_HELP_TEXT["terminal_false_breakout_wick_ratio"],
                disabled=not enabled,
            )
        st.caption("贴近通道边缘、价格远离中轴、突破推进不足且上影线/下影线明显时，会累计末端风险分。")
    return TerminalFalseBreakoutInputs(
        enabled=bool(enabled),
        detectors=tuple(str(item) for item in detectors),
        lookback=int(lookback),
        atr_period=int(atr_period),
        min_regime_bars=int(min_regime_bars),
        extension_atr_multiple=float(extension_atr_multiple),
        edge_lookback=int(edge_lookback),
        edge_pos=float(edge_pos),
        edge_min_count=int(edge_min_count),
        weak_progress_atr=float(weak_progress_atr),
        wick_ratio=float(wick_ratio),
        min_score=int(min_score),
    )


def _show_saved_experiment_path(output_dir: str, name: str) -> None:
    saved_path = Path(output_dir or f"runs/{name}").expanduser()
    st.caption(f"实验产物已保存：{saved_path}")


def _fetch_panel(data_root: Path, adjust: str, tdx_path: str) -> None:
    st.subheader("TDX K 线落地")
    fetch_cols = st.columns([2, 2, 1, 1, 1])
    with fetch_cols[0]:
        symbols = _symbol_input(
            "标的代码",
            "000001.SZ,600519.SH",
            key="fetch_symbols",
            help_text=BACKTEST_HELP_TEXT["symbols"],
        )
    with fetch_cols[1]:
        timeframes = st.multiselect(
            "周期",
            DATA_TIMEFRAMES,
            default=DATA_TIMEFRAMES,
            key="fetch_timeframes",
            help=BACKTEST_HELP_TEXT["timeframe"],
        )
    with fetch_cols[2]:
        start = str(
            st.date_input(
                "开始日期",
                date.today() - timedelta(days=20),
                key="fetch_start",
                help=BACKTEST_HELP_TEXT["start_date"],
            )
        )
    with fetch_cols[3]:
        end = str(st.date_input("结束日期", date.today(), key="fetch_end", help=BACKTEST_HELP_TEXT["end_date"]))
    with fetch_cols[4]:
        prepare_min_coverage_input = st.number_input(
            "补齐最低覆盖率",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="fetch_prepare_min_coverage",
            help=BACKTEST_HELP_TEXT["min_coverage_ratio"],
        )
    prepare_min_coverage = float(prepare_min_coverage_input) if float(prepare_min_coverage_input) > 0 else None
    if st.button("查看本地缓存库存"):
        repo = MarketDataRepository(data_root, adjust=adjust)
        inventory = repo.inventory(
            timeframes=tuple(timeframes),
            symbols=tuple(symbols) if symbols else None,
        )
        st.dataframe(inventory, use_container_width=True)
    if st.button("生成TDX补齐计划"):
        repo = MarketDataRepository(data_root, adjust=adjust)
        plan = repo.plan_from_tdx(
            symbols=tuple(symbols),
            timeframes=tuple(timeframes),
            start=start,
            end=end,
            min_coverage_ratio=prepare_min_coverage,
        )
        st.dataframe(plan, use_container_width=True)
    if st.button("抓取并写入 parquet", type="primary"):
        summaries: list[pd.DataFrame] = []
        repo = MarketDataRepository(data_root, adjust=adjust)
        for timeframe in timeframes:
            summaries.append(
                repo.update_from_tdx(
                    symbols=tuple(symbols),
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    tqcenter_path=tdx_path,
                )
            )
        st.dataframe(pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(), use_container_width=True)
    if st.button("审计并补齐TDX数据"):
        repo = MarketDataRepository(data_root, adjust=adjust)
        summary = repo.prepare_from_tdx(
            symbols=tuple(symbols),
            timeframes=tuple(timeframes),
            start=start,
            end=end,
            tqcenter_path=tdx_path,
            min_coverage_ratio=prepare_min_coverage,
        )
        st.dataframe(summary, use_container_width=True)


def _load_backtest_bundle(
    data_root: Path,
    adjust: str,
    *,
    symbols: list[str],
    timeframe: str,
    start: str,
    end: str,
    strict_data_quality: bool = True,
    min_coverage_ratio: float | None = None,
) -> BacktestDataBundle:
    return MarketDataRepository(data_root, adjust=adjust).load_backtest_data(
        symbols=tuple(symbols),
        timeframe=timeframe,
        start=start,
        end=end,
        strict_data_quality=strict_data_quality,
        min_coverage_ratio=min_coverage_ratio,
    )


def _strategy_controls(prefix: str) -> StrategyConfig:
    c1, c2, c3 = st.columns(3)
    with c1:
        landmark_lookback = st.number_input(
            "标志K回看",
            min_value=2,
            value=20,
            key=f"{prefix}_landmark_lookback",
            help=BACKTEST_HELP_TEXT["landmark_lookback"],
        )
        landmark_range_multiple = st.number_input(
            "标志K振幅倍数",
            min_value=0.1,
            value=1.8,
            step=0.1,
            key=f"{prefix}_range",
            help=BACKTEST_HELP_TEXT["landmark_range_multiple"],
        )
    with c2:
        channel_lookback = st.number_input(
            "通道回看",
            min_value=3,
            value=40,
            key=f"{prefix}_channel",
            help=BACKTEST_HELP_TEXT["channel_lookback"],
        )
        trigger_volume_multiple = st.number_input(
            "突破量能倍数",
            min_value=0.1,
            value=1.5,
            step=0.1,
            key=f"{prefix}_volume",
            help=BACKTEST_HELP_TEXT["trigger_volume_multiple"],
        )
    with c3:
        close_buffer = st.number_input(
            "突破收盘缓冲",
            min_value=0.0,
            value=0.0,
            step=0.005,
            format="%.3f",
            key=f"{prefix}_buffer",
            help=BACKTEST_HELP_TEXT["close_buffer"],
        )
        require_landmark = st.checkbox(
            "突破必须同时是标志K",
            value=True,
            key=f"{prefix}_require_landmark",
            help=BACKTEST_HELP_TEXT["require_landmark"],
        )
    return StrategyConfig(
        landmark_lookback=int(landmark_lookback),
        landmark_range_multiple=float(landmark_range_multiple),
        channel_lookback=int(channel_lookback),
        trigger_volume_multiple=float(trigger_volume_multiple),
        trigger_close_buffer_pct=float(close_buffer),
        require_landmark_trigger=bool(require_landmark),
    )


def _load_panel_inputs(data_root: Path, prefix: str) -> tuple[list[str], str, str, str]:
    scope_cols = st.columns([1, 2, 1, 1])
    with scope_cols[0]:
        timeframe = st.selectbox(
            "周期",
            INTRADAY_TIMEFRAMES,
            index=2,
            key=f"{prefix}_tf",
            help=BACKTEST_HELP_TEXT["timeframe"],
        )
    available = available_symbols(data_root, timeframe=timeframe)
    default_symbols = ",".join(available[:5]) if available else "000001.SZ"
    with scope_cols[1]:
        symbols = _symbol_input("标的代码", default_symbols, key=f"{prefix}_symbols", help_text=BACKTEST_HELP_TEXT["symbols"])
    with scope_cols[2]:
        start = str(
            st.date_input(
                "开始日期",
                date.today() - timedelta(days=20),
                key=f"{prefix}_start",
                help=BACKTEST_HELP_TEXT["start_date"],
            )
        )
    with scope_cols[3]:
        end = str(st.date_input("结束日期", date.today(), key=f"{prefix}_end", help=BACKTEST_HELP_TEXT["end_date"]))
    return symbols, timeframe, start, end


def _scan_panel(data_root: Path, adjust: str) -> None:
    st.subheader("标志K + 趋势通道 + 突破扫描")
    scan_cols = st.columns([2, 2, 1, 1])
    with scan_cols[0]:
        timeframes = st.multiselect(
            "周期",
            INTRADAY_TIMEFRAMES,
            default=["30m", "60m"],
            key="scan_timeframes",
            help=BACKTEST_HELP_TEXT["timeframe"],
        )
    available = sorted({symbol for timeframe in timeframes for symbol in available_symbols(data_root, timeframe=timeframe)})
    default_symbols = ",".join(available[:5]) if available else "000001.SZ"
    with scan_cols[1]:
        symbols = _symbol_input("标的代码", default_symbols, key="scan_symbols", help_text=BACKTEST_HELP_TEXT["symbols"])
    with scan_cols[2]:
        start = str(
            st.date_input(
                "开始日期",
                date.today() - timedelta(days=20),
                key="scan_start",
                help=BACKTEST_HELP_TEXT["start_date"],
            )
        )
    with scan_cols[3]:
        end = str(st.date_input("结束日期", date.today(), key="scan_end", help=BACKTEST_HELP_TEXT["end_date"]))
    config = _strategy_controls("scan")
    if st.button("运行扫描", type="primary"):
        result = scan_timeframes(
            data_root=data_root,
            timeframes=tuple(timeframes),
            adjust=adjust,
            symbols=tuple(symbols),
            start=start,
            end=end,
            strategy=config,
        )
        trigger_count = int(result.full["breakout_trigger"].sum()) if not result.full.empty else 0
        st.metric("触发数量", trigger_count)
        st.dataframe(result.latest, use_container_width=True)
        if not result.full.empty:
            st.dataframe(result.full.loc[result.full["breakout_trigger"]].tail(100), use_container_width=True)
        if len(timeframes) == 1 and not result.full.empty:
            _chart_close(result.full)


def _backtest_panel(data_root: Path, adjust: str) -> None:
    st.subheader("突破策略回测")
    scope = _backtest_scope_module(data_root)
    risk = _backtest_risk_module(scope.mode)
    quality_col, higher_col = st.columns([1, 1])
    with quality_col:
        quality = _backtest_data_quality_module()
    with higher_col:
        higher = _backtest_higher_timeframe_module(scope.timeframe)

    if scope.mode == "旧突破回测":
        _legacy_backtest_module(data_root, adjust, scope, risk, quality)
        return

    if scope.mode == "单策略回测":
        single_col, single_run_col = st.columns([3, 1])
        with single_col:
            single = _single_strategy_module(scope)
        with single_run_col:
            output = _backtest_output_module("single", single.experiment_name, "运行单策略回测", compact=True)
        if output.run_clicked:
            _execute_single_strategy_experiment(data_root, adjust, scope, risk, quality, higher, single, output)
        return

    portfolio_allocation_col, portfolio_detector_col = st.columns([1, 1])
    with portfolio_allocation_col:
        allocation = _portfolio_allocation_module(scope)
    with portfolio_detector_col:
        detector = _portfolio_detector_module()
    output = _backtest_output_module("pf", allocation.experiment_name, "运行组合回测", title="7. 保存与运行")
    if output.run_clicked:
        _execute_portfolio_strategy_experiment(data_root, adjust, scope, risk, quality, higher, allocation, detector, output)


def _backtest_module_container(title: str, caption: str = ""):
    """回测页统一模块容器，避免控件在一个长表单里堆叠。"""
    container = st.container(border=True)
    with container:
        st.markdown(f"#### {title}")
        if caption:
            st.caption(caption)
    return container


def _backtest_scope_module(data_root: Path) -> BacktestScopeInputs:
    with _backtest_module_container("1. 样本范围", "先确定本次回测的数据窗口和回测模式。"):
        symbols, timeframe, start, end = _load_panel_inputs(data_root, "bt")
        mode = st.radio(
            "回测模式",
            ["旧突破回测", "单策略回测", "组合策略回测"],
            horizontal=True,
            help=BACKTEST_HELP_TEXT["scope_mode"],
        )
    return BacktestScopeInputs(symbols=symbols, timeframe=timeframe, start=start, end=end, mode=str(mode))


def _resolve_trailing_take_profit_controls(
    enabled: bool,
    activation_pct: float,
    drawdown_pct: float,
    ma_period: int,
) -> tuple[float, float, int]:
    """把页面开关转换成撮合参数；关闭时强制归零，避免隐藏参数继续影响回测。"""
    if not enabled:
        return 0.0, 0.0, 0
    return float(activation_pct), float(drawdown_pct), int(ma_period)


def _backtest_risk_module(mode: str) -> BacktestRiskInputs:
    is_legacy = mode == "旧突破回测"
    caption = (
        "固定止盈止损、持有周期、手续费和滑点用于旧突破撮合。"
        if is_legacy
        else "单策略和组合策略用信号K结构止损价、结构止损最大风险、盈亏比目标平仓价和盈利通道回撤止盈控制退出。"
    )
    with _backtest_module_container("2. 基础风控与成本", caption):
        take_profit = 0.06
        stop_loss = 0.03
        if is_legacy:
            c1, c2, c3 = st.columns(3)
            with c1:
                take_profit = st.number_input(
                    "止盈",
                    min_value=0.001,
                    value=0.06,
                    step=0.005,
                    format="%.3f",
                    help=BACKTEST_HELP_TEXT["take_profit"],
                )
            with c2:
                stop_loss = st.number_input(
                    "止损",
                    min_value=0.001,
                    value=0.03,
                    step=0.005,
                    format="%.3f",
                    help=BACKTEST_HELP_TEXT["stop_loss"],
                )
            with c3:
                max_holding = st.number_input("最大持有K数", min_value=1, value=12, help=BACKTEST_HELP_TEXT["max_holding"])
        else:
            holding_col, stop_col = st.columns([1, 2])
            with holding_col:
                max_holding = st.number_input("最大持有K数", min_value=1, value=12, help=BACKTEST_HELP_TEXT["max_holding"])
            with stop_col:
                st.caption(
                    "结构止损价说明：现代策略使用信号K/形态模块输出止损价；可在单策略或组合参数里的结构止损最大风险限制止损距离。"
                )
        enable_trailing_take_profit = st.checkbox(
            "启用盈利通道回撤止盈",
            value=False,
            key="bt_enable_trailing_take_profit",
            help=BACKTEST_HELP_TEXT["enable_trailing_take_profit"],
        )
        trailing1, trailing2, trailing3 = st.columns(3)
        with trailing1:
            trailing_take_profit_activation_pct = st.number_input(
                "盈利通道启动浮盈",
                min_value=0.0,
                max_value=1.0,
                value=0.04,
                step=0.005,
                format="%.3f",
                key="bt_trailing_take_profit_activation_pct",
                disabled=not enable_trailing_take_profit,
                help=BACKTEST_HELP_TEXT["trailing_take_profit_activation_pct"],
            )
        with trailing2:
            trailing_take_profit_drawdown_pct = st.number_input(
                "最大盈利回撤幅度",
                min_value=0.0,
                max_value=0.99,
                value=0.015,
                step=0.005,
                format="%.3f",
                key="bt_trailing_take_profit_drawdown_pct",
                disabled=not enable_trailing_take_profit,
                help=BACKTEST_HELP_TEXT["trailing_take_profit_drawdown_pct"],
            )
        with trailing3:
            trailing_take_profit_ma_period = st.number_input(
                "当前周期均线周期",
                min_value=0,
                value=0,
                step=1,
                key="bt_trailing_take_profit_ma_period",
                disabled=not enable_trailing_take_profit,
                help=BACKTEST_HELP_TEXT["trailing_take_profit_ma_period"],
            )
        cost1, cost2, cost3 = st.columns(3)
        with cost1:
            fee_rate = st.number_input(
                "手续费率",
                min_value=0.0,
                value=0.0,
                step=0.0001,
                format="%.4f",
                key="bt_fee_rate",
                help=BACKTEST_HELP_TEXT["fee_rate"],
            )
        with cost2:
            slippage_bps = st.number_input(
                "滑点bps",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.1f",
                key="bt_slippage_bps",
                help=BACKTEST_HELP_TEXT["slippage_bps"],
            )
        with cost3:
            initial_equity = st.number_input(
                "初始资金",
                min_value=0.0001,
                value=1.0,
                step=0.1,
                format="%.4f",
                key="bt_initial_equity",
                help=BACKTEST_HELP_TEXT["initial_equity"],
            )
        intrabar_exit_policy = st.selectbox(
            "同K止盈止损冲突",
            ["conservative", "optimistic"],
            format_func=lambda value: "保守：止损优先" if value == "conservative" else "乐观：止盈优先",
            key="bt_intrabar_policy",
            help=BACKTEST_HELP_TEXT["intrabar_exit_policy"],
        )
    trailing_activation, trailing_drawdown, trailing_ma_period = _resolve_trailing_take_profit_controls(
        bool(enable_trailing_take_profit),
        float(trailing_take_profit_activation_pct),
        float(trailing_take_profit_drawdown_pct),
        int(trailing_take_profit_ma_period),
    )
    return BacktestRiskInputs(
        take_profit=float(take_profit),
        stop_loss=float(stop_loss),
        max_holding=int(max_holding),
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
        initial_equity=float(initial_equity),
        intrabar_exit_policy=str(intrabar_exit_policy),
        trailing_take_profit_enabled=bool(enable_trailing_take_profit),
        trailing_take_profit_activation_pct=trailing_activation,
        trailing_take_profit_drawdown_pct=trailing_drawdown,
        trailing_take_profit_ma_period=trailing_ma_period,
    )


def _backtest_data_quality_module() -> BacktestDataQualityInputs:
    with _backtest_module_container("3. 数据质量检查", "先过滤低质量本地 K 线，再进入形态识别和撮合。"):
        q1, q2 = st.columns(2)
        with q1:
            strict_data_quality = st.checkbox(
                "严格数据质量检查",
                value=True,
                key="bt_strict_quality",
                help=BACKTEST_HELP_TEXT["strict_data_quality"],
            )
        with q2:
            min_coverage_ratio_input = st.number_input(
                "最低K线覆盖率",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                format="%.2f",
                key="bt_min_coverage_ratio",
                help=BACKTEST_HELP_TEXT["min_coverage_ratio"],
            )
    min_coverage_ratio = float(min_coverage_ratio_input) if float(min_coverage_ratio_input) > 0 else None
    return BacktestDataQualityInputs(
        strict_data_quality=bool(strict_data_quality),
        min_coverage_ratio=min_coverage_ratio,
    )


def _backtest_higher_timeframe_module(timeframe: str) -> HigherTimeframeInputs:
    with _backtest_module_container("4. 大周期方向过滤", "用更大周期判断主方向，只过滤逆势订单，不改趋势、区间、通道或反转识别结果。"):
        mtf1, mtf2 = st.columns(2)
        higher_timeframe_options = ["", *[item for item in INTRADAY_TIMEFRAMES if item != timeframe]]
        with mtf1:
            higher_timeframe = st.selectbox(
                "大周期方向过滤",
                higher_timeframe_options,
                format_func=lambda value: "关闭" if value == "" else value,
                key="bt_higher_timeframe",
                help=BACKTEST_HELP_TEXT["higher_timeframe"],
            )
        with mtf2:
            higher_timeframe_max_age = st.number_input(
                "大周期信号有效分钟",
                min_value=0,
                value=0,
                step=15,
                key="bt_higher_timeframe_max_age",
                help=BACKTEST_HELP_TEXT["higher_timeframe_max_age"],
            )
    max_age = int(higher_timeframe_max_age) if int(higher_timeframe_max_age) > 0 else None
    return HigherTimeframeInputs(higher_timeframe=str(higher_timeframe), higher_timeframe_max_age_minutes=max_age)


def _legacy_backtest_module(
    data_root: Path,
    adjust: str,
    scope: BacktestScopeInputs,
    risk: BacktestRiskInputs,
    quality: BacktestDataQualityInputs,
) -> None:
    with _backtest_module_container("5. 旧突破参数", "兼容早期标志K + 通道突破回测，后续主线优先使用单策略或组合策略。"):
        strategy_config = _strategy_controls("bt")
    output = _backtest_output_module("legacy", _experiment_name("legacy-breakout", scope.timeframe, scope.start, scope.end), "运行回测", enable_save=False)
    if not output.run_clicked:
        return
    bundle = _load_backtest_bundle(
        data_root,
        adjust,
        symbols=scope.symbols,
        timeframe=scope.timeframe,
        start=scope.start,
        end=scope.end,
        strict_data_quality=quality.strict_data_quality,
        min_coverage_ratio=quality.min_coverage_ratio,
    )
    scanned = scan_bars(bundle.bars, strategy_config)
    result = run_backtest(
        scanned,
        BacktestConfig(
            take_profit_pct=risk.take_profit,
            stop_loss_pct=risk.stop_loss,
            max_holding_bars=risk.max_holding,
            fee_rate=risk.fee_rate,
            slippage_bps=risk.slippage_bps,
            initial_equity=risk.initial_equity,
            intrabar_exit_policy=risk.intrabar_exit_policy,
            trailing_take_profit_activation_pct=risk.trailing_take_profit_activation_pct,
            trailing_take_profit_drawdown_pct=risk.trailing_take_profit_drawdown_pct,
            trailing_take_profit_ma_period=risk.trailing_take_profit_ma_period,
        ),
    )
    _render_backtest_result(result, bundle, stock_names=_backtest_stock_names(data_root, scope.symbols))


def _single_strategy_module(scope: BacktestScopeInputs) -> SingleStrategyInputs:
    with _backtest_module_container("5. 单策略参数", "只绑定一个形态识别模块，用于验证单一交易形态。"):
        top1, top2 = st.columns([2, 1])
        with top1:
            detector = st.selectbox(
                "单策略形态",
                ["trend", "range", "channel", "reversal"],
                index=0,
                key="single_detector",
                help=BACKTEST_HELP_TEXT["single_detector"],
            )
        with top2:
            side_mode = st.selectbox(
                "单策略交易方向",
                list(SUPPORTED_SIDE_MODES),
                index=0,
                key="single_side_mode",
                format_func=_side_mode_label,
                help=BACKTEST_HELP_TEXT["side_mode"],
            )
        experiment_name = _experiment_name(f"single-{detector}", scope.timeframe, scope.start, scope.end)
        s1, s2, s3 = st.columns(3)
        with s1:
            risk_reward = st.number_input(
                "单策略盈亏比",
                min_value=0.1,
                value=2.0,
                step=0.1,
                key="single_rr",
                help=BACKTEST_HELP_TEXT["risk_reward"],
            )
            trend_lookback = st.number_input(
                "趋势回看",
                min_value=3,
                value=20,
                key="single_trend_lookback",
                help=BACKTEST_HELP_TEXT["trend_lookback"],
            )
        with s2:
            trend_min_score = st.number_input(
                "趋势最低评分",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="single_trend_score",
                help=BACKTEST_HELP_TEXT["trend_min_score"],
            )
            range_lookback = st.number_input(
                "区间回看",
                min_value=3,
                value=20,
                key="single_range_lookback",
                help=BACKTEST_HELP_TEXT["range_lookback"],
            )
        with s3:
            trend_h2_min_pullback_legs = st.number_input(
                "H2/L2最少回撤腿数",
                min_value=1,
                value=2,
                key="single_h2_legs",
                help=BACKTEST_HELP_TEXT["trend_h2_min_pullback_legs"],
            )
            channel_lookback = st.number_input(
                "通道回看",
                min_value=3,
                value=40,
                key="single_channel_lookback",
                help=BACKTEST_HELP_TEXT["channel_lookback"],
            )
        channel_method = st.selectbox(
            "通道算法",
            ["regression", "swing"],
            format_func=lambda value: "回归通道" if value == "regression" else "摆动点通道",
            key="single_channel_method",
            help=BACKTEST_HELP_TEXT["channel_method"],
        )
        risk_c1, risk_c2 = st.columns(2)
        with risk_c1:
            max_actual_risk_pct = st.number_input(
                "结构止损最大风险",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key="single_max_actual_risk_pct",
                help=BACKTEST_HELP_TEXT["max_actual_risk_pct"],
            )
        with risk_c2:
            max_chase_pct = st.number_input(
                "最大追价距离",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key="single_max_chase_pct",
                help=BACKTEST_HELP_TEXT["max_chase_pct"],
            )
        reversal_lookback = st.number_input(
            "反转回看",
            min_value=3,
            value=20,
            key="single_reversal_lookback",
            help=BACKTEST_HELP_TEXT["reversal_lookback"],
        )
        channel_sigma = st.number_input(
            "通道带宽倍数",
            min_value=0.1,
            value=2.0,
            step=0.1,
            key="single_channel_sigma",
            help=BACKTEST_HELP_TEXT["channel_sigma"],
        )
        advanced_detector = _detector_parameter_controls("single")
        terminal_false_breakout = _terminal_false_breakout_controls("single")
        r1, r2, r3 = st.columns(3)
        with r1:
            reversal_old_extreme_tolerance_pct = st.number_input(
                "反转旧极端容忍度",
                min_value=0.0,
                value=0.01,
                step=0.005,
                format="%.3f",
                key="single_reversal_old_extreme_tolerance_pct",
                help=BACKTEST_HELP_TEXT["reversal_old_extreme_tolerance_pct"],
            )
        with r2:
            reversal_require_old_extreme_test = st.checkbox(
                "要求旧极端失败测试",
                value=True,
                key="single_reversal_require_old_extreme_test",
                help=BACKTEST_HELP_TEXT["reversal_require_old_extreme_test"],
            )
        with r3:
            reversal_require_structure_confirmation = st.checkbox(
                "要求结构确认",
                value=True,
                key="single_reversal_require_structure_confirmation",
                help=BACKTEST_HELP_TEXT["reversal_require_structure_confirmation"],
            )
    return SingleStrategyInputs(
        detector=str(detector),
        experiment_name=experiment_name,
        risk_reward=float(risk_reward),
        side_mode=str(side_mode),
        trend_lookback=int(trend_lookback),
        trend_min_score=float(trend_min_score),
        trend_h2_min_pullback_legs=int(trend_h2_min_pullback_legs),
        range_lookback=int(range_lookback),
        channel_lookback=int(channel_lookback),
        channel_method=str(channel_method),
        channel_sigma=float(channel_sigma),
        max_actual_risk_pct=float(max_actual_risk_pct) if float(max_actual_risk_pct) > 0 else None,
        max_chase_pct=float(max_chase_pct) if float(max_chase_pct) > 0 else None,
        reversal_lookback=int(reversal_lookback),
        reversal_old_extreme_tolerance_pct=float(reversal_old_extreme_tolerance_pct),
        reversal_require_old_extreme_test=bool(reversal_require_old_extreme_test),
        reversal_require_structure_confirmation=bool(reversal_require_structure_confirmation),
        advanced_detector=advanced_detector,
        terminal_false_breakout=terminal_false_breakout,
    )


def _portfolio_allocation_module(scope: BacktestScopeInputs) -> PortfolioAllocationInputs:
    with _backtest_module_container("5. 组合仓位与资金", "组合层只处理多个单策略订单之间的资金、容量和暴露约束。"):
        top1, top2 = st.columns([3, 1])
        with top1:
            detectors = st.multiselect(
                "组合形态",
                ["trend", "range", "channel", "reversal"],
                default=["trend", "range", "channel"],
                help=BACKTEST_HELP_TEXT["single_detector"],
            )
        with top2:
            side_mode = st.selectbox(
                "组合交易方向",
                list(SUPPORTED_SIDE_MODES),
                index=0,
                key="pf_side_mode",
                format_func=_side_mode_label,
                help=BACKTEST_HELP_TEXT["side_mode"],
            )
        experiment_name = _experiment_name("portfolio", scope.timeframe, scope.start, scope.end)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            risk_reward = st.number_input(
                "组合盈亏比",
                min_value=0.1,
                value=2.0,
                step=0.1,
                key="pf_rr",
                help=BACKTEST_HELP_TEXT["risk_reward"],
            )
        with p2:
            max_open_positions = st.number_input(
                "最大组合持仓",
                min_value=1,
                value=5,
                key="pf_max_open",
                help=BACKTEST_HELP_TEXT["max_open_positions"],
            )
        with p3:
            risk_per_trade = st.number_input(
                "单笔风险预算",
                min_value=0.0,
                value=0.0,
                step=0.0025,
                format="%.4f",
                key="pf_risk",
                help=BACKTEST_HELP_TEXT["risk_per_trade"],
            )
        with p4:
            short_margin_rate = st.number_input(
                "空头保证金倍数",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="pf_short_margin",
                help=BACKTEST_HELP_TEXT["short_margin_rate"],
            )
        alloc1, alloc2, alloc3, alloc4 = st.columns(4)
        with alloc1:
            capital_per_trade = st.number_input(
                "固定单笔仓位",
                min_value=0.0,
                value=0.0,
                step=0.05,
                format="%.3f",
                key="pf_capital",
                help=BACKTEST_HELP_TEXT["capital_per_trade"],
            )
        with alloc2:
            max_capital_per_trade = st.number_input(
                "最大单笔仓位",
                min_value=0.001,
                max_value=1.0,
                value=1.0,
                step=0.05,
                format="%.3f",
                key="pf_max_capital",
                help=BACKTEST_HELP_TEXT["max_capital_per_trade"],
            )
        with alloc3:
            reserve_cash = st.number_input(
                "预留现金",
                min_value=0.0,
                max_value=0.99,
                value=0.0,
                step=0.05,
                format="%.3f",
                key="pf_reserve_cash",
                help=BACKTEST_HELP_TEXT["reserve_cash"],
            )
        with alloc4:
            allow_same_symbol_overlap = st.checkbox(
                "允许同票重叠",
                value=False,
                key="pf_allow_overlap",
                help=BACKTEST_HELP_TEXT["allow_same_symbol_overlap"],
            )
        map1, map2 = st.columns(2)
        with map1:
            strategy_priority_text = st.text_area(
                "策略优先级",
                value="",
                placeholder="trend_signal_bar=1,range_signal_bar=2",
                key="pf_strategy_priority",
                help=BACKTEST_HELP_TEXT["strategy_priority"],
            )
            strategy_capital_limit_text = st.text_area(
                "策略资金上限",
                value="",
                placeholder="trend_signal_bar=0.6",
                key="pf_strategy_limit",
                help=BACKTEST_HELP_TEXT["strategy_capital_limit"],
            )
        with map2:
            sector_capital_limit_text = st.text_area(
                "行业资金上限",
                value="",
                placeholder="银行=0.5,新能源=0.4",
                key="pf_sector_limit",
                help=BACKTEST_HELP_TEXT["sector_capital_limit"],
            )
            symbol_sector_map_text = st.text_area(
                "股票行业映射",
                value="",
                placeholder="000001.SZ=银行,300750.SZ=新能源",
                key="pf_symbol_sector",
                help=BACKTEST_HELP_TEXT["symbol_sector_map"],
            )
        pf_r1, pf_r2 = st.columns(2)
        with pf_r1:
            max_actual_risk_pct = st.number_input(
                "组合结构止损最大风险",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key="pf_max_actual_risk_pct",
                help=BACKTEST_HELP_TEXT["max_actual_risk_pct"],
            )
        with pf_r2:
            max_chase_pct = st.number_input(
                "组合最大追价距离",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key="pf_max_chase_pct",
                help=BACKTEST_HELP_TEXT["max_chase_pct"],
            )
    return PortfolioAllocationInputs(
        detectors=tuple(str(item) for item in detectors),
        experiment_name=experiment_name,
        risk_reward=float(risk_reward),
        side_mode=str(side_mode),
        max_open_positions=int(max_open_positions),
        risk_per_trade=float(risk_per_trade) if float(risk_per_trade) > 0 else None,
        short_margin_rate=float(short_margin_rate),
        capital_per_trade=float(capital_per_trade) if float(capital_per_trade) > 0 else None,
        max_capital_per_trade=float(max_capital_per_trade),
        reserve_cash=float(reserve_cash),
        allow_same_symbol_overlap=bool(allow_same_symbol_overlap),
        strategy_priority_text=str(strategy_priority_text),
        strategy_capital_limit_text=str(strategy_capital_limit_text),
        sector_capital_limit_text=str(sector_capital_limit_text),
        symbol_sector_map_text=str(symbol_sector_map_text),
        max_actual_risk_pct=float(max_actual_risk_pct) if float(max_actual_risk_pct) > 0 else None,
        max_chase_pct=float(max_chase_pct) if float(max_chase_pct) > 0 else None,
    )


def _portfolio_detector_module() -> PortfolioDetectorInputs:
    with _backtest_module_container("6. 组合形态识别参数", "这里仍只配置各类形态的识别阈值，不配置仓位。"):
        detector_c1, detector_c2, detector_c3 = st.columns(3)
        with detector_c1:
            trend_lookback = st.number_input(
                "组合趋势回看",
                min_value=3,
                value=20,
                key="pf_trend_lookback",
                help=BACKTEST_HELP_TEXT["trend_lookback"],
            )
            channel_lookback = st.number_input(
                "组合通道回看",
                min_value=3,
                value=40,
                key="pf_channel_lookback",
                help=BACKTEST_HELP_TEXT["channel_lookback"],
            )
        with detector_c2:
            trend_min_score = st.number_input(
                "组合趋势最低评分",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="pf_trend_score",
                help=BACKTEST_HELP_TEXT["trend_min_score"],
            )
            channel_sigma = st.number_input(
                "组合通道带宽倍数",
                min_value=0.1,
                value=2.0,
                step=0.1,
                key="pf_channel_sigma",
                help=BACKTEST_HELP_TEXT["channel_sigma"],
            )
        with detector_c3:
            range_lookback = st.number_input(
                "组合区间回看",
                min_value=3,
                value=20,
                key="pf_range_lookback",
                help=BACKTEST_HELP_TEXT["range_lookback"],
            )
            reversal_lookback = st.number_input(
                "组合反转回看",
                min_value=3,
                value=20,
                key="pf_reversal_lookback",
                help=BACKTEST_HELP_TEXT["reversal_lookback"],
            )
        trend_h2_min_pullback_legs = st.number_input(
            "组合H2/L2最少回撤腿数",
            min_value=1,
            value=2,
            key="pf_h2_legs",
            help=BACKTEST_HELP_TEXT["trend_h2_min_pullback_legs"],
        )
        channel_method = st.selectbox(
            "组合通道算法",
            ["regression", "swing"],
            format_func=lambda value: "回归通道" if value == "regression" else "摆动点通道",
            key="pf_channel_method",
            help=BACKTEST_HELP_TEXT["channel_method"],
        )
        advanced_detector = _detector_parameter_controls("pf", "组合")
        terminal_false_breakout = _terminal_false_breakout_controls("pf", "组合")
        pr1, pr2, pr3 = st.columns(3)
        with pr1:
            reversal_old_extreme_tolerance_pct = st.number_input(
                "组合反转旧极端容忍度",
                min_value=0.0,
                value=0.01,
                step=0.005,
                format="%.3f",
                key="pf_reversal_old_extreme_tolerance_pct",
                help=BACKTEST_HELP_TEXT["reversal_old_extreme_tolerance_pct"],
            )
        with pr2:
            reversal_require_old_extreme_test = st.checkbox(
                "组合要求旧极端失败测试",
                value=True,
                key="pf_reversal_require_old_extreme_test",
                help=BACKTEST_HELP_TEXT["reversal_require_old_extreme_test"],
            )
        with pr3:
            reversal_require_structure_confirmation = st.checkbox(
                "组合要求结构确认",
                value=True,
                key="pf_reversal_require_structure_confirmation",
                help=BACKTEST_HELP_TEXT["reversal_require_structure_confirmation"],
            )
    return PortfolioDetectorInputs(
        trend_lookback=int(trend_lookback),
        channel_lookback=int(channel_lookback),
        trend_min_score=float(trend_min_score),
        channel_sigma=float(channel_sigma),
        range_lookback=int(range_lookback),
        reversal_lookback=int(reversal_lookback),
        trend_h2_min_pullback_legs=int(trend_h2_min_pullback_legs),
        channel_method=str(channel_method),
        reversal_old_extreme_tolerance_pct=float(reversal_old_extreme_tolerance_pct),
        reversal_require_old_extreme_test=bool(reversal_require_old_extreme_test),
        reversal_require_structure_confirmation=bool(reversal_require_structure_confirmation),
        advanced_detector=advanced_detector,
        terminal_false_breakout=terminal_false_breakout,
    )


def _backtest_output_module(
    prefix: str,
    experiment_name: str,
    button_label: str,
    *,
    enable_save: bool = True,
    title: str = "6. 保存与运行",
    compact: bool = False,
) -> BacktestOutputInputs:
    with _backtest_module_container(title, "输出目录和运行按钮单独放置，避免和参数区混在一起。"):
        if enable_save:
            save_outputs, output_dir = _experiment_output_controls(prefix, experiment_name, compact=compact)
        else:
            save_outputs, output_dir = False, ""
        run_clicked = st.button(button_label, type="primary")
    return BacktestOutputInputs(save_outputs=bool(save_outputs), output_dir=str(output_dir), run_clicked=bool(run_clicked))


def _execute_single_strategy_experiment(
    data_root: Path,
    adjust: str,
    scope: BacktestScopeInputs,
    risk: BacktestRiskInputs,
    quality: BacktestDataQualityInputs,
    higher: HigherTimeframeInputs,
    single: SingleStrategyInputs,
    output: BacktestOutputInputs,
) -> None:
    terminal_false_breakout = single.terminal_false_breakout
    experiment = run_single_strategy_experiment(
        SingleStrategyExperimentConfig(
            name=single.experiment_name,
            data_root=str(data_root),
            symbols=tuple(scope.symbols),
            timeframe=scope.timeframe,
            higher_timeframe=higher.higher_timeframe,
            higher_timeframe_max_age_minutes=higher.higher_timeframe_max_age_minutes,
            start=scope.start,
            end=scope.end,
            detector=single.detector,
            adjust=adjust,
            risk_reward=single.risk_reward,
            side_mode=single.side_mode,
            max_holding_bars=risk.max_holding,
            max_actual_risk_pct=single.max_actual_risk_pct,
            max_chase_pct=single.max_chase_pct,
            fee_rate=risk.fee_rate,
            slippage_bps=risk.slippage_bps,
            initial_equity=risk.initial_equity,
            intrabar_exit_policy=risk.intrabar_exit_policy,
            trailing_take_profit_activation_pct=risk.trailing_take_profit_activation_pct,
            trailing_take_profit_drawdown_pct=risk.trailing_take_profit_drawdown_pct,
            trailing_take_profit_ma_period=risk.trailing_take_profit_ma_period,
            terminal_false_breakout_enabled=terminal_false_breakout.enabled,
            terminal_false_breakout_detectors=terminal_false_breakout.detectors,
            terminal_false_breakout_lookback=terminal_false_breakout.lookback,
            terminal_false_breakout_atr_period=terminal_false_breakout.atr_period,
            terminal_false_breakout_min_regime_bars=terminal_false_breakout.min_regime_bars,
            terminal_false_breakout_extension_atr_multiple=terminal_false_breakout.extension_atr_multiple,
            terminal_false_breakout_edge_lookback=terminal_false_breakout.edge_lookback,
            terminal_false_breakout_edge_pos=terminal_false_breakout.edge_pos,
            terminal_false_breakout_edge_min_count=terminal_false_breakout.edge_min_count,
            terminal_false_breakout_weak_progress_atr=terminal_false_breakout.weak_progress_atr,
            terminal_false_breakout_wick_ratio=terminal_false_breakout.wick_ratio,
            terminal_false_breakout_min_score=terminal_false_breakout.min_score,
            strict_data_quality=quality.strict_data_quality,
            min_coverage_ratio=quality.min_coverage_ratio,
            output_dir=output.output_dir if output.save_outputs else "",
            trend_lookback=single.trend_lookback,
            trend_min_score=single.trend_min_score,
            trend_strong_close_pos=float(single.advanced_detector["trend_strong_close_pos"]),
            trend_min_body_ratio=float(single.advanced_detector["trend_min_body_ratio"]),
            trend_pullback_lookback=int(single.advanced_detector["trend_pullback_lookback"]),
            trend_h2_min_pullback_legs=single.trend_h2_min_pullback_legs,
            range_lookback=single.range_lookback,
            range_middle_low=float(single.advanced_detector["range_middle_low"]),
            range_middle_high=float(single.advanced_detector["range_middle_high"]),
            range_false_break_buffer=float(single.advanced_detector["range_false_break_buffer"]),
            range_strong_close_pos=float(single.advanced_detector["range_strong_close_pos"]),
            range_min_score=float(single.advanced_detector["range_min_score"]),
            channel_method=single.channel_method,
            channel_lookback=single.channel_lookback,
            channel_sigma_multiple=single.channel_sigma,
            channel_break_buffer=float(single.advanced_detector["channel_break_buffer"]),
            channel_swing_left_bars=int(single.advanced_detector["channel_swing_left_bars"]),
            channel_swing_right_bars=int(single.advanced_detector["channel_swing_right_bars"]),
            reversal_lookback=single.reversal_lookback,
            reversal_strong_close_pos=float(single.advanced_detector["reversal_strong_close_pos"]),
            reversal_min_body_ratio=float(single.advanced_detector["reversal_min_body_ratio"]),
            reversal_old_extreme_tolerance_pct=single.reversal_old_extreme_tolerance_pct,
            reversal_require_old_extreme_test=single.reversal_require_old_extreme_test,
            reversal_require_structure_confirmation=single.reversal_require_structure_confirmation,
        ),
        save=output.save_outputs,
    )
    stock_names = _backtest_stock_names(data_root, scope.symbols)
    _render_backtest_result(
        experiment.backtest,
        bars=experiment.bars,
        filtered_limit_open_count=experiment.filtered_limit_open_count,
        data_coverage=experiment.data_coverage,
        stock_names=stock_names,
    )
    _render_experiment_breakdowns(experiment, stock_names=stock_names)
    if output.save_outputs:
        _show_saved_experiment_path(experiment.config.output_dir, experiment.config.name)


def _execute_portfolio_strategy_experiment(
    data_root: Path,
    adjust: str,
    scope: BacktestScopeInputs,
    risk: BacktestRiskInputs,
    quality: BacktestDataQualityInputs,
    higher: HigherTimeframeInputs,
    allocation: PortfolioAllocationInputs,
    detector: PortfolioDetectorInputs,
    output: BacktestOutputInputs,
) -> None:
    advanced_detector = detector.advanced_detector
    terminal_false_breakout = detector.terminal_false_breakout
    experiment = run_portfolio_experiment(
        PortfolioExperimentConfig(
            name=allocation.experiment_name,
            data_root=str(data_root),
            symbols=tuple(scope.symbols),
            timeframe=scope.timeframe,
            higher_timeframe=higher.higher_timeframe,
            higher_timeframe_max_age_minutes=higher.higher_timeframe_max_age_minutes,
            start=scope.start,
            end=scope.end,
            adjust=adjust,
            detectors=allocation.detectors,
            risk_reward=allocation.risk_reward,
            side_mode=allocation.side_mode,
            max_holding_bars=risk.max_holding,
            max_actual_risk_pct=allocation.max_actual_risk_pct,
            max_chase_pct=allocation.max_chase_pct,
            max_open_positions=allocation.max_open_positions,
            capital_per_trade=allocation.capital_per_trade,
            risk_per_trade=allocation.risk_per_trade,
            max_capital_per_trade=allocation.max_capital_per_trade,
            short_margin_rate=allocation.short_margin_rate,
            reserve_cash=allocation.reserve_cash,
            allow_same_symbol_overlap=allocation.allow_same_symbol_overlap,
            strategy_priority=_parse_int_mapping(allocation.strategy_priority_text),
            strategy_capital_limit=_parse_float_mapping(allocation.strategy_capital_limit_text),
            sector_capital_limit=_parse_float_mapping(allocation.sector_capital_limit_text),
            symbol_sector_map=_parse_text_mapping(allocation.symbol_sector_map_text),
            fee_rate=risk.fee_rate,
            slippage_bps=risk.slippage_bps,
            initial_equity=risk.initial_equity,
            intrabar_exit_policy=risk.intrabar_exit_policy,
            trailing_take_profit_activation_pct=risk.trailing_take_profit_activation_pct,
            trailing_take_profit_drawdown_pct=risk.trailing_take_profit_drawdown_pct,
            trailing_take_profit_ma_period=risk.trailing_take_profit_ma_period,
            terminal_false_breakout_enabled=terminal_false_breakout.enabled,
            terminal_false_breakout_detectors=terminal_false_breakout.detectors,
            terminal_false_breakout_lookback=terminal_false_breakout.lookback,
            terminal_false_breakout_atr_period=terminal_false_breakout.atr_period,
            terminal_false_breakout_min_regime_bars=terminal_false_breakout.min_regime_bars,
            terminal_false_breakout_extension_atr_multiple=terminal_false_breakout.extension_atr_multiple,
            terminal_false_breakout_edge_lookback=terminal_false_breakout.edge_lookback,
            terminal_false_breakout_edge_pos=terminal_false_breakout.edge_pos,
            terminal_false_breakout_edge_min_count=terminal_false_breakout.edge_min_count,
            terminal_false_breakout_weak_progress_atr=terminal_false_breakout.weak_progress_atr,
            terminal_false_breakout_wick_ratio=terminal_false_breakout.wick_ratio,
            terminal_false_breakout_min_score=terminal_false_breakout.min_score,
            strict_data_quality=quality.strict_data_quality,
            min_coverage_ratio=quality.min_coverage_ratio,
            output_dir=output.output_dir if output.save_outputs else "",
            trend_lookback=detector.trend_lookback,
            trend_min_score=detector.trend_min_score,
            trend_strong_close_pos=float(advanced_detector["trend_strong_close_pos"]),
            trend_min_body_ratio=float(advanced_detector["trend_min_body_ratio"]),
            trend_pullback_lookback=int(advanced_detector["trend_pullback_lookback"]),
            trend_h2_min_pullback_legs=detector.trend_h2_min_pullback_legs,
            range_lookback=detector.range_lookback,
            range_middle_low=float(advanced_detector["range_middle_low"]),
            range_middle_high=float(advanced_detector["range_middle_high"]),
            range_false_break_buffer=float(advanced_detector["range_false_break_buffer"]),
            range_strong_close_pos=float(advanced_detector["range_strong_close_pos"]),
            range_min_score=float(advanced_detector["range_min_score"]),
            channel_method=detector.channel_method,
            channel_lookback=detector.channel_lookback,
            channel_sigma_multiple=detector.channel_sigma,
            channel_break_buffer=float(advanced_detector["channel_break_buffer"]),
            channel_swing_left_bars=int(advanced_detector["channel_swing_left_bars"]),
            channel_swing_right_bars=int(advanced_detector["channel_swing_right_bars"]),
            reversal_lookback=detector.reversal_lookback,
            reversal_strong_close_pos=float(advanced_detector["reversal_strong_close_pos"]),
            reversal_min_body_ratio=float(advanced_detector["reversal_min_body_ratio"]),
            reversal_old_extreme_tolerance_pct=detector.reversal_old_extreme_tolerance_pct,
            reversal_require_old_extreme_test=detector.reversal_require_old_extreme_test,
            reversal_require_structure_confirmation=detector.reversal_require_structure_confirmation,
        ),
        save=output.save_outputs,
    )
    stock_names = _backtest_stock_names(data_root, scope.symbols)
    _render_backtest_result(
        experiment.backtest,
        bars=experiment.bars,
        filtered_limit_open_count=experiment.filtered_limit_open_count,
        data_coverage=experiment.data_coverage,
        stock_names=stock_names,
    )
    _render_experiment_breakdowns(experiment, stock_names=stock_names)
    if output.save_outputs:
        _show_saved_experiment_path(experiment.config.output_dir, experiment.config.name)


def _render_backtest_result(
    result,
    bundle: BacktestDataBundle | None = None,
    *,
    bars: pd.DataFrame | None = None,
    filtered_limit_open_count: int | None = None,
    data_coverage: pd.DataFrame | None = None,
    stock_names: Mapping[str, str] | None = None,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("交易数", int(result.stats["trade_count"]))
    c2.metric("胜率", f"{result.stats['win_rate']:.1%}")
    c3.metric("总收益", f"{result.stats['total_return']:.1%}")
    c4.metric("最大回撤", f"{result.stats['max_drawdown']:.1%}")
    filtered_count = len(bundle.filtered_limit_open_days) if bundle is not None else int(filtered_limit_open_count or 0)
    if filtered_count > 0:
        st.caption(f"已过滤涨停开盘交易日：{filtered_count} 条")
    chart_bars = bundle.bars if bundle is not None else bars
    if chart_bars is not None and not chart_bars.empty:
        _render_strategy_kline_chart(chart_bars, result.trades, stock_names=stock_names)
    _render_display_table("核心绩效概览", _performance_summary_frame(result.stats), stock_names=stock_names)
    if not result.equity_curve.empty:
        st.markdown("##### 净值曲线")
        _render_equity_chart(result.equity_curve)
        st.markdown("##### 回撤曲线")
        _render_equity_drawdown_chart(result.equity_curve)
        _render_display_table("回撤区间明细", _equity_drawdown_episodes_frame(result.equity_curve), stock_names=stock_names)
    if data_coverage is not None and not data_coverage.empty:
        _render_data_coverage_chart(data_coverage, stock_names=stock_names)
        _render_display_table("数据覆盖率检查", data_coverage, stock_names=stock_names)
    _render_order_decision_charts(result.order_decisions)
    _render_display_table("逐笔交易", result.trades, stock_names=stock_names)
    if not result.equity_curve.empty:
        _render_display_table("净值明细", result.equity_curve, tail=200, stock_names=stock_names)


def _render_experiment_breakdowns(experiment, *, stock_names: Mapping[str, str] | None = None) -> None:
    """展示实验拆分统计；单策略和组合回测复用同一组产物。"""
    if not experiment.diagnostic_report.empty:
        st.markdown("##### 实验诊断摘要")
        _render_diagnostic_status_chart(experiment.diagnostic_report)
        _render_display_table("实验诊断摘要", experiment.diagnostic_report, stock_names=stock_names)
    if not experiment.trade_path_distribution_stats.empty:
        st.markdown("##### 交易路径分布")
        _render_trade_path_distribution_chart(experiment.trade_path_distribution_stats)
        _render_display_table("交易路径分布", experiment.trade_path_distribution_stats, stock_names=stock_names)
    for title, frame in [
        ("策略绩效", experiment.strategy_stats),
        ("识别模块绩效", experiment.detector_stats),
        ("信号形态绩效", experiment.setup_stats),
        ("股票绩效", experiment.symbol_stats),
        ("方向绩效", experiment.side_stats),
        ("退出原因绩效", experiment.exit_reason_stats),
        ("开平仓路径绩效", experiment.signal_lifecycle_stats),
        ("信号类型绩效", experiment.event_type_stats),
        ("订单决策统计", experiment.order_decision_stats),
        ("策略过滤统计", experiment.strategy_filter_stats),
        ("信号形态撮合统计", experiment.setup_order_decision_stats),
        ("信号形态过滤统计", experiment.setup_strategy_filter_stats),
        ("月度收益", experiment.monthly_returns),
    ]:
        _render_display_table(title, frame, stock_names=stock_names)


def _backtest_stock_names(data_root: Path, symbols: list[str]) -> dict[str, str]:
    return MarketDataRepository(data_root).symbol_names(symbols=tuple(symbols))


def _chart_close(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    chart = frame.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
    chart = chart.rename(columns={column: _stock_name_label(column) for column in chart.columns})
    st.line_chart(chart)


if __name__ == "__main__":
    main()
