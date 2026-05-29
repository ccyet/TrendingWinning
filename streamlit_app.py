from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from trending_winning.backtest.engine import BacktestConfig, run_backtest
from trending_winning.backtest.experiment import (
    PortfolioExperimentConfig,
    SingleStrategyExperimentConfig,
    run_portfolio_experiment,
    run_single_strategy_experiment,
)
from trending_winning.data.repository import BacktestDataBundle, MarketDataRepository, available_symbols
from trending_winning.multitimeframe import scan_timeframes
from trending_winning.strategy import StrategyConfig, scan_bars

DEFAULT_DATA_ROOT = "/Users/a1234/Desktop/trend-backtest/data/market/daily"
DEFAULT_OUTPUT_ROOT = "runs"
# 数据管理要补日 K，策略执行仍只跑分钟级，避免日 K 误入同级别策略回测。
DATA_TIMEFRAMES = ["1d", "5m", "15m", "30m", "60m"]
INTRADAY_TIMEFRAMES = ["5m", "15m", "30m", "60m"]


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


def _directory_picker(label: str, default: str | Path, *, key: str, disabled: bool = False) -> Path:
    """本地文件夹选择器；用下拉框和按钮浏览路径，避免让用户手写目录字符串。"""
    selected_key = f"{key}_selected_path"
    current_key = f"{key}_current_path"
    default_path = Path(default).expanduser()
    if selected_key not in st.session_state:
        st.session_state[selected_key] = str(default_path)
    if current_key not in st.session_state:
        st.session_state[current_key] = str(_initial_browse_directory(default_path))

    st.markdown(f"**{label}**")
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
    st.caption(f"已选文件夹：{_display_path(str(selected))}")
    return selected


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


def _symbol_input(label: str, default: str = "000001.SZ", *, key: str) -> list[str]:
    raw = st.text_input(label, default, key=key)
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _date_inputs(key_prefix: str) -> tuple[str, str]:
    today = date.today()
    start = st.date_input("开始日期", today - timedelta(days=20), key=f"{key_prefix}_start")
    end = st.date_input("结束日期", today, key=f"{key_prefix}_end")
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


def _experiment_output_controls(prefix: str, default_name: str) -> tuple[bool, str]:
    c1, c2 = st.columns([1, 3])
    with c1:
        save_outputs = st.checkbox("保存实验产物", value=False, key=f"{prefix}_save_outputs")
    with c2:
        output_root = _directory_picker(
            "输出父目录",
            DEFAULT_OUTPUT_ROOT,
            key=f"{prefix}_output_root",
            disabled=not save_outputs,
        )
        output_dir = output_root / default_name
        if save_outputs:
            st.caption(f"本次保存到：{_display_path(str(output_dir))}")
    return bool(save_outputs), str(output_dir)


def _detector_parameter_controls(prefix: str, label_prefix: str = "") -> dict[str, float | int]:
    """高级 detector 参数；单策略和组合回测共用，避免 Web 表单和配置层脱节。"""
    with st.expander(f"{label_prefix}高级 detector 参数", expanded=False):
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
            )
        with trend_c3:
            trend_pullback_lookback = st.number_input(
                f"{label_prefix}趋势回撤窗口",
                min_value=1,
                value=5,
                key=f"{prefix}_trend_pullback_lookback",
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
            )
        with range_c3:
            range_false_break_buffer = st.number_input(
                f"{label_prefix}区间失败突破缓冲",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key=f"{prefix}_range_false_break_buffer",
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
            )
        with range_c5:
            range_min_score = st.number_input(
                f"{label_prefix}区间最低评分",
                min_value=0.0,
                value=0.8,
                step=0.1,
                format="%.2f",
                key=f"{prefix}_range_min_score",
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
            )
        with channel_c2:
            channel_swing_left_bars = st.number_input(
                f"{label_prefix}摆动左侧K数",
                min_value=1,
                value=2,
                key=f"{prefix}_channel_swing_left_bars",
            )
        with channel_c3:
            channel_swing_right_bars = st.number_input(
                f"{label_prefix}摆动右侧K数",
                min_value=1,
                value=2,
                key=f"{prefix}_channel_swing_right_bars",
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


def _show_saved_experiment_path(output_dir: str, name: str) -> None:
    saved_path = Path(output_dir or f"runs/{name}").expanduser()
    st.caption(f"实验产物已保存：{saved_path}")


def _fetch_panel(data_root: Path, adjust: str, tdx_path: str) -> None:
    st.subheader("TDX K 线落地")
    fetch_cols = st.columns([2, 2, 1, 1, 1])
    with fetch_cols[0]:
        symbols = _symbol_input("标的代码", "000001.SZ,600519.SH", key="fetch_symbols")
    with fetch_cols[1]:
        timeframes = st.multiselect("周期", DATA_TIMEFRAMES, default=DATA_TIMEFRAMES, key="fetch_timeframes")
    with fetch_cols[2]:
        start = str(st.date_input("开始日期", date.today() - timedelta(days=20), key="fetch_start"))
    with fetch_cols[3]:
        end = str(st.date_input("结束日期", date.today(), key="fetch_end"))
    with fetch_cols[4]:
        prepare_min_coverage_input = st.number_input(
            "补齐最低覆盖率",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="fetch_prepare_min_coverage",
        )
    prepare_min_coverage = float(prepare_min_coverage_input) if float(prepare_min_coverage_input) > 0 else None
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
        landmark_lookback = st.number_input("标志K回看", min_value=2, value=20, key=f"{prefix}_landmark_lookback")
        landmark_range_multiple = st.number_input("标志K振幅倍数", min_value=0.1, value=1.8, step=0.1, key=f"{prefix}_range")
    with c2:
        channel_lookback = st.number_input("通道回看", min_value=3, value=40, key=f"{prefix}_channel")
        trigger_volume_multiple = st.number_input("突破量能倍数", min_value=0.1, value=1.5, step=0.1, key=f"{prefix}_volume")
    with c3:
        close_buffer = st.number_input("突破收盘缓冲", min_value=0.0, value=0.0, step=0.005, format="%.3f", key=f"{prefix}_buffer")
        require_landmark = st.checkbox("突破必须同时是标志K", value=True, key=f"{prefix}_require_landmark")
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
        timeframe = st.selectbox("周期", INTRADAY_TIMEFRAMES, index=2, key=f"{prefix}_tf")
    available = available_symbols(data_root, timeframe=timeframe)
    default_symbols = ",".join(available[:5]) if available else "000001.SZ"
    with scope_cols[1]:
        symbols = _symbol_input("标的代码", default_symbols, key=f"{prefix}_symbols")
    with scope_cols[2]:
        start = str(st.date_input("开始日期", date.today() - timedelta(days=20), key=f"{prefix}_start"))
    with scope_cols[3]:
        end = str(st.date_input("结束日期", date.today(), key=f"{prefix}_end"))
    return symbols, timeframe, start, end


def _scan_panel(data_root: Path, adjust: str) -> None:
    st.subheader("标志K + 趋势通道 + 突破扫描")
    scan_cols = st.columns([2, 2, 1, 1])
    with scan_cols[0]:
        timeframes = st.multiselect("周期", INTRADAY_TIMEFRAMES, default=["30m", "60m"], key="scan_timeframes")
    available = sorted({symbol for timeframe in timeframes for symbol in available_symbols(data_root, timeframe=timeframe)})
    default_symbols = ",".join(available[:5]) if available else "000001.SZ"
    with scan_cols[1]:
        symbols = _symbol_input("标的代码", default_symbols, key="scan_symbols")
    with scan_cols[2]:
        start = str(st.date_input("开始日期", date.today() - timedelta(days=20), key="scan_start"))
    with scan_cols[3]:
        end = str(st.date_input("结束日期", date.today(), key="scan_end"))
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
    symbols, timeframe, start, end = _load_panel_inputs(data_root, "bt")
    mode = st.radio("回测模式", ["旧突破回测", "单策略回测", "组合策略回测"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        take_profit = st.number_input("止盈", min_value=0.001, value=0.06, step=0.005, format="%.3f")
    with c2:
        stop_loss = st.number_input("止损", min_value=0.001, value=0.03, step=0.005, format="%.3f")
    with c3:
        max_holding = st.number_input("最大持有K数", min_value=1, value=12)
    cost1, cost2, cost3 = st.columns(3)
    with cost1:
        fee_rate = st.number_input("手续费率", min_value=0.0, value=0.0, step=0.0001, format="%.4f", key="bt_fee_rate")
    with cost2:
        slippage_bps = st.number_input("滑点bps", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="bt_slippage_bps")
    with cost3:
        initial_equity = st.number_input("初始资金", min_value=0.0001, value=1.0, step=0.1, format="%.4f", key="bt_initial_equity")
    q1, q2 = st.columns(2)
    with q1:
        strict_data_quality = st.checkbox("严格数据质量门禁", value=True, key="bt_strict_quality")
    with q2:
        min_coverage_ratio_input = st.number_input(
            "最低覆盖率门禁",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="bt_min_coverage_ratio",
        )
    min_coverage_ratio = float(min_coverage_ratio_input) if float(min_coverage_ratio_input) > 0 else None
    intrabar_exit_policy = st.selectbox(
        "同K止盈止损冲突",
        ["conservative", "optimistic"],
        format_func=lambda value: "保守：止损优先" if value == "conservative" else "乐观：止盈优先",
        key="bt_intrabar_policy",
    )
    mtf1, mtf2 = st.columns(2)
    higher_timeframe_options = ["", *[item for item in INTRADAY_TIMEFRAMES if item != timeframe]]
    with mtf1:
        higher_timeframe = st.selectbox(
            "高周期方向门控",
            higher_timeframe_options,
            format_func=lambda value: "关闭" if value == "" else value,
            key="bt_higher_timeframe",
        )
    with mtf2:
        higher_timeframe_max_age = st.number_input(
            "高周期最大过期分钟",
            min_value=0,
            value=0,
            step=15,
            key="bt_higher_timeframe_max_age",
        )
    higher_timeframe_max_age_minutes = int(higher_timeframe_max_age) if int(higher_timeframe_max_age) > 0 else None
    if mode == "旧突破回测":
        strategy_config = _strategy_controls("bt")
        if st.button("运行回测", type="primary"):
            bundle = _load_backtest_bundle(
                data_root,
                adjust,
                symbols=symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                strict_data_quality=bool(strict_data_quality),
                min_coverage_ratio=min_coverage_ratio,
            )
            scanned = scan_bars(bundle.bars, strategy_config)
            result = run_backtest(
                scanned,
                BacktestConfig(
                    take_profit_pct=float(take_profit),
                    stop_loss_pct=float(stop_loss),
                    max_holding_bars=int(max_holding),
                    fee_rate=float(fee_rate),
                    slippage_bps=float(slippage_bps),
                    initial_equity=float(initial_equity),
                    intrabar_exit_policy=str(intrabar_exit_policy),
                ),
            )
            _render_backtest_result(result, bundle)
        return

    if mode == "单策略回测":
        detector = st.selectbox("单策略 detector", ["trend", "range", "channel", "reversal"], index=0, key="single_detector")
        experiment_name = _experiment_name(f"single-{detector}", timeframe, start, end)
        s1, s2, s3 = st.columns(3)
        with s1:
            risk_reward = st.number_input("单策略盈亏比", min_value=0.1, value=2.0, step=0.1, key="single_rr")
            trend_lookback = st.number_input("趋势回看", min_value=3, value=20, key="single_trend_lookback")
        with s2:
            trend_min_score = st.number_input("趋势最低评分", min_value=0.0, value=1.0, step=0.1, key="single_trend_score")
            range_lookback = st.number_input("区间回看", min_value=3, value=20, key="single_range_lookback")
        with s3:
            trend_h2_min_pullback_legs = st.number_input("H2/L2最少回撤腿数", min_value=1, value=2, key="single_h2_legs")
            channel_lookback = st.number_input("通道回看", min_value=3, value=40, key="single_channel_lookback")
        channel_method = st.selectbox(
            "通道算法",
            ["regression", "swing"],
            format_func=lambda value: "回归通道" if value == "regression" else "摆动点通道",
            key="single_channel_method",
        )
        risk_c1, risk_c2 = st.columns(2)
        with risk_c1:
            max_actual_risk_pct = st.number_input(
                "最大实际风险",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key="single_max_actual_risk_pct",
            )
        with risk_c2:
            max_chase_pct = st.number_input(
                "最大追价距离",
                min_value=0.0,
                value=0.0,
                step=0.005,
                format="%.3f",
                key="single_max_chase_pct",
            )
        reversal_lookback = st.number_input("反转回看", min_value=3, value=20, key="single_reversal_lookback")
        channel_sigma = st.number_input("通道带宽倍数", min_value=0.1, value=2.0, step=0.1, key="single_channel_sigma")
        advanced_detector = _detector_parameter_controls("single")
        r1, r2, r3 = st.columns(3)
        with r1:
            reversal_old_extreme_tolerance_pct = st.number_input(
                "反转旧极端容忍度",
                min_value=0.0,
                value=0.01,
                step=0.005,
                format="%.3f",
                key="single_reversal_old_extreme_tolerance_pct",
            )
        with r2:
            reversal_require_old_extreme_test = st.checkbox(
                "要求旧极端失败测试",
                value=True,
                key="single_reversal_require_old_extreme_test",
            )
        with r3:
            reversal_require_structure_confirmation = st.checkbox(
                "要求结构确认",
                value=True,
                key="single_reversal_require_structure_confirmation",
            )
        save_outputs, output_dir = _experiment_output_controls("single", experiment_name)
        if st.button("运行单策略回测", type="primary"):
            experiment = run_single_strategy_experiment(
                SingleStrategyExperimentConfig(
                    name=experiment_name,
                    data_root=str(data_root),
                    symbols=tuple(symbols),
                    timeframe=timeframe,
                    higher_timeframe=str(higher_timeframe),
                    higher_timeframe_max_age_minutes=higher_timeframe_max_age_minutes,
                    start=start,
                    end=end,
                    detector=str(detector),
                    adjust=adjust,
                    risk_reward=float(risk_reward),
                    max_holding_bars=int(max_holding),
                    max_actual_risk_pct=float(max_actual_risk_pct) if float(max_actual_risk_pct) > 0 else None,
                    max_chase_pct=float(max_chase_pct) if float(max_chase_pct) > 0 else None,
                    fee_rate=float(fee_rate),
                    slippage_bps=float(slippage_bps),
                    initial_equity=float(initial_equity),
                    intrabar_exit_policy=str(intrabar_exit_policy),
                    strict_data_quality=bool(strict_data_quality),
                    min_coverage_ratio=min_coverage_ratio,
                    output_dir=output_dir if save_outputs else "",
                    trend_lookback=int(trend_lookback),
                    trend_min_score=float(trend_min_score),
                    trend_strong_close_pos=float(advanced_detector["trend_strong_close_pos"]),
                    trend_min_body_ratio=float(advanced_detector["trend_min_body_ratio"]),
                    trend_pullback_lookback=int(advanced_detector["trend_pullback_lookback"]),
                    trend_h2_min_pullback_legs=int(trend_h2_min_pullback_legs),
                    range_lookback=int(range_lookback),
                    range_middle_low=float(advanced_detector["range_middle_low"]),
                    range_middle_high=float(advanced_detector["range_middle_high"]),
                    range_false_break_buffer=float(advanced_detector["range_false_break_buffer"]),
                    range_strong_close_pos=float(advanced_detector["range_strong_close_pos"]),
                    range_min_score=float(advanced_detector["range_min_score"]),
                    channel_method=str(channel_method),
                    channel_lookback=int(channel_lookback),
                    channel_sigma_multiple=float(channel_sigma),
                    channel_break_buffer=float(advanced_detector["channel_break_buffer"]),
                    channel_swing_left_bars=int(advanced_detector["channel_swing_left_bars"]),
                    channel_swing_right_bars=int(advanced_detector["channel_swing_right_bars"]),
                    reversal_lookback=int(reversal_lookback),
                    reversal_strong_close_pos=float(advanced_detector["reversal_strong_close_pos"]),
                    reversal_min_body_ratio=float(advanced_detector["reversal_min_body_ratio"]),
                    reversal_old_extreme_tolerance_pct=float(reversal_old_extreme_tolerance_pct),
                    reversal_require_old_extreme_test=bool(reversal_require_old_extreme_test),
                    reversal_require_structure_confirmation=bool(reversal_require_structure_confirmation),
                ),
                save=save_outputs,
            )
            _render_backtest_result(
                experiment.backtest,
                filtered_limit_open_count=experiment.filtered_limit_open_count,
                data_coverage=experiment.data_coverage,
            )
            _render_experiment_breakdowns(experiment)
            if save_outputs:
                _show_saved_experiment_path(experiment.config.output_dir, experiment.config.name)
        return

    detectors = st.multiselect("组合 detector", ["trend", "range", "channel", "reversal"], default=["trend", "range", "channel"])
    experiment_name = _experiment_name("portfolio", timeframe, start, end)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        risk_reward = st.number_input("组合盈亏比", min_value=0.1, value=2.0, step=0.1, key="pf_rr")
    with p2:
        max_open_positions = st.number_input("最大组合持仓", min_value=1, value=5, key="pf_max_open")
    with p3:
        risk_per_trade = st.number_input("单笔风险预算", min_value=0.0, value=0.0, step=0.0025, format="%.4f", key="pf_risk")
    with p4:
        short_margin_rate = st.number_input("空头保证金倍数", min_value=0.1, value=1.0, step=0.1, key="pf_short_margin")
    alloc1, alloc2, alloc3, alloc4 = st.columns(4)
    with alloc1:
        capital_per_trade = st.number_input("固定单笔仓位", min_value=0.0, value=0.0, step=0.05, format="%.3f", key="pf_capital")
    with alloc2:
        max_capital_per_trade = st.number_input("最大单笔仓位", min_value=0.001, max_value=1.0, value=1.0, step=0.05, format="%.3f", key="pf_max_capital")
    with alloc3:
        reserve_cash = st.number_input("预留现金", min_value=0.0, max_value=0.99, value=0.0, step=0.05, format="%.3f", key="pf_reserve_cash")
    with alloc4:
        allow_same_symbol_overlap = st.checkbox("允许同票重叠", value=False, key="pf_allow_overlap")
    map1, map2 = st.columns(2)
    with map1:
        strategy_priority_text = st.text_area("策略优先级", value="", placeholder="trend_signal_bar=1,range_signal_bar=2", key="pf_strategy_priority")
        strategy_capital_limit_text = st.text_area("策略资金上限", value="", placeholder="trend_signal_bar=0.6", key="pf_strategy_limit")
    with map2:
        sector_capital_limit_text = st.text_area("行业资金上限", value="", placeholder="银行=0.5,新能源=0.4", key="pf_sector_limit")
        symbol_sector_map_text = st.text_area("股票行业映射", value="", placeholder="000001.SZ=银行,300750.SZ=新能源", key="pf_symbol_sector")
    pf_r1, pf_r2 = st.columns(2)
    with pf_r1:
        max_actual_risk_pct = st.number_input(
            "组合最大实际风险",
            min_value=0.0,
            value=0.0,
            step=0.005,
            format="%.3f",
            key="pf_max_actual_risk_pct",
        )
    with pf_r2:
        max_chase_pct = st.number_input(
            "组合最大追价距离",
            min_value=0.0,
            value=0.0,
            step=0.005,
            format="%.3f",
            key="pf_max_chase_pct",
        )
    detector_c1, detector_c2, detector_c3 = st.columns(3)
    with detector_c1:
        trend_lookback = st.number_input("组合趋势回看", min_value=3, value=20, key="pf_trend_lookback")
        channel_lookback = st.number_input("组合通道回看", min_value=3, value=40, key="pf_channel_lookback")
    with detector_c2:
        trend_min_score = st.number_input("组合趋势最低评分", min_value=0.0, value=1.0, step=0.1, key="pf_trend_score")
        channel_sigma = st.number_input("组合通道带宽倍数", min_value=0.1, value=2.0, step=0.1, key="pf_channel_sigma")
    with detector_c3:
        range_lookback = st.number_input("组合区间回看", min_value=3, value=20, key="pf_range_lookback")
        reversal_lookback = st.number_input("组合反转回看", min_value=3, value=20, key="pf_reversal_lookback")
    trend_h2_min_pullback_legs = st.number_input("组合H2/L2最少回撤腿数", min_value=1, value=2, key="pf_h2_legs")
    channel_method = st.selectbox(
        "组合通道算法",
        ["regression", "swing"],
        format_func=lambda value: "回归通道" if value == "regression" else "摆动点通道",
        key="pf_channel_method",
    )
    advanced_detector = _detector_parameter_controls("pf", "组合")
    pr1, pr2, pr3 = st.columns(3)
    with pr1:
        reversal_old_extreme_tolerance_pct = st.number_input(
            "组合反转旧极端容忍度",
            min_value=0.0,
            value=0.01,
            step=0.005,
            format="%.3f",
            key="pf_reversal_old_extreme_tolerance_pct",
        )
    with pr2:
        reversal_require_old_extreme_test = st.checkbox(
            "组合要求旧极端失败测试",
            value=True,
            key="pf_reversal_require_old_extreme_test",
        )
    with pr3:
        reversal_require_structure_confirmation = st.checkbox(
            "组合要求结构确认",
            value=True,
            key="pf_reversal_require_structure_confirmation",
        )
    save_outputs, output_dir = _experiment_output_controls("pf", experiment_name)
    if st.button("运行组合回测", type="primary"):
        experiment = run_portfolio_experiment(
            PortfolioExperimentConfig(
                name=experiment_name,
                data_root=str(data_root),
                symbols=tuple(symbols),
                timeframe=timeframe,
                higher_timeframe=str(higher_timeframe),
                higher_timeframe_max_age_minutes=higher_timeframe_max_age_minutes,
                start=start,
                end=end,
                adjust=adjust,
                detectors=tuple(detectors),
                risk_reward=float(risk_reward),
                max_holding_bars=int(max_holding),
                max_actual_risk_pct=float(max_actual_risk_pct) if float(max_actual_risk_pct) > 0 else None,
                max_chase_pct=float(max_chase_pct) if float(max_chase_pct) > 0 else None,
                max_open_positions=int(max_open_positions),
                capital_per_trade=float(capital_per_trade) if float(capital_per_trade) > 0 else None,
                risk_per_trade=float(risk_per_trade) if float(risk_per_trade) > 0 else None,
                max_capital_per_trade=float(max_capital_per_trade),
                short_margin_rate=float(short_margin_rate),
                reserve_cash=float(reserve_cash),
                allow_same_symbol_overlap=bool(allow_same_symbol_overlap),
                strategy_priority=_parse_int_mapping(str(strategy_priority_text)),
                strategy_capital_limit=_parse_float_mapping(str(strategy_capital_limit_text)),
                sector_capital_limit=_parse_float_mapping(str(sector_capital_limit_text)),
                symbol_sector_map=_parse_text_mapping(str(symbol_sector_map_text)),
                fee_rate=float(fee_rate),
                slippage_bps=float(slippage_bps),
                initial_equity=float(initial_equity),
                intrabar_exit_policy=str(intrabar_exit_policy),
                strict_data_quality=bool(strict_data_quality),
                min_coverage_ratio=min_coverage_ratio,
                output_dir=output_dir if save_outputs else "",
                trend_lookback=int(trend_lookback),
                trend_min_score=float(trend_min_score),
                trend_strong_close_pos=float(advanced_detector["trend_strong_close_pos"]),
                trend_min_body_ratio=float(advanced_detector["trend_min_body_ratio"]),
                trend_pullback_lookback=int(advanced_detector["trend_pullback_lookback"]),
                trend_h2_min_pullback_legs=int(trend_h2_min_pullback_legs),
                range_lookback=int(range_lookback),
                range_middle_low=float(advanced_detector["range_middle_low"]),
                range_middle_high=float(advanced_detector["range_middle_high"]),
                range_false_break_buffer=float(advanced_detector["range_false_break_buffer"]),
                range_strong_close_pos=float(advanced_detector["range_strong_close_pos"]),
                range_min_score=float(advanced_detector["range_min_score"]),
                channel_method=str(channel_method),
                channel_lookback=int(channel_lookback),
                channel_sigma_multiple=float(channel_sigma),
                channel_break_buffer=float(advanced_detector["channel_break_buffer"]),
                channel_swing_left_bars=int(advanced_detector["channel_swing_left_bars"]),
                channel_swing_right_bars=int(advanced_detector["channel_swing_right_bars"]),
                reversal_lookback=int(reversal_lookback),
                reversal_strong_close_pos=float(advanced_detector["reversal_strong_close_pos"]),
                reversal_min_body_ratio=float(advanced_detector["reversal_min_body_ratio"]),
                reversal_old_extreme_tolerance_pct=float(reversal_old_extreme_tolerance_pct),
                reversal_require_old_extreme_test=bool(reversal_require_old_extreme_test),
                reversal_require_structure_confirmation=bool(reversal_require_structure_confirmation),
            ),
            save=save_outputs,
        )
        _render_backtest_result(
            experiment.backtest,
            filtered_limit_open_count=experiment.filtered_limit_open_count,
            data_coverage=experiment.data_coverage,
        )
        _render_experiment_breakdowns(experiment)
        if save_outputs:
            _show_saved_experiment_path(experiment.config.output_dir, experiment.config.name)


def _render_backtest_result(
    result,
    bundle: BacktestDataBundle | None = None,
    *,
    filtered_limit_open_count: int | None = None,
    data_coverage: pd.DataFrame | None = None,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("交易数", int(result.stats["trade_count"]))
    c2.metric("胜率", f"{result.stats['win_rate']:.1%}")
    c3.metric("总收益", f"{result.stats['total_return']:.1%}")
    c4.metric("最大回撤", f"{result.stats['max_drawdown']:.1%}")
    filtered_count = len(bundle.filtered_limit_open_days) if bundle is not None else int(filtered_limit_open_count or 0)
    if filtered_count > 0:
        st.caption(f"已过滤涨停开盘交易日：{filtered_count} 条")
    if data_coverage is not None and not data_coverage.empty:
        st.dataframe(data_coverage, use_container_width=True)
    st.dataframe(result.trades, use_container_width=True)
    if not result.equity_curve.empty:
        if "date" in result.equity_curve.columns:
            st.line_chart(result.equity_curve.set_index("date")["net_value"])
            st.dataframe(result.equity_curve.tail(200), use_container_width=True)
        else:
            st.line_chart(result.equity_curve.set_index("trade_no")["net_value"])


def _render_experiment_breakdowns(experiment) -> None:
    """展示实验拆分统计；单策略和组合回测复用同一组产物。"""
    for frame in [
        experiment.strategy_stats,
        experiment.symbol_stats,
        experiment.side_stats,
        experiment.exit_reason_stats,
        experiment.event_type_stats,
        experiment.monthly_returns,
    ]:
        if not frame.empty:
            st.dataframe(frame, use_container_width=True)


def _chart_close(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    chart = frame.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
    st.line_chart(chart)


if __name__ == "__main__":
    main()
