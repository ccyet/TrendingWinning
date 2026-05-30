from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    _build_strategy_kline_altair_chart,
    _equity_chart_frame,
    _equity_y_domain,
    _format_display_value,
    _parse_float_mapping,
    _parse_int_mapping,
    _parse_text_mapping,
    _prepare_display_frame,
    _resolve_native_directory_choice,
    _style_display_frame,
    _strategy_kline_chart_frame,
    _strategy_kline_symbol_options,
    _strategy_stop_segment_frame,
    _strategy_trade_marker_frame,
)


def test_streamlit_app_renders_without_widget_id_conflicts() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"))

    app.run(timeout=5)

    assert not app.exception
    assert app.title[0].value == "TrendingWinning"


def test_streamlit_app_exposes_tdx_prepare_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"))

    app.run(timeout=5)

    assert not app.exception
    assert any(item.label == "补齐最低覆盖率" for item in app.number_input)
    assert any(button.label == "查看本地缓存库存" for button in app.button)
    assert any(button.label == "生成TDX补齐计划" for button in app.button)
    assert any(button.label == "审计并补齐TDX数据" for button in app.button)


def test_streamlit_data_prepare_includes_daily_without_enabling_daily_strategy_timeframes() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"))

    app.run(timeout=5)

    assert not app.exception
    fetch_timeframe = app.multiselect[0]
    scan_timeframe = next(item for item in app.multiselect if item.key == "scan_timeframes")
    backtest_timeframe = next(item for item in app.selectbox if item.key == "bt_tf")

    assert fetch_timeframe.options == ["1d", "5m", "15m", "30m", "60m"]
    assert "1d" in fetch_timeframe.value
    assert scan_timeframe.options == ["5m", "15m", "30m", "60m"]
    assert backtest_timeframe.options == ["5m", "15m", "30m", "60m"]


def test_streamlit_app_exposes_portfolio_backtest_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"))

    app.run(timeout=5)

    assert not app.exception
    assert any(item.label == "回测模式" for item in app.radio)
    app.radio[0].set_value("组合策略回测").run(timeout=5)
    assert any(button.label == "运行组合回测" for button in app.button)
    assert any(checkbox.label == "严格数据质量检查" for checkbox in app.checkbox)
    assert any(item.label == "最低K线覆盖率" for item in app.number_input)
    assert any(item.label == "手续费率" for item in app.number_input)
    assert any(item.label == "滑点bps" for item in app.number_input)
    assert any(item.label == "初始资金" for item in app.number_input)
    assert any(checkbox.label == "组合要求旧极端失败测试" for checkbox in app.checkbox)
    assert any(checkbox.label == "组合要求结构确认" for checkbox in app.checkbox)
    assert any(item.label == "同K止盈止损冲突" for item in app.selectbox)
    assert any(item.label == "大周期方向过滤" for item in app.selectbox)
    assert any(item.label == "大周期信号有效分钟" for item in app.number_input)
    assert any(item.label == "组合反转旧极端容忍度" for item in app.number_input)
    assert any(item.label == "组合最大实际风险" for item in app.number_input)
    assert any(item.label == "组合最大追价距离" for item in app.number_input)
    assert any(item.label == "组合趋势回看" for item in app.number_input)
    assert any(item.label == "组合趋势最低评分" for item in app.number_input)
    assert any(item.label == "组合趋势强收盘" for item in app.number_input)
    assert any(item.label == "组合趋势最小实体" for item in app.number_input)
    assert any(item.label == "组合趋势回撤窗口" for item in app.number_input)
    assert any(item.label == "组合区间回看" for item in app.number_input)
    assert any(item.label == "组合区间中部下沿" for item in app.number_input)
    assert any(item.label == "组合区间中部上沿" for item in app.number_input)
    assert any(item.label == "组合区间失败突破缓冲" for item in app.number_input)
    assert any(item.label == "组合区间强收盘" for item in app.number_input)
    assert any(item.label == "组合区间最低评分" for item in app.number_input)
    assert any(item.label == "组合通道回看" for item in app.number_input)
    assert any(item.label == "组合通道带宽倍数" for item in app.number_input)
    assert any(item.label == "组合通道突破缓冲" for item in app.number_input)
    assert any(item.label == "组合摆动左侧K数" for item in app.number_input)
    assert any(item.label == "组合摆动右侧K数" for item in app.number_input)
    assert any(item.label == "组合反转回看" for item in app.number_input)
    assert any(item.label == "组合反转强收盘" for item in app.number_input)
    assert any(item.label == "组合反转最小实体" for item in app.number_input)
    assert any(item.label == "固定单笔仓位" for item in app.number_input)
    assert any(item.label == "最大单笔仓位" for item in app.number_input)
    assert any(item.label == "预留现金" for item in app.number_input)
    assert any(checkbox.label == "允许同票重叠" for checkbox in app.checkbox)
    assert any(item.label == "策略优先级" for item in app.text_area)
    assert any(item.label == "策略资金上限" for item in app.text_area)
    assert any(item.label == "行业资金上限" for item in app.text_area)
    assert any(item.label == "股票行业映射" for item in app.text_area)
    assert any(checkbox.label == "保存实验产物" for checkbox in app.checkbox)
    assert any(item.label == "输出父目录子文件夹" for item in app.selectbox)


def test_streamlit_app_exposes_single_strategy_backtest_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"))

    app.run(timeout=5)

    assert not app.exception
    app.radio[0].set_value("单策略回测").run(timeout=5)
    assert any(item.label == "单策略形态" for item in app.selectbox)
    assert any(item.label == "大周期方向过滤" for item in app.selectbox)
    assert any(item.label == "大周期信号有效分钟" for item in app.number_input)
    assert any(item.label == "最大实际风险" for item in app.number_input)
    assert any(item.label == "最大追价距离" for item in app.number_input)
    assert any(item.label == "趋势强收盘" for item in app.number_input)
    assert any(item.label == "趋势最小实体" for item in app.number_input)
    assert any(item.label == "趋势回撤窗口" for item in app.number_input)
    assert any(item.label == "区间中部下沿" for item in app.number_input)
    assert any(item.label == "区间中部上沿" for item in app.number_input)
    assert any(item.label == "区间失败突破缓冲" for item in app.number_input)
    assert any(item.label == "区间强收盘" for item in app.number_input)
    assert any(item.label == "区间最低评分" for item in app.number_input)
    assert any(item.label == "通道突破缓冲" for item in app.number_input)
    assert any(item.label == "摆动左侧K数" for item in app.number_input)
    assert any(item.label == "摆动右侧K数" for item in app.number_input)
    assert any(item.label == "反转强收盘" for item in app.number_input)
    assert any(item.label == "反转最小实体" for item in app.number_input)
    assert any(item.label == "反转旧极端容忍度" for item in app.number_input)
    assert any(checkbox.label == "要求旧极端失败测试" for checkbox in app.checkbox)
    assert any(checkbox.label == "要求结构确认" for checkbox in app.checkbox)
    assert any(checkbox.label == "保存实验产物" for checkbox in app.checkbox)
    assert any(item.label == "输出父目录子文件夹" for item in app.selectbox)
    assert any(button.label == "运行单策略回测" for button in app.button)


def test_streamlit_path_controls_use_folder_picker_instead_of_text_inputs() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert 'st.text_input("行情根目录"' not in source
    assert 'st.text_input("TDX PYPlugins/user"' not in source
    assert 'st.text_input(\n            "输出目录"' not in source
    assert "_directory_picker(" in source
    assert "_open_native_directory_dialog(" in source
    assert "选择文件夹" in source
    assert "已选文件夹" in source


def test_streamlit_path_controls_render_native_folder_buttons() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "streamlit_app.py"))

    app.run(timeout=5)

    assert not app.exception
    assert any(button.label == "选择文件夹" for button in app.button)


def test_native_directory_choice_uses_existing_parent_and_handles_cancel(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    selected_path = tmp_path / "selected"

    def askdirectory(**kwargs: object) -> str:
        calls.update(kwargs)
        return str(selected_path)

    selected = _resolve_native_directory_choice(tmp_path / "missing" / "daily", "选择目录", askdirectory)

    assert selected == selected_path
    assert calls["title"] == "选择目录"
    assert calls["initialdir"] == str(tmp_path)
    assert calls["mustexist"] is False
    assert _resolve_native_directory_choice(tmp_path, "选择目录", lambda **_: "") is None


def test_streamlit_primary_inputs_are_grouped_horizontally() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert "scope_cols = st.columns([1, 2, 1, 1])" in source
    assert "fetch_cols = st.columns([2, 2, 1, 1, 1])" in source
    assert "scan_cols = st.columns([2, 2, 1, 1])" in source


def test_streamlit_backtest_interface_is_split_into_functional_modules() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    for helper in [
        "_backtest_scope_module(",
        "_backtest_risk_module(",
        "_backtest_data_quality_module(",
        "_backtest_higher_timeframe_module(",
        "_single_strategy_module(",
        "_portfolio_allocation_module(",
        "_portfolio_detector_module(",
        "_backtest_output_module(",
    ]:
        assert helper in source

    for title in [
        "1. 样本范围",
        "2. 基础风控与成本",
        "3. 数据质量检查",
        "4. 大周期方向过滤",
        "5. 单策略参数",
        "5. 组合仓位与资金",
        "6. 保存与运行",
    ]:
        assert title in source


def test_streamlit_backtest_parameters_have_hover_help_text() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    for help_key in [
        "scope_mode",
        "take_profit",
        "strict_data_quality",
        "higher_timeframe",
        "trend_h2_min_pullback_legs",
        "range_middle_low",
        "channel_sigma",
        "reversal_old_extreme_tolerance_pct",
        "risk_per_trade",
        "save_outputs",
        "output_parent",
    ]:
        assert f'BACKTEST_HELP_TEXT["{help_key}"]' in source


def test_backtest_display_tables_are_localized_and_formatted() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "600519.SH"],
            "case_count": [2, 10],
            "pareto_hit_rate": [0.25, 1.0],
            "positive_return_rate": [0.5, 0.9],
            "std_total_return": [0.0312, 0.12],
            "best_total_return": [0.2, 1.0],
            "worst_total_return": [-0.1, 0.2],
            "trade_count": [3, 4],
            "win_rate": [0.096, 1.0],
            "total_return": [0.096, 1.0],
            "return_pct": [9.6, -0.5],
            "positive_expectancy_probability": [0.096, 1.0],
            "avg_return_standard_error": [0.0123, 0.0],
            "win_rate_ci_lower": [0.0432, 0.9],
            "avg_holding_bars": [2.45, 3.2],
        }
    )

    display = _prepare_display_frame(frame)
    custom_display = _prepare_display_frame(frame, stock_names={"000001.SZ": "自定义银行"})

    assert "stock_code" not in display.columns
    assert display["股票名称"].tolist() == ["平安银行", "贵州茅台"]
    assert custom_display["股票名称"].tolist()[0] == "自定义银行"
    assert display["参数组数"].tolist() == ["2", "10"]
    assert display["Pareto命中率"].tolist() == ["25.00%", "100.00%"]
    assert display["正收益率"].tolist() == ["50.00%", "90.00%"]
    assert display["总收益标准差"].tolist() == ["3.12%", "12.00%"]
    assert display["最好总收益"].tolist() == ["20.00%", "100.00%"]
    assert display["最差总收益"].tolist() == ["-10.00%", "20.00%"]
    assert display["交易次数"].tolist() == ["3", "4"]
    assert display["胜率"].tolist() == ["9.60%", "100.00%"]
    assert display["总收益"].tolist() == ["9.60%", "100.00%"]
    assert display["收益率"].tolist() == ["9.60%", "-0.50%"]
    assert display["正期望概率"].tolist() == ["9.60%", "100.00%"]
    assert display["平均收益标准误"].tolist() == ["1.23%", "0.00%"]
    assert display["胜率95%下限"].tolist() == ["4.32%", "90.00%"]
    assert display["平均持有K数"].tolist() == ["2.45", "3.20"]
    assert _format_display_value("max_drawdown", 0.096) == "9.60%"


def test_backtest_display_table_style_centers_cells_and_has_grid_lines() -> None:
    styled = _style_display_frame(pd.DataFrame({"股票名称": ["平安银行"], "胜率": ["9.60%"]}))
    html = styled.to_html()

    assert "text-align: center" in html
    assert "border-bottom" in html
    assert "background-color" in html


def test_backtest_equity_chart_domain_is_anchored_at_one() -> None:
    assert _equity_y_domain(pd.Series([1.04, 1.10]))[0] == 1.0

    lower, upper = _equity_y_domain(pd.Series([0.96, 1.08]))

    assert lower < 1.0
    assert upper > 1.08


def test_backtest_equity_chart_uses_relative_net_value_starting_at_one() -> None:
    equity = pd.DataFrame({"trade_no": [0, 1, 2], "net_value": [2.0, 2.2, 1.98]})

    chart = _equity_chart_frame(equity)

    assert chart["净值比例"].tolist() == [1.0, 1.1, 0.99]
    assert chart["交易序号"].tolist() == [0, 1, 2]


def test_strategy_kline_chart_frame_keeps_full_symbol_backtest_window() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00", "2026-05-25 10:30", "2026-05-25 11:00"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2, 10.1],
            "high": [10.4, 10.5, 10.3],
            "low": [9.9, 10.0, 9.8],
            "close": [10.2, 10.1, 10.0],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [10200.0, 11110.0, 12000.0],
        }
    )

    chart = _strategy_kline_chart_frame(bars, "000001.SZ")

    assert chart["时间"].tolist() == bars["date"].tolist()
    assert chart["开盘"].tolist() == [10.0, 10.2, 10.1]
    assert chart["收盘"].tolist() == [10.2, 10.1, 10.0]
    assert chart["涨跌"].tolist() == ["上涨", "下跌", "下跌"]


def test_strategy_trade_markers_include_long_short_entries_and_stop_loss() -> None:
    trades = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ"],
            "side": ["long", "short"],
            "entry_date": pd.to_datetime(["2026-05-25 10:30", "2026-05-25 11:00"]),
            "entry_price": [10.5, 10.1],
            "stop_price": [10.0, 10.6],
            "exit_date": pd.to_datetime(["2026-05-25 11:30", "2026-05-25 14:00"]),
            "exit_price": [10.0, 9.8],
            "exit_reason": ["stop_loss", "take_profit"],
        }
    )

    markers = _strategy_trade_marker_frame(trades, "000001.SZ")
    stops = _strategy_stop_segment_frame(trades, "000001.SZ")

    assert markers["标注"].tolist() == ["开多", "开空", "止损"]
    assert markers["价格"].tolist() == [10.5, 10.1, 10.0]
    assert stops["止损价"].tolist() == [10.0, 10.6]
    assert stops["开始时间"].tolist() == trades["entry_date"].tolist()
    assert stops["结束时间"].tolist() == trades["exit_date"].tolist()


def test_strategy_kline_altair_chart_contains_candles_entries_and_stop_layers() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00", "2026-05-25 10:30"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.4, 10.5],
            "low": [9.9, 10.0],
            "close": [10.2, 10.1],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11110.0],
        }
    )
    trades = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"],
            "side": ["long"],
            "entry_date": pd.to_datetime(["2026-05-25 10:30"]),
            "entry_price": [10.5],
            "stop_price": [10.0],
            "exit_date": pd.to_datetime(["2026-05-25 11:00"]),
            "exit_price": [10.0],
            "exit_reason": ["stop_loss"],
        }
    )

    chart = _build_strategy_kline_altair_chart(
        _strategy_kline_chart_frame(bars, "000001.SZ"),
        _strategy_trade_marker_frame(trades, "000001.SZ"),
        _strategy_stop_segment_frame(trades, "000001.SZ"),
    )
    spec = chart.to_dict()

    assert spec["height"] == 420
    assert len(spec["layer"]) == 5
    assert "开多" in str(spec)
    assert "止损" in str(spec)


def test_strategy_kline_symbol_options_prioritize_symbols_with_trades() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00"] * 3),
            "stock_code": ["000001.SZ", "600519.SH", "300750.SZ"],
            "open": [10.0, 20.0, 30.0],
            "high": [10.2, 20.2, 30.2],
            "low": [9.9, 19.9, 29.9],
            "close": [10.1, 20.1, 30.1],
            "volume": [1000.0, 1000.0, 1000.0],
            "amount": [10100.0, 20100.0, 30100.0],
        }
    )
    trades = pd.DataFrame({"stock_code": ["300750.SZ", "000001.SZ"]})

    assert _strategy_kline_symbol_options(bars, trades) == ["300750.SZ", "000001.SZ", "600519.SH"]


def test_readme_usage_guide_html_exists_with_core_sections() -> None:
    html = (Path(__file__).resolve().parents[1] / "docs" / "usage_guide.html").read_text(encoding="utf-8")

    assert "TrendingWinning 使用指南" in html
    assert "backtest_kline_guide.html" in html
    assert "路径选择" in html
    assert "单策略回测" in html
    assert "组合策略回测" in html
    assert "TDX K线" in html
    assert "inventory-data" in html
    assert "本地缓存库存" in html
    assert "data_inventory.csv" in html
    assert "symbol_metadata.csv" in html
    assert "monthly_win_rate" in html
    assert "周期稳定性" in html
    assert "策略K线运行区间" in html


def test_backtest_kline_guide_html_exists_with_examples_and_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    html = (root / "docs" / "backtest_kline_guide.html").read_text(encoding="utf-8")

    assert "docs/backtest_kline_guide.html" in readme
    assert "回测界面 K 线使用说明" in html
    assert "术语对照" in html
    assert "趋势回撤：H2 顺势做多" in html
    assert "下降趋势：L2 顺势做空" in html
    assert "H1/H2/L1/L2" in html
    assert "H 是 High 1/High 2" in html
    assert "L 是 Low 1/Low 2" in html
    assert "不是单根 K 线" in html
    assert "H 不是 high 的简称" not in html
    assert "H2 多头二次入场" in html
    assert "L2 空头二次入场" in html
    assert "交易区间下沿：失败突破做多" in html
    assert "交易区间上沿：失败突破做空" in html
    assert "通道突破：顺势延续" in html
    assert "主要反转：第二次信号才切换" in html
    assert "策略K线运行区间" in html
    assert "开多、开空、止损标注" in html
    assert html.count("<svg") >= 6
    assert "门禁" not in html
    assert "高周期门控" not in html
    for title in [
        "1. 样本范围",
        "2. 基础风控与成本",
        "3. 数据质量检查",
        "4. 大周期方向过滤",
        "5. 单策略参数",
        "5. 组合仓位与资金",
        "6. 保存与运行",
    ]:
        assert title in html


def test_usage_docs_pin_local_parallels_tdx_test_path() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "usage_guide.html").read_text(encoding="utf-8")

    assert "选择文件夹" in readme
    assert "选择文件夹" in guide
    assert "无法打开系统选择框" in guide
    assert r"C:\new_tdx64\PYPlugins\user" in readme
    assert r"C:\\new_tdx64\\PYPlugins\\user" in guide
    assert "Mac 端 TDX 接口测试以 Parallels/Windows 通达信为准" in readme
    assert "Mac 端 TDX 接口测试以 Parallels/Windows 通达信为准" in guide
    assert "分钟 no_data" in readme
    assert "分钟 no_data" in guide
    assert "monthly_worst_return" in readme
    assert "monthly_worst_return" in guide
    assert "symbol_metadata.csv" in readme
    assert "symbol_metadata.csv" in guide
    assert "monthly_max_consecutive_losses" in readme
    assert "monthly_max_consecutive_losses" in guide
    assert "monthly_max_recovery_periods" in readme
    assert "monthly_max_recovery_periods" in guide
    assert "按收益、回撤、月度稳定性、交易数和 case 名稳定排序" in readme
    assert "按收益、回撤、月度稳定性、交易数和 case 名稳定排序" in guide
    assert "data_coverage_p05" in readme
    assert "data_coverage_p05" in guide
    assert "data_coverage_below_min_count" in readme
    assert "data_coverage_below_min_count" in guide
    assert "sweep_rank" in readme
    assert "sweep_rank" in guide
    assert "pareto_rank" in readme
    assert "pareto_rank" in guide
    assert "case_config_hash" in readme
    assert "case_config_hash" in guide
    assert "case_configs.jsonl" in readme
    assert "case_configs.jsonl" in guide
    assert "replay-case" in readme
    assert "replay-case" in guide
    assert "拒绝回放" in readme
    assert "拒绝回放" in guide
    assert r"C:\new_tdx\T0002\PYPlugins\user" not in readme
    assert r"C:\\new_tdx\\T0002\\PYPlugins\\user" not in guide


def test_streamlit_mapping_inputs_accept_comma_and_newline_pairs() -> None:
    assert _parse_int_mapping("trend_signal_bar=1\nrange_signal_bar=2") == {
        "trend_signal_bar": 1,
        "range_signal_bar": 2,
    }
    assert _parse_float_mapping("银行=0.5, 新能源=0.4") == {"银行": 0.5, "新能源": 0.4}
    assert _parse_text_mapping("000001.SZ=银行\n300750.SZ=新能源") == {
        "000001.SZ": "银行",
        "300750.SZ": "新能源",
    }


def test_streamlit_modern_backtests_use_experiment_runners() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert "run_single_strategy_experiment(" in source
    assert "run_portfolio_experiment(" in source
    assert "run_single_strategy_backtest(" not in source
    assert "run_portfolio_backtest(" not in source


def test_streamlit_app_surfaces_full_experiment_breakdowns_once() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert "experiment.side_stats" in source
    assert "experiment.exit_reason_stats" in source
    assert "experiment.event_type_stats" in source
    assert source.count('metric("胜率"') == 1


def test_streamlit_app_passes_advanced_detector_parameters_to_experiments() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    for field in [
        "trend_strong_close_pos",
        "trend_min_body_ratio",
        "trend_pullback_lookback",
        "range_middle_low",
        "range_middle_high",
        "range_false_break_buffer",
        "range_strong_close_pos",
        "range_min_score",
        "channel_break_buffer",
        "channel_swing_left_bars",
        "channel_swing_right_bars",
        "reversal_strong_close_pos",
        "reversal_min_body_ratio",
    ]:
        assert source.count(f"{field}=") >= 2
