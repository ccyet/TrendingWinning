from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from streamlit_app import _parse_float_mapping, _parse_int_mapping, _parse_text_mapping


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
    assert any(checkbox.label == "严格数据质量门禁" for checkbox in app.checkbox)
    assert any(item.label == "最低覆盖率门禁" for item in app.number_input)
    assert any(item.label == "手续费率" for item in app.number_input)
    assert any(item.label == "滑点bps" for item in app.number_input)
    assert any(item.label == "初始资金" for item in app.number_input)
    assert any(checkbox.label == "组合要求旧极端失败测试" for checkbox in app.checkbox)
    assert any(checkbox.label == "组合要求结构确认" for checkbox in app.checkbox)
    assert any(item.label == "同K止盈止损冲突" for item in app.selectbox)
    assert any(item.label == "高周期方向门控" for item in app.selectbox)
    assert any(item.label == "高周期最大过期分钟" for item in app.number_input)
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
    assert any(item.label == "单策略 detector" for item in app.selectbox)
    assert any(item.label == "高周期方向门控" for item in app.selectbox)
    assert any(item.label == "高周期最大过期分钟" for item in app.number_input)
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
    assert "已选文件夹" in source


def test_streamlit_primary_inputs_are_grouped_horizontally() -> None:
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()

    assert "scope_cols = st.columns([1, 2, 1, 1])" in source
    assert "fetch_cols = st.columns([2, 2, 1, 1, 1])" in source
    assert "scan_cols = st.columns([2, 2, 1, 1])" in source


def test_readme_usage_guide_html_exists_with_core_sections() -> None:
    html = (Path(__file__).resolve().parents[1] / "docs" / "usage_guide.html").read_text(encoding="utf-8")

    assert "TrendingWinning 使用指南" in html
    assert "路径选择" in html
    assert "单策略回测" in html
    assert "组合策略回测" in html
    assert "TDX K线" in html
    assert "inventory-data" in html
    assert "本地缓存库存" in html
    assert "data_inventory.csv" in html
    assert "monthly_win_rate" in html
    assert "周期稳定性" in html


def test_usage_docs_pin_local_parallels_tdx_test_path() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "usage_guide.html").read_text(encoding="utf-8")

    assert r"C:\new_tdx64\PYPlugins\user" in readme
    assert r"C:\\new_tdx64\\PYPlugins\\user" in guide
    assert "Mac 端 TDX 接口测试以 Parallels/Windows 通达信为准" in readme
    assert "Mac 端 TDX 接口测试以 Parallels/Windows 通达信为准" in guide
    assert "分钟 no_data" in readme
    assert "分钟 no_data" in guide
    assert "monthly_worst_return" in readme
    assert "monthly_worst_return" in guide
    assert "monthly_max_consecutive_losses" in readme
    assert "monthly_max_consecutive_losses" in guide
    assert "monthly_max_recovery_periods" in readme
    assert "monthly_max_recovery_periods" in guide
    assert "按收益、回撤、月度稳定性、交易数和 case 名稳定排序" in readme
    assert "按收益、回撤、月度稳定性、交易数和 case 名稳定排序" in guide
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
