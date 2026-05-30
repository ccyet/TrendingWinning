# TrendingWinning

TDX-only A 股 K 线趋势策略工作台。

精简版 HTML 使用指南见 `docs/usage_guide.html`；回测界面和 K 线范例见 `docs/backtest_kline_guide.html`，均可直接用浏览器打开。

首版能力：

- TDX `tqcenter` K 线：日 K `1d` 和分钟 K `5m / 15m / 30m / 60m`
- TDX 分钟兜底：`15m / 30m / 60m` 原生周期无数据时，自动用 TDX `5m` 聚合生成目标周期 K 线
- 本地 parquet 落地：兼容 `trend-backtest/data/market/<timeframe>/<adjust>/<symbol>.parquet`，其中 `1d` 写入既有 `market/daily/<adjust>` 目录
- 本地缓存库存：按 `1d / 5m / 15m / 30m / 60m` 列出 parquet 是否存在、行数、起止时间、文件大小和状态
- 标志K识别：振幅、量能、实体比例参数化
- 趋势通道识别：支持批量滚动 log 回归通道和摆动点趋势通道，上下轨、斜率、R²、方向和锚点可追溯
- 突破 trigger：基于上一根已完成通道上轨，避免把当前突破 K 纳入边界
- 多周期扫描：一次聚合 `5m / 15m / 30m / 60m` 的最新通道和突破状态
- 大周期方向过滤：用 60m/30m 判断主方向，只过滤低周期逆势订单，不改趋势/区间/通道/反转识别结果
- 数据审计：本地 parquet 覆盖、日期/标的代码、字段质量、OHLC 合法性、重复 K、交易时段内缺失 K、缺口首尾时间、最长连续缺口边界、零流动性 K 和覆盖率显式输出
- 独立形态识别策略：趋势、区间、通道、反转模块解耦，单策略回测只消费本策略事件
- Detector 事件契约：非空事件表必须包含 `event_id / detector_name / stock_code / timeframe / date / bar_index / event_type / direction / signal_price / entry_price / stop_price / confidence / metadata`，策略层统一校验，缺字段会直接报错
- 订单契约：撮合层要求 `order_id / event_id / stock_code / signal_date / signal_bar_index / side / entry_price / stop_price / target_price` 必备，组合回测额外要求 `strategy_name`；空 `order_id` 或 `event_id` 记为 `invalid_order`，重复 `order_id` 记为 `duplicate_order_id`
- 高性能订单链路：信号 K 策略用列运算把 detector 事件转换为挂单，撮合层用排序后的记录流处理订单和候选成交，避免在参数遍历热路径逐行创建 pandas 行对象
- 趋势回撤事件：`TrendDetector` 输出 `trend_state / pullback_legs`，并把顺势回撤信号区分为 `bull_h1_setup / bull_h2_setup / bear_l1_setup / bear_l2_setup`
- 区间识别评分：`RangeDetector` 输出 `range_score / overlap_mean / ema_flatness / directional_efficiency`，过滤强趋势里的中部噪声
- 反转确认：`ReversalDetector` 默认第一次反转只观察，第二次反转必须满足旧极端失败测试和结构确认后才输出交易事件
- 组合回测：按真实入场时间和真实成交风险做策略优先级、持仓互斥、风险预算、行业/策略资金上限、空头保证金和逐 K 净值重估
- 实际风险过滤：信号 K 挂单策略透传 `max_actual_risk_pct` 和 `max_chase_pct`，由撮合层按真实成交价统一判定并记录拒绝原因
- Detector 参数透传：趋势强收盘/实体/回撤窗口、区间中部/失败突破/区间评分、通道突破缓冲/摆动锚点、反转强收盘/实体阈值都可配置
- 回测统计：逐笔统计与净值曲线统计分离，输出总收益、最大回撤、胜率置信区间、平均收益标准误、正期望概率、平均回撤、Ulcer Index、水下时间比例、年化收益、年化波动、Calmar、现金占比、净暴露、总暴露、持仓数和策略/标的/方向/退出原因拆分
- 事件类型拆分：订单和成交透传 `event_type`，可单独评估 H1/H2、失败突破、通道突破、二次反转等 setup 表现
- 真实撮合边界：跳空穿越按开盘成交；同 K 同时触发止盈止损时默认保守止损优先，可显式改为乐观止盈优先
- 流动性检查：分钟 K 的 `volume` 或 `amount` 为 0 时会从回测数据包剔除；裸策略/撮合入口仍不允许用这类 K 入场、止盈止损或统计路径波动
- 回测数据过滤：按主板/科创/创业/BJ 的日 K 涨停开盘规则剔除污染交易日
- 多周期数据包：`5m / 15m / 30m / 60m` 可统一审计、统一日线过滤、按周期拆分给扫描和回测
- Streamlit Web 应用，后续可挂 Docker 数据卷

## 本地运行

```bash
cd /Users/a1234/Documents/TrendingWinning
python -m pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8520
```

TDX 真取数只支持 Windows/Parallels 内的通达信。Mac 本机通达信不支持 `tqcenter` 取数，Mac 端 CLI 默认用 `--runtime auto` 调度到 Parallels；Windows 侧运行时默认本地执行。`60m` 会按 TDX 接口要求映射为 `1h` 请求；`15m / 30m / 60m` 如果原生周期无数据，会自动回退到 TDX `5m` 并按 A 股上午、下午交易段聚合。分钟线能否返回数据仍取决于 Windows 通达信本地是否已有对应 5m 数据。

Mac 端 TDX 接口测试以 Parallels/Windows 通达信为准，先跑 `tdx-doctor --runtime parallels`，不要尝试从 Mac 本机通达信直接取数。若日 K 返回 `ok`、分钟 no_data，说明 Parallels/Windows 通达信本地没有返回分钟 K 线；先在 Windows 通达信内确认 5m 分钟数据缓存，再做 `15m / 30m / 60m` 原生请求或 5m 聚合。

Parallels 默认配置：

- VM：`Windows 11`，可用 `--parallels-vm` 或 `TDX_PARALLELS_VM` 覆盖
- Windows Python：`C:\Users\Public\venvs\trending-winning\Scripts\python.exe`，可用 `--windows-python` 或 `TDX_PARALLELS_PYTHON` 覆盖
- Windows 仓库路径：默认把 Mac 当前仓库映射为 `C:\Mac\Home\...`，可用 `--windows-repo` 或 `TDX_PARALLELS_REPO` 覆盖
- TDX 插件目录：用 Windows 路径传给 `--tdx-path`，本机 Parallels 实测目录为 `C:\new_tdx64\PYPlugins\user`

Web 回测页里的“单策略回测”和“组合策略回测”与 CLI 复用同一套实验运行器。
Web 页面所有文件夹路径都通过“选择文件夹”按钮弹出系统选择框；若本机窗口环境不可用，页面会明确提示并保留站内目录浏览器。
单策略只绑定一种形态识别模块，不进入组合仓位分配层；组合策略才启用策略优先级、资金上限、行业上限和持仓互斥。
需要 60m 判主方向、15m/5m 触发时，可用 `HigherTimeframeAlignmentStrategy` 包装任一基础策略；它按订单 `signal_date` 向前匹配大周期上下文，拒绝方向不一致或上下文过旧的订单，基础形态识别仍保持独立，拒绝原因会单独写入策略层过滤日志。
“高级形态识别参数”在 Web 里按单策略和组合策略分开配置，可单独调整趋势、区间、通道、反转识别阈值。
勾选“保存实验产物”后会把 `config.json`、`stats.json`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`setup_order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`setup_strategy_filter_stats.csv`、`equity_curve.csv`、`data_inventory.csv`、`symbol_metadata.csv` 和数据审计文件写入页面指定的输出目录。
“TDX K线”页支持“查看本地缓存库存”和“审计并补齐TDX数据”：先确认本地 parquet 已有哪些周期和标的，再只对缺失、质量错误或覆盖率低于门槛的标的周期请求 TDX。

## CLI

抓取并写入本地 parquet：

```bash
python -m trending_winning.cli tdx-doctor \
  --symbols 000001.SZ,600519.SH \
  --timeframes 1d,5m,15m,30m,60m \
  --start "2026-05-25 09:30:00" \
  --end "2026-05-25 15:00:00" \
  --runtime parallels \
  --tdx-path "C:\\new_tdx64\\PYPlugins\\user"

python -m trending_winning.cli fetch \
  --symbols 000001.SZ,600519.SH \
  --timeframe 1d \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily
```

`tdx-doctor` 不写本地文件，只诊断 TDX 通道：逐周期输出 `ok / no_data / request_error / init_error`、TDX 实际 period、返回行数、样本起止时间和错误信息。Mac 端请用 Parallels runtime 让命令进 Windows VM；先确认 Windows 通达信已启动登录、`PYPlugins/user` 路径有效、`5m/15m/30m/60m/1d` 都能返回样本，再执行补数和回测。
诊断日 K 时会自动按整天取样，即使 `start` 写成 `09:30:00` 也不会把当天日线排除。

审计并按需补齐多周期本地 parquet：

```bash
python -m trending_winning.cli inventory-data \
  --symbols 000001.SZ,600519.SH \
  --timeframes 1d,5m,15m,30m,60m \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily

python -m trending_winning.cli plan-data \
  --symbols 000001.SZ,600519.SH \
  --timeframes 1d,5m,15m,30m,60m \
  --start "2026-05-01 09:30:00" \
  --end "2026-05-25 15:00:00" \
  --min-coverage-ratio 0.95 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily

python -m trending_winning.cli prepare-data \
  --symbols 000001.SZ,600519.SH \
  --timeframes 1d,5m,15m,30m,60m \
  --start "2026-05-01 09:30:00" \
  --end "2026-05-25 15:00:00" \
  --min-coverage-ratio 0.95 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily
```

`inventory-data` 只列本地缓存库存，不做覆盖率判断；指定 `--symbols` 时会把缺失的 `stock_code/timeframe` 标成 `missing_file`，不指定时会按已有 parquet 自动发现代码。库存扫描只读取 parquet 元数据和 `date/stock_code` 两列，不会为了看行数和起止时间加载完整 OHLCV。
`plan-data` 只读取本地 parquet 和日 K 交易日锚点，不请求 TDX；它会把每个 `stock_code/timeframe` 标成 `cached` 或 `fetch`，并写明 `missing_file / quality_error / coverage_below_min / local_ok`。
`prepare-data` 会输出每个 `stock_code/timeframe` 的 `cached/fetched` 动作、补数前后状态、写入行数、补前/补后覆盖率、补前/补后缺失 K 数、补前/补后最长连续缺口、补前/补后全局缺口首尾时间、补前/补后最长连续缺口起止时间和本地 parquet 路径；默认补完后仍不达标会直接失败。
`plan-data` 和 `prepare-data` 遇到分钟周期会自动把 `1d` 作为依赖先审计/补齐；补完日 K 后再用它锚定分钟线交易日覆盖率和一字涨停开盘过滤。
即使 `start` 写成 `2026-05-01 09:30:00`，日 K 请求和审计也会覆盖 5 月 1 日整天，不会因为日 K 时间戳是 00:00 而漏掉首日。
所有本地读取、审计、补数计划和回测数据包入口都会校验 `start <= end`；时间窗口写反会直接失败，不会静默返回空行情。
TDX 原始取数和 `tdx-doctor` 也会在初始化通达信客户端前校验 `start <= end`，避免错误参数触发真实行情请求。

回测前检查本地数据覆盖和质量：

```bash
python -m trending_winning.cli audit-data \
  --symbols 000001.SZ,600519.SH \
  --timeframe 30m \
  --higher-timeframe 60m \
  --higher-timeframe-max-age-minutes 120 \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily
```

组合回测默认启用严格数据质量检查：本地 parquet 缺失、请求窗口无可交易数据、缺字段、日期无法解析、标的代码为空或无法规范化、重复 K、OHLC 为空、价格非正、high/low 与 open/close 不一致、volume/amount 为空或负数，以及日 K 缺失导致一字涨停过滤无法执行，都会直接失败。`volume` 或 `amount` 为 0 会记录为 `zero_volume_amount_rows`，不直接记为字段质量错误，但覆盖率按剔除这些 K 后的可交易 K 计算，回测数据包也会先剔除这些 K；裸策略/撮合入口仍不会用这类 K 入场、止盈止损或统计路径波动。
临时排查时可以在 CLI 里加 `--allow-bad-data`，页面里取消“严格数据质量检查”；此时 `read_error / missing_columns` 等无法装载的 parquet 会保留在审计表里并从回测数据包跳过，不会被当成有效 K 线参与交易。
多周期扫描会复用同一套数据质量检查和涨停开盘过滤，不再直接绕过本地行情审计。
`audit-data` 会同时输出 `invalid_date_rows / invalid_symbol_rows / zero_volume_amount_rows / expected_rows / missing_rows / coverage_ratio / max_missing_gap_minutes / first_missing_at / last_missing_at / max_missing_gap_start_at / max_missing_gap_end_at`；其中 `rows_in_window / missing_rows / coverage_ratio` 按剔除零流动性后的可交易 K 计算，`zero_volume_amount_rows` 单独保留原始零流动性数量。
直接审计和回测加载都会优先用本地日 K 作为交易日锚点计算分钟线覆盖度，可暴露整天分钟 K 缺失；日 K 不存在时才退回按已观测分钟交易日计算。
需要把覆盖度也变成硬性检查时，在单策略、组合或参数遍历命令里加 `--min-coverage-ratio 0.95`。

扫描并回测：

```bash
python -m trending_winning.cli backtest \
  --symbols 000001.SZ,600519.SH \
  --timeframe 60m \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily
```

单策略回测只绑定一种形态识别模块，不进入组合仓位分配层：

```bash
python -m trending_winning.cli single-backtest \
  --symbols 000001.SZ,600519.SH \
  --timeframe 30m \
  --higher-timeframe 60m \
  --higher-timeframe-max-age-minutes 120 \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --detector trend \
  --risk-reward 2.0 \
  --max-holding-bars 12 \
  --max-actual-risk-pct 0.03 \
  --max-chase-pct 0.02 \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --trend-lookback 20 \
  --trend-min-score 1.0 \
  --trend-strong-close-pos 0.65 \
  --trend-min-body-ratio 0.45 \
  --trend-pullback-lookback 5 \
  --trend-h2-min-pullback-legs 2 \
  --range-lookback 20 \
  --range-middle-low 0.25 \
  --range-middle-high 0.75 \
  --range-false-break-buffer 0 \
  --range-strong-close-pos 0.65 \
  --range-min-score 0.8 \
  --channel-method regression \
  --channel-lookback 40 \
  --channel-sigma-multiple 2.0 \
  --channel-break-buffer 0 \
  --channel-swing-left-bars 2 \
  --channel-swing-right-bars 2 \
  --reversal-lookback 20 \
  --reversal-strong-close-pos 0.65 \
  --reversal-min-body-ratio 0.45 \
  --reversal-old-extreme-tolerance-pct 0.01 \
  --intrabar-exit-policy conservative \
  --output-dir runs/single-trend-001
```

保存目录只包含单策略产物：`config.json`、`stats.json`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`setup_order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`setup_strategy_filter_stats.csv`、`equity_curve.csv`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv`、`limit_filter_audit.csv`、
`strategy_stats.csv`、`detector_stats.csv`、`setup_stats.csv`、`symbol_stats.csv`、`side_stats.csv`、`exit_reason_stats.csv`、`event_type_stats.csv`、`monthly_returns.csv`。
`monthly_returns.csv` 的周期收益以上一条净值作为本期起点，避免漏掉月初第一笔净值变化；同时包含周期内最大回撤和净值样本数。
`stats.json` 会同步写入周期稳定性摘要，例如 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_return_std / monthly_worst_drawdown / monthly_max_consecutive_losses / monthly_max_recovery_periods`，避免参数对比时再手工汇总月度收益和连续亏损风险。逐笔交易统计还包含 `win_rate_ci_lower / win_rate_ci_upper / avg_return_standard_error / avg_return_ci_lower / avg_return_ci_upper / positive_expectancy_probability`，用于判断胜率和平均收益是否只是小样本波动。
单策略 `equity_curve.csv` 从 `trade_no=0` 的初始资金点开始；成交存在 `entry_date / exit_date` 时会同步写入 `date`，自然周期收益和年化统计直接使用这条时间轴。即使没有成交也会保留初始资金行；`stats.json` 同时包含逐笔交易统计和净值曲线统计，例如 `annualized_return / annualized_volatility / equity_sharpe / calmar_ratio / ulcer_index / time_under_water_ratio`。
`trades.csv` 保留 `order_id / event_id / event_type / signal_date / signal_bar_index / side / planned_entry_price / stop_price / target_price / risk_per_share / r_multiple / mae_pct / mfe_pct / mae_r / mfe_r / metadata`，
可直接回查每笔成交来自哪根信号 K、哪个形态识别事件和哪类信号形态。
`order_decisions.csv` 记录单策略订单是否 `accepted`，以及 `invalid_order`、`duplicate_order_id`、`no_fill`、`no_liquidity`、`already_open`、`no_bars`、`actual_risk_too_high`、`chase_too_far`、`target_not_favorable` 等未成交原因；同时写入 `actual_entry_price / actual_risk_pct / actual_chase_pct / actual_reward_to_risk`，用于解释坏订单字段、重复订单身份、零流动性入场、跳空成交、过度追价和目标价失效。
`strategy_filter_decisions.csv` 记录策略层过滤结果，例如形态输出观察/中部不交易方向、信号 K 无流动性、大周期方向不一致、无可用大周期上下文或上下文过旧；基础策略过滤和大周期方向过滤会叠加保留，它早于撮合层，不和 `order_decisions.csv` 混用。
`limit_filter_audit.csv` 记录日 K 一字涨停过滤是否真实执行；严格模式下日线缺失或损坏会直接失败，只有显式关闭严格数据质量检查时才会继续输出 `daily_missing / daily_read_error / daily_missing_columns / daily_quality_error` 审计。
`order_decision_stats.csv` 和 `strategy_filter_stats.csv` 按策略、状态和原因聚合决策分布；`setup_order_decision_stats.csv` 和 `setup_strategy_filter_stats.csv` 按 `detector_name / event_type / side` 进一步拆解信号形态的撮合失败、风控拒绝和策略过滤拒绝。`decision_rate` 表示占全部决策的比例，`group_decision_rate` 表示在当前策略、信号形态或过滤器组内的比例。订单聚合表还会汇总实际风险、追价和实际盈亏比，用于定位哪类参数在撮合层失效。
`detector_stats.csv` 按 `detector_name` 独立汇总趋势、区间、通道、反转的成交绩效；`setup_stats.csv` 按 `detector_name / event_type / side` 汇总标志 K、失败突破、通道突破、H1/H2/L1/L2 等信号形态的多空表现。组合回测的 `strategy_stats.csv / detector_stats.csv / setup_stats.csv / symbol_stats.csv / side_stats.csv` 会额外给出 `return_contribution / capital_turnover / capital_weighted_raw_return`，用于拆解策略、识别模块、信号形态、标的和方向对组合净值的资金贡献；`capital_exposure_bars / margin_exposure_bars` 按仓位或保证金占用乘以持仓 K 数，衡量长期占资压力。组合 `stats.json` 还会从逐 K 净值曲线计算 `avg_cash_ratio / min_cash_ratio / max_cash_ratio / avg_net_exposure / min_net_exposure / max_net_exposure`，用于判断现金拖累、空头资金占用和多空偏向。
`data_inventory.csv` 保存本次实验涉及的日 K、主周期和高周期 parquet 缓存快照，包含是否存在、行数、起止时间、文件大小、修改时间和路径；`symbol_metadata.csv` 保存股票代码、股票名称和名称来源，优先读取行情目录 `symbols.csv / stock_names.csv`，其次读取 TDX `hq_cache` 的 `shm/szm/bjm.tnf`；`stats.json` 同步写入 `data_inventory_row_count / data_inventory_cached_count / data_inventory_missing_file_count / data_inventory_signature` 等摘要，其中 `data_inventory_signature` 是不含本机绝对路径的数据快照指纹，便于 Mac/Win 对照同一批缓存。
`stats.json` 同步写入 `order_count / accepted_order_count / rejected_order_count / acceptance_rate / rejected_no_fill_count / rejected_no_liquidity_count / rejected_no_bars_count / rejected_invalid_order_count / rejected_duplicate_order_id_count / rejected_already_open_count`
以及已接受订单的平均/最大 `capital_fraction / risk_fraction / margin_fraction`，若启用策略层过滤，还会写入 `strategy_signal_count / strategy_filter_acceptance_rate / strategy_rejected_signal_bar_no_liquidity_count / strategy_rejected_higher_timeframe_mismatch_count` 等摘要。数据审计摘要也会进入同一个文件，包括 `data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / data_audit_missing_file_count / data_audit_missing_columns_count / data_audit_no_window_data_count / data_audit_read_error_count / limit_filter_failed_count / limit_filter_daily_missing_count / limit_filter_daily_read_error_count / limit_filter_daily_missing_columns_count / limit_filter_daily_quality_error_count / limit_filter_filtered_days`。
多标的单策略回测会按实际入场时间排序成交，避免逐笔统计和净值曲线受股票代码顺序影响。

组合策略回测并保存产物：

```bash
python -m trending_winning.cli portfolio-backtest \
  --symbols 000001.SZ,600519.SH \
  --timeframe 30m \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --detectors trend,range,channel,reversal \
  --risk-reward 2.0 \
  --max-open-positions 5 \
  --max-actual-risk-pct 0.03 \
  --max-chase-pct 0.02 \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --trend-h2-min-pullback-legs 2 \
  --trend-pullback-lookback 5 \
  --channel-method regression \
  --channel-break-buffer 0 \
  --reversal-old-extreme-tolerance-pct 0.01 \
  --capital-per-trade 0.25 \
  --risk-per-trade 0.01 \
  --short-margin-rate 1.0 \
  --reserve-cash 0.1 \
  --allow-same-symbol-overlap \
  --strategy-priority trend_signal_bar=1,range_signal_bar=2 \
  --strategy-capital-limit trend_signal_bar=0.6,range_signal_bar=0.4 \
  --sector-capital-limit 银行=0.5,新能源=0.4 \
  --symbol-sector-map 000001.SZ=银行,300750.SZ=新能源 \
  --intrabar-exit-policy conservative \
  --output-dir runs/case-001 \
  --benchmark
```

保存目录会包含 `config.json`、`stats.json`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`setup_order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`setup_strategy_filter_stats.csv`、`equity_curve.csv`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv`、`limit_filter_audit.csv`、
`strategy_stats.csv`、`detector_stats.csv`、`setup_stats.csv`、`symbol_stats.csv`、`side_stats.csv`、`exit_reason_stats.csv`、`event_type_stats.csv`、`monthly_returns.csv` 和可选 `benchmark.json`。
`monthly_returns.csv` 的周期收益以上一条净值作为本期起点，避免漏掉月初第一笔净值变化；同时包含周期内最大回撤和净值样本数。
`stats.json` 会同步写入周期稳定性摘要，例如 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_return_std / monthly_worst_drawdown / monthly_max_consecutive_losses / monthly_max_recovery_periods`，用于比较不同形态、仓位参数和大周期方向过滤下的月度稳定性。逐笔交易统计还包含 `win_rate_ci_lower / win_rate_ci_upper / avg_return_standard_error / avg_return_ci_lower / avg_return_ci_upper / positive_expectancy_probability`，用于判断胜率和平均收益是否只是小样本波动。
组合仓位容量按实际 `entry_date` 分配，不按信号时间提前占用资金；风险预算按真实 `entry_price / risk_per_share` 计算，不用计划挂单价低估跳空成交风险。
`--capital-per-trade` 是固定单笔名义仓位，`--risk-per-trade` 是按真实入场风险反推仓位；两者都为空时按最大持仓数均分。
`--reserve-cash` 预留现金，`--allow-same-symbol-overlap` 允许同一股票多策略重叠持仓。
趋势、区间、通道、反转的识别参数各自传入 detector，组合层只负责排序、容量和资金分配。
`--strategy-priority`、`--strategy-capital-limit`、`--sector-capital-limit` 和 `--symbol-sector-map` 使用 `key=value,key=value` 格式，分别控制策略排序、策略资金上限、行业资金上限和股票所属行业。
`order_decisions.csv` 记录组合层接受或拒绝每个候选订单的原因，例如 `invalid_order`、`duplicate_order_id`、`no_fill`、`no_liquidity`、`no_bars`、`actual_risk_too_high`、`chase_too_far`、`target_not_favorable`、`max_open_positions`、`same_symbol_overlap`、`no_capital`；组合容量或资金拒绝也会保留候选成交的 `actual_entry_price / actual_risk_pct / actual_chase_pct / actual_reward_to_risk`。
`strategy_filter_decisions.csv` 记录订单进入组合撮合前被策略过滤的原因，包括观察/中部不交易方向、信号 K 无流动性和大周期方向不一致；包装策略会保留内层过滤日志，便于单策略回测和组合回测分别定位问题。
`limit_filter_audit.csv` 记录日线过滤状态；严格模式下 `daily_missing / daily_read_error / daily_missing_columns / daily_quality_error` 会中止回测，关闭严格数据质量检查排查时重点看这些状态和 `filtered_days`。
`order_decision_stats.csv` 和 `strategy_filter_stats.csv` 分别汇总撮合层与策略门控层的决策分布；`setup_order_decision_stats.csv` 和 `setup_strategy_filter_stats.csv` 用同一口径按 setup 继续拆分拒绝结构；`decision_rate` 是全局占比，`group_decision_rate` 是当前策略、setup 或过滤器组内占比，撮合层聚合表包含实际风险、追价和实际盈亏比摘要。
手续费率、滑点 bps 和初始资金会写入 `config.json`，并直接传给单策略、组合策略和参数遍历撮合层。
`--benchmark` 复用本次组合回测结果生成 `benchmark.json`，不再重复加载数据或重复撮合。
`stats.json` 同时保存逐笔交易指标和净值曲线指标：`annualized_return`、`annualized_volatility`、
`annualized_sharpe`、`annualized_sortino`、`calmar_ratio`、`avg_drawdown`、`ulcer_index`、`time_under_water_ratio`、`avg_gross_exposure`、`max_gross_exposure`、
`exposure_bar_ratio`、`avg_open_positions`、`max_open_positions`、`avg_cash_ratio`、`min_cash_ratio`、`max_cash_ratio`、`avg_net_exposure`、`min_net_exposure`、`max_net_exposure`、`avg_r_multiple`、`r_profit_factor`、`system_quality_number`、`avg_mae_pct`、`avg_mfe_pct`、
`return_p05`、`return_p25`、`return_p50`、`return_p75`、`return_p95`、`cvar_95`、`win_rate_ci_lower`、`win_rate_ci_upper`、`avg_return_standard_error`、`avg_return_ci_lower`、`avg_return_ci_upper`、`positive_expectancy_probability`、`capital_exposure_bars`、`margin_exposure_bars`、
以及 `order_count / accepted_order_count / rejected_order_count / acceptance_rate / rejection_rate /
rejected_invalid_order_count / rejected_duplicate_order_id_count / rejected_max_open_positions_count / rejected_no_capital_count / rejected_actual_risk_too_high_count / rejected_chase_too_far_count / rejected_target_not_favorable_count /
avg_executed_actual_risk_pct / max_executed_actual_risk_pct / avg_executed_actual_chase_pct / max_executed_actual_chase_pct / avg_executed_actual_reward_to_risk`
等订单摘要、策略过滤摘要和已接受订单的资金、风险、保证金占用统计。数据审计摘要也会进入同一个文件，包括 `data_inventory_signature / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / data_audit_missing_file_count / data_audit_missing_columns_count / data_audit_no_window_data_count / data_audit_read_error_count / limit_filter_failed_count / limit_filter_daily_missing_count / limit_filter_daily_read_error_count / limit_filter_daily_missing_columns_count / limit_filter_daily_quality_error_count / limit_filter_filtered_days`。

单策略独立参数遍历：

```bash
python -m trending_winning.cli single-sweep \
  --symbols 000001.SZ,600519.SH \
  --timeframe 30m \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --detector trend \
  --risk-rewards 1.5,2.0,2.5 \
  --max-holding-bars-list 6,12,18 \
  --trend-min-scores 0.8,1.0,1.2 \
  --max-actual-risk-pct 0.03 \
  --max-chase-pct 0.02 \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --output-dir runs/single-sweep-001
```

`single-sweep` 只绑定一个 detector，不进入组合仓位分配层；一次加载数据，多组参数复用同一批 K 线，订单参数不变时复用已生成订单。
除固定的 `--risk-rewards / --max-holding-bars-list` 外，也可以重复传 `--grid 字段=值1,值2` 遍历任意实验配置字段，例如
`--grid range_min_score=0.7,0.9 --grid fee_rate=0,0.0003`；布尔字段用 `true/false`。
重复的 grid 值会在笛卡尔积展开前按稳定指纹去重，避免重复配置进入订单生成和回测热路径；`summary.json` 仍保留原始 `grid_case_count` 和去重后的 `case_count`。
固定启用 detector 的 sweep 只保留已启用 detector 相关参数、通用撮合参数和实际启用的高周期 trend 上下文字段；只改未启用模块参数时，不会生成额外 case，也不会改变 `case_config_hash`。
结果按收益、回撤、月度稳定性、交易数和 case 名稳定排序保存 `sweep.csv`，首列 `sweep_rank` 是可直接筛选的参数排名；`pareto_rank=1` 表示按收益、回撤、Ulcer、月度最差收益、月度收益波动和交易样本数得到的第一层非支配候选集，分层时使用 NumPy 批量支配矩阵，避免逐候选嵌套比较拖慢大网格。`case_config_hash` 是完整实验配置的 SHA-256 指纹，`data_inventory_signature` 是本次 K 线缓存快照指纹，方便跨机器复现和对照。同时保存 `config.json`、`summary.json`、`pareto.csv`、`parameter_summary.csv`、`case_setup_stats.csv`、`case_setup_order_decision_stats.csv`、`case_setup_strategy_filter_stats.csv`、`case_configs.jsonl`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv` 和 `limit_filter_audit.csv`；`summary.json` 汇总 `grid_case_count / case_count / pareto_case_count / best_case_name / best_case_config_hash / data_inventory_signature / order_cache_hit_rate / data_coverage_below_min_count / data_weighted_coverage_ratio / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / limit_filter_filtered_days / case_setup_order_decision_count / case_setup_order_rejected_count / case_setup_strategy_filter_decision_count / case_setup_strategy_filter_rejected_count`，其中 `grid_case_count` 是原始笛卡尔积组合数，`case_count` 是去重后实际运行数，`pareto.csv` 只保留第一层 Pareto 候选，`parameter_summary.csv` 按 grid 字段和值聚合收益、回撤、月度稳定性、撮合/策略过滤接受率、Pareto 命中率、正收益率和收益离散度，避免只看平均收益选参数，`case_setup_stats.csv` 按 `case_config_hash / detector_name / event_type / side` 下钻每个参数组的 setup 绩效，`case_setup_order_decision_stats.csv` 和 `case_setup_strategy_filter_stats.csv` 则按同一 setup 维度拆分撮合拒绝和策略过滤拒绝。`case_configs.jsonl` 按 `sweep.csv` 排名顺序逐行保存完整 case 配置。每行会带 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_return_std / monthly_max_consecutive_losses / monthly_max_recovery_periods`、`data_inventory_signature / data_inventory_cached_count / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / limit_filter_filtered_days`，参数结果不脱离周期稳定性和数据质量语境。

组合层复用同一批 K 线做参数遍历：

```bash
python -m trending_winning.cli portfolio-sweep \
  --symbols 000001.SZ,600519.SH \
  --timeframe 30m \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --detectors trend,range,channel \
  --risk-rewards 1.5,2.0,2.5 \
  --max-holding-bars-list 6,12,18 \
  --max-actual-risk-pct 0.03 \
  --max-chase-pct 0.02 \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --max-open-positions-list 3,5 \
  --trend-lookback 20 \
  --trend-min-score 1.0 \
  --trend-h2-min-pullback-legs 2 \
  --range-lookback 20 \
  --channel-method regression \
  --channel-lookback 40 \
  --channel-sigma-multiple 2.0 \
  --reversal-lookback 20 \
  --reversal-old-extreme-tolerance-pct 0.01 \
  --output-dir runs/sweep-001
```

`portfolio-sweep` 会先做一次数据加载和质量检查；当遍历项不改变形态、信号订单参数和大周期方向过滤参数时，会复用同一批形态订单。
当遍历项只改变仓位、容量、行业/策略上限或初始资金时，还会复用已撮合的候选成交，避免重复扫描 K 线和重复模拟成交路径。
参数遍历热路径直接使用已加载的标准化 K 线，不在每个参数组重复做行情 schema 标准化。
同一次 `portfolio-sweep` 固定复用同一批 K 线，grid 不能改变 `data_root / symbols / timeframe / start / end / adjust / strict_data_quality / min_coverage_ratio`。
组合遍历同样支持重复 `--grid 字段=值1,值2`，可用于 `reserve_cash / allow_same_symbol_overlap / trend_min_score / range_min_score` 等非数据范围字段。
mapping 参数用分号分隔多个方案，用 `+` 分隔同一方案内多个键值，例如
`--grid strategy_capital_limit=trend_signal_bar=0.4+range_signal_bar=0.3;trend_signal_bar=0.7+range_signal_bar=0.2`，
也可用于 `symbol_sector_map` 这类映射参数。
重复的 grid 值会在展开前去重，未启用 detector 的参数也会被过滤，避免同一组合重复生成订单、候选成交或组合分配结果。
参数笛卡尔积结果会按收益、回撤、月度稳定性、交易数和 case 名稳定排序保存 `sweep.csv`，首列 `sweep_rank` 是参数排名；`pareto_rank` 用收益、回撤、Ulcer、月度最差收益、月度收益波动和交易样本数标记非支配层级；`case_config_hash` 记录完整 case 配置指纹，`data_inventory_signature` 记录本次 K 线缓存快照指纹。
保存目录同时包含 `config.json`、`summary.json`、`pareto.csv`、`parameter_summary.csv`、`case_setup_stats.csv`、`case_setup_order_decision_stats.csv`、`case_setup_strategy_filter_stats.csv`、`case_configs.jsonl`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv` 和 `limit_filter_audit.csv`；`config.json` 写入基础配置和 `sweep_grid`，`summary.json` 写入原始组合数、去重后实际 case 数、最佳 case、数据快照指纹、Pareto 候选数量、订单/候选缓存命中率、数据覆盖率、最长连续数据缺口、日 K 涨停开盘过滤摘要和信号形态决策摘要，`pareto.csv` 写入 `pareto_rank=1` 的候选集合，`parameter_summary.csv` 按每个 grid 参数值聚合 `case_count / pareto_case_count / pareto_hit_rate / positive_return_case_count / positive_return_rate / std_total_return / best_total_return / worst_total_return / best_sweep_rank / avg_total_return / avg_max_drawdown / avg_monthly_worst_return / avg_monthly_return_std / avg_acceptance_rate / avg_rejection_rate / avg_rejected_no_fill_count / avg_strategy_filter_acceptance_rate / avg_strategy_filter_rejection_rate`，`case_setup_stats.csv` 按 case 排名保留信号形态级绩效，两张 `case_setup_*_decision_stats.csv` 按 case 排名保留信号形态级撮合拒绝和策略过滤拒绝。`case_configs.jsonl` 写入每个参数组的完整配置。`sweep.csv` 会附带 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_return_std / monthly_max_consecutive_losses / monthly_max_recovery_periods`、`order_cache_status / candidate_cache_status / generated_order_count / candidate_count / candidate_rejection_count`、`order_count / acceptance_rate / rejected_no_fill_count`、`data_inventory_signature / data_inventory_cached_count / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / limit_filter_filtered_days` 等周期稳定性、性能、订单决策和数据质量摘要，方便解释参数组表现；如果只改变 `higher_timeframe_max_age_minutes`，会重新生成大周期方向过滤后的订单，不复用旧订单。

从遍历结果回放某个 case：

```bash
python -m trending_winning.cli replay-case \
  --case-configs runs/sweep-001/case_configs.jsonl \
  --case-config-hash <sweep.csv里的case_config_hash> \
  --output-dir runs/replay-001
```

`replay-case` 会读取完整 case 配置并运行对应的单策略或组合回测；回放前会重新计算配置指纹，发现 `case_config_hash` 与配置内容不一致时拒绝回放；传入 `--output-dir` 时会保存新的 `config.json / stats.json / trades.csv / equity_curve.csv` 等回测产物。

## Docker

```bash
docker build -t trending-winning .
docker run --rm -p 8520:8501 \
  -v /Users/a1234/Desktop/trend-backtest/data:/data \
  trending-winning
```

容器内默认可把页面数据目录改成 `/data/market/daily`。TDX 真取数仍建议在 Windows/Parallels 侧运行；Docker 更适合读取已落地 parquet 后做扫描和回测。
