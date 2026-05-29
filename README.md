# TrendingWinning

TDX-only A 股 K 线趋势策略工作台。

精简版 HTML 使用指南见 `docs/usage_guide.html`，可直接用浏览器打开。

首版能力：

- TDX `tqcenter` K 线：日 K `1d` 和分钟 K `5m / 15m / 30m / 60m`
- 本地 parquet 落地：兼容 `trend-backtest/data/market/<timeframe>/<adjust>/<symbol>.parquet`，其中 `1d` 写入既有 `market/daily/<adjust>` 目录
- 标志K识别：振幅、量能、实体比例参数化
- 趋势通道识别：支持批量滚动 log 回归通道和摆动点趋势通道，上下轨、斜率、R²、方向和锚点可追溯
- 突破 trigger：基于上一根已完成通道上轨，避免把当前突破 K 纳入边界
- 多周期扫描：一次聚合 `5m / 15m / 30m / 60m` 的最新通道和突破状态
- 多周期门控策略：高周期方向只过滤低周期订单，不改趋势/区间/通道/反转 detector 输出
- 数据审计：本地 parquet 覆盖、日期/标的代码、字段质量、OHLC 合法性、重复 K、交易时段内缺失 K、零流动性 K 和覆盖率显式输出
- 独立 detector 策略：趋势、区间、通道、反转模块解耦，单策略回测只消费本策略事件
- Detector 事件契约：非空事件表必须包含 `event_id / detector_name / stock_code / timeframe / date / bar_index / event_type / direction / signal_price / entry_price / stop_price / confidence / metadata`，策略层统一校验，缺字段会直接报错
- 订单契约：撮合层要求 `order_id / event_id / stock_code / signal_date / signal_bar_index / side / entry_price / stop_price / target_price` 必备，组合回测额外要求 `strategy_name`；空 `order_id` 或 `event_id` 记为 `invalid_order`，重复 `order_id` 记为 `duplicate_order_id`
- 高性能订单链路：信号 K 策略用列运算把 detector 事件转换为挂单，撮合层用排序后的记录流处理订单和候选成交，避免在参数遍历热路径逐行创建 pandas 行对象
- 趋势回撤事件：`TrendDetector` 输出 `trend_state / pullback_legs`，并把顺势回撤信号区分为 `bull_h1_setup / bull_h2_setup / bear_l1_setup / bear_l2_setup`
- 区间识别评分：`RangeDetector` 输出 `range_score / overlap_mean / ema_flatness / directional_efficiency`，过滤强趋势里的中部噪声
- 反转确认：`ReversalDetector` 默认第一次反转只观察，第二次反转必须满足旧极端失败测试和结构确认后才输出交易事件
- 组合回测：按真实入场时间和真实成交风险做策略优先级、持仓互斥、风险预算、行业/策略资金上限、空头保证金和逐 K 净值重估
- 实际风险过滤：信号 K 挂单策略透传 `max_actual_risk_pct` 和 `max_chase_pct`，由撮合层按真实成交价统一判定并记录拒绝原因
- Detector 参数透传：趋势强收盘/实体/回撤窗口、区间中部/失败突破/区间评分、通道突破缓冲/摆动锚点、反转强收盘/实体阈值都可配置
- 回测统计：逐笔统计与净值曲线统计分离，输出总收益、最大回撤、年化收益、年化波动、Calmar、暴露度、持仓数和策略/标的/方向/退出原因拆分
- 事件类型拆分：订单和成交透传 `event_type`，可单独评估 H1/H2、失败突破、通道突破、二次反转等 setup 表现
- 真实撮合边界：跳空穿越按开盘成交；同 K 同时触发止盈止损时默认保守止损优先，可显式改为乐观止盈优先
- 流动性门禁：分钟 K 的 `volume` 或 `amount` 为 0 时会从回测数据包剔除；裸策略/撮合入口仍不允许用这类 K 入场、止盈止损或统计路径波动
- 回测数据过滤：按主板/科创/创业/BJ 的日 K 涨停开盘规则剔除污染交易日
- 多周期数据包：`5m / 15m / 30m / 60m` 可统一审计、统一日线过滤、按周期拆分给扫描和回测
- Streamlit Web 应用，后续可挂 Docker 数据卷

## 本地运行

```bash
cd /Users/a1234/Documents/TrendingWinning
python -m pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8520
```

TDX 真取数只支持 Windows/Parallels 内的通达信。Mac 本机通达信不支持 `tqcenter` 取数，Mac 端 CLI 默认用 `--runtime auto` 调度到 Parallels；Windows 侧运行时默认本地执行。`60m` 会按 TDX 接口要求映射为 `1h` 请求；分钟线是否能返回数据取决于 Windows 通达信本地是否已下载对应周期数据。

Parallels 默认配置：

- VM：`Windows 11`，可用 `--parallels-vm` 或 `TDX_PARALLELS_VM` 覆盖
- Windows Python：`C:\Users\Public\venvs\trending-winning\Scripts\python.exe`，可用 `--windows-python` 或 `TDX_PARALLELS_PYTHON` 覆盖
- Windows 仓库路径：默认把 Mac 当前仓库映射为 `C:\Mac\Home\...`，可用 `--windows-repo` 或 `TDX_PARALLELS_REPO` 覆盖
- TDX 插件目录：用 Windows 路径传给 `--tdx-path`，例如 `C:\new_tdx\T0002\PYPlugins\user`

Web 回测页里的“单策略回测”和“组合策略回测”与 CLI 复用同一套实验运行器。
单策略只绑定一个 detector，不进入组合仓位分配层；组合策略才启用策略优先级、资金上限、行业上限和持仓互斥。
需要 60m 判主方向、15m/5m 触发时，可用 `HigherTimeframeAlignmentStrategy` 包装任一基础策略；它按订单 `signal_date` 向前匹配高周期上下文，拒绝方向不一致或上下文过期的订单，基础 detector 仍保持独立，拒绝原因会单独写入策略层过滤日志。
“高级 detector 参数”在 Web 里按单策略和组合策略分开配置，可单独调整趋势、区间、通道、反转识别阈值。
勾选“保存实验产物”后会把 `config.json`、`stats.json`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`equity_curve.csv` 和数据审计文件写入页面指定的输出目录。
“TDX K线”页支持“审计并补齐TDX数据”：先审计本地 parquet，只对缺失、质量错误或覆盖率低于门槛的标的周期请求 TDX。

## CLI

抓取并写入本地 parquet：

```bash
python -m trending_winning.cli tdx-doctor \
  --symbols 000001.SZ,600519.SH \
  --timeframes 1d,5m,15m,30m,60m \
  --start "2026-05-25 09:30:00" \
  --end "2026-05-25 15:00:00" \
  --runtime parallels \
  --tdx-path "C:\\new_tdx\\T0002\\PYPlugins\\user"

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

`plan-data` 只读取本地 parquet 和日 K 交易日锚点，不请求 TDX；它会把每个 `stock_code/timeframe` 标成 `cached` 或 `fetch`，并写明 `missing_file / quality_error / coverage_below_min / local_ok`。
`prepare-data` 会输出每个 `stock_code/timeframe` 的 `cached/fetched` 动作、补数前后状态、写入行数、补前/补后覆盖率、补前/补后缺失 K 数、补前/补后最长连续缺口和本地 parquet 路径；默认补完后仍不达标会直接失败。
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

组合回测默认启用严格数据质量门禁：本地 parquet 缺失、请求窗口无可交易数据、缺字段、日期无法解析、标的代码为空或无法规范化、重复 K、OHLC 为空、价格非正、high/low 与 open/close 不一致、volume/amount 为空或负数，以及日 K 缺失导致一字涨停过滤无法执行，都会直接失败。`volume` 或 `amount` 为 0 会记录为 `zero_volume_amount_rows`，不直接记为字段质量错误，但覆盖率按剔除这些 K 后的可交易 K 计算，回测数据包也会先剔除这些 K；裸策略/撮合入口仍不会用这类 K 入场、止盈止损或统计路径波动。
临时排查时可以在 CLI 里加 `--allow-bad-data`，页面里取消“严格数据质量门禁”。
多周期扫描会复用同一套数据门禁和涨停开盘过滤，不再直接绕过本地行情审计。
`audit-data` 会同时输出 `invalid_date_rows / invalid_symbol_rows / zero_volume_amount_rows / expected_rows / missing_rows / coverage_ratio / max_missing_gap_minutes`；其中 `rows_in_window / missing_rows / coverage_ratio` 按剔除零流动性后的可交易 K 计算，`zero_volume_amount_rows` 单独保留原始零流动性数量。
直接审计和回测加载都会优先用本地日 K 作为交易日锚点计算分钟线覆盖度，可暴露整天分钟 K 缺失；日 K 不存在时才退回按已观测分钟交易日计算。
需要把覆盖度也变成硬门禁时，在单策略、组合或参数遍历命令里加 `--min-coverage-ratio 0.95`。

扫描并回测：

```bash
python -m trending_winning.cli backtest \
  --symbols 000001.SZ,600519.SH \
  --timeframe 60m \
  --start 2026-05-01 \
  --end 2026-05-25 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily
```

单策略回测只绑定一个 detector，不进入组合仓位分配层：

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

保存目录只包含单策略产物：`config.json`、`stats.json`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`equity_curve.csv`、`data_coverage.csv`、`limit_filter_audit.csv`、
`strategy_stats.csv`、`symbol_stats.csv`、`side_stats.csv`、`exit_reason_stats.csv`、`event_type_stats.csv`、`monthly_returns.csv`。
`monthly_returns.csv` 的周期收益以上一条净值作为本期起点，避免漏掉月初第一笔净值变化；同时包含周期内最大回撤和净值样本数。
单策略 `equity_curve.csv` 从 `trade_no=0` 的初始资金点开始，即使没有成交也会保留这一行；`stats.json` 同时包含逐笔交易统计和净值曲线统计，例如 `annualized_return / annualized_volatility / equity_sharpe / calmar_ratio`。
`trades.csv` 保留 `order_id / event_id / event_type / signal_date / signal_bar_index / side / planned_entry_price / stop_price / target_price / risk_per_share / r_multiple / mae_pct / mfe_pct / mae_r / mfe_r / metadata`，
可直接回查每笔成交来自哪根信号 K、哪个 detector 事件和哪类 setup。
`order_decisions.csv` 记录单策略订单是否 `accepted`，以及 `invalid_order`、`duplicate_order_id`、`no_fill`、`no_liquidity`、`already_open`、`no_bars`、`actual_risk_too_high`、`chase_too_far`、`target_not_favorable` 等未成交原因；同时写入 `actual_entry_price / actual_risk_pct / actual_chase_pct / actual_reward_to_risk`，用于解释坏订单字段、重复订单身份、零流动性入场、跳空成交、过度追价和目标价失效。
`strategy_filter_decisions.csv` 记录策略层过滤结果，例如 detector 输出观察/中部不交易方向、信号 K 无流动性、高周期方向不一致、无可用高周期上下文或上下文过期；基础策略过滤和高周期门控过滤会叠加保留，它早于撮合层，不和 `order_decisions.csv` 混用。
`limit_filter_audit.csv` 记录日 K 一字涨停过滤是否真实执行；严格模式下日线缺失会直接失败，只有显式关闭严格数据门禁时才会继续输出 `daily_missing` 审计。
`order_decision_stats.csv` 和 `strategy_filter_stats.csv` 按策略、状态和原因聚合决策分布；`decision_rate` 表示占全部决策的比例，`group_decision_rate` 表示在当前策略或过滤器组内的比例。订单聚合表还会汇总实际风险、追价和实际盈亏比，用于定位哪类参数在撮合层失效。
组合回测的 `strategy_stats.csv / symbol_stats.csv / side_stats.csv` 会额外给出 `return_contribution / capital_turnover / capital_weighted_raw_return`，用于拆解策略、标的和方向对组合净值的资金贡献；`capital_exposure_bars / margin_exposure_bars` 按仓位或保证金占用乘以持仓 K 数，衡量长期占资压力。
`stats.json` 同步写入 `order_count / accepted_order_count / rejected_order_count / acceptance_rate / rejected_no_fill_count / rejected_no_liquidity_count / rejected_no_bars_count / rejected_invalid_order_count / rejected_duplicate_order_id_count / rejected_already_open_count`
以及已接受订单的平均/最大 `capital_fraction / risk_fraction / margin_fraction`，若启用策略层过滤，还会写入 `strategy_signal_count / strategy_filter_acceptance_rate / strategy_rejected_signal_bar_no_liquidity_count / strategy_rejected_higher_timeframe_mismatch_count` 等摘要。
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

保存目录会包含 `config.json`、`stats.json`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`equity_curve.csv`、`data_coverage.csv`、`limit_filter_audit.csv`、
`strategy_stats.csv`、`symbol_stats.csv`、`side_stats.csv`、`exit_reason_stats.csv`、`event_type_stats.csv`、`monthly_returns.csv` 和可选 `benchmark.json`。
`monthly_returns.csv` 的周期收益以上一条净值作为本期起点，避免漏掉月初第一笔净值变化；同时包含周期内最大回撤和净值样本数。
组合仓位容量按实际 `entry_date` 分配，不按信号时间提前占用资金；风险预算按真实 `entry_price / risk_per_share` 计算，不用计划挂单价低估跳空成交风险。
`--capital-per-trade` 是固定单笔名义仓位，`--risk-per-trade` 是按真实入场风险反推仓位；两者都为空时按最大持仓数均分。
`--reserve-cash` 预留现金，`--allow-same-symbol-overlap` 允许同一股票多策略重叠持仓。
趋势、区间、通道、反转的识别参数各自传入 detector，组合层只负责排序、容量和资金分配。
`--strategy-priority`、`--strategy-capital-limit`、`--sector-capital-limit` 和 `--symbol-sector-map` 使用 `key=value,key=value` 格式，分别控制策略排序、策略资金上限、行业资金上限和股票所属行业。
`order_decisions.csv` 记录组合层接受或拒绝每个候选订单的原因，例如 `invalid_order`、`duplicate_order_id`、`no_fill`、`no_liquidity`、`no_bars`、`actual_risk_too_high`、`chase_too_far`、`target_not_favorable`、`max_open_positions`、`same_symbol_overlap`、`no_capital`；组合容量或资金拒绝也会保留候选成交的 `actual_entry_price / actual_risk_pct / actual_chase_pct / actual_reward_to_risk`。
`strategy_filter_decisions.csv` 记录订单进入组合撮合前被策略门控过滤的原因，包括观察/中部不交易方向、信号 K 无流动性和高周期方向门控；包装策略会保留内层过滤日志，便于单策略回测和组合回测分别定位问题。
`limit_filter_audit.csv` 记录日线过滤状态；严格模式下 `daily_missing` 会中止回测，关闭严格门禁排查时重点看 `daily_missing` 和 `filtered_days`。
`order_decision_stats.csv` 和 `strategy_filter_stats.csv` 分别汇总撮合层与策略门控层的决策分布；`decision_rate` 是全局占比，`group_decision_rate` 是当前策略或过滤器组内占比，撮合层聚合表包含实际风险、追价和实际盈亏比摘要。
手续费率、滑点 bps 和初始资金会写入 `config.json`，并直接传给单策略、组合策略和参数遍历撮合层。
`--benchmark` 复用本次组合回测结果生成 `benchmark.json`，不再重复加载数据或重复撮合。
`stats.json` 同时保存逐笔交易指标和净值曲线指标：`annualized_return`、`annualized_volatility`、
`annualized_sharpe`、`annualized_sortino`、`calmar_ratio`、`avg_gross_exposure`、`max_gross_exposure`、
`exposure_bar_ratio`、`avg_open_positions`、`max_open_positions`、`avg_r_multiple`、`r_profit_factor`、`system_quality_number`、`avg_mae_pct`、`avg_mfe_pct`、
`return_p05`、`return_p25`、`return_p50`、`return_p75`、`return_p95`、`cvar_95`、`capital_exposure_bars`、`margin_exposure_bars`、
以及 `order_count / accepted_order_count / rejected_order_count / acceptance_rate / rejection_rate /
rejected_invalid_order_count / rejected_duplicate_order_id_count / rejected_max_open_positions_count / rejected_no_capital_count / rejected_actual_risk_too_high_count / rejected_chase_too_far_count / rejected_target_not_favorable_count /
avg_executed_actual_risk_pct / max_executed_actual_risk_pct / avg_executed_actual_chase_pct / max_executed_actual_chase_pct / avg_executed_actual_reward_to_risk`
等订单摘要、策略过滤摘要和已接受订单的资金、风险、保证金占用统计。

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
结果按收益排序保存 `sweep.csv`，并同时保存 `config.json`、`data_coverage.csv` 和 `limit_filter_audit.csv`。

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

`portfolio-sweep` 会先做一次数据加载和质量门禁；当遍历项不改变 detector、信号订单参数和高周期门控参数时，会复用同一批 detector 订单。
当遍历项只改变仓位、容量、行业/策略上限或初始资金时，还会复用已撮合的候选成交，避免重复扫描 K 线和重复模拟成交路径。
参数遍历热路径直接使用已加载的标准化 K 线，不在每个参数组重复做行情 schema 标准化。
同一次 `portfolio-sweep` 固定复用同一批 K 线，grid 不能改变 `data_root / symbols / timeframe / start / end / adjust / strict_data_quality / min_coverage_ratio`。
组合遍历同样支持重复 `--grid 字段=值1,值2`，可用于 `reserve_cash / allow_same_symbol_overlap / trend_min_score / range_min_score` 等非数据范围字段。
mapping 参数用分号分隔多个方案，用 `+` 分隔同一方案内多个键值，例如
`--grid strategy_capital_limit=trend_signal_bar=0.4+range_signal_bar=0.3;trend_signal_bar=0.7+range_signal_bar=0.2`，
也可用于 `symbol_sector_map` 这类映射参数。
参数笛卡尔积结果会按收益排序保存 `sweep.csv`。
保存目录同时包含 `config.json`、`data_coverage.csv` 和 `limit_filter_audit.csv`；`config.json` 写入基础配置和 `sweep_grid`，`sweep.csv` 会附带 `order_cache_status / candidate_cache_status / generated_order_count / candidate_count / candidate_rejection_count`、`order_count / acceptance_rate / rejected_no_fill_count` 等性能和订单决策摘要，方便解释参数组表现；如果只改变 `higher_timeframe_max_age_minutes`，会重新生成高周期门控后的订单，不复用旧订单。

## Docker

```bash
docker build -t trending-winning .
docker run --rm -p 8520:8501 \
  -v /Users/a1234/Desktop/trend-backtest/data:/data \
  trending-winning
```

容器内默认可把页面数据目录改成 `/data/market/daily`。TDX 真取数仍建议在 Windows/Parallels 侧运行；Docker 更适合读取已落地 parquet 后做扫描和回测。
