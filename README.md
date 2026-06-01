# TrendingWinning

TDX-only A 股 K 线趋势策略工作台。

精简版 HTML 使用指南见 `docs/usage_guide.html`；回测界面和 K 线范例见 `docs/backtest_kline_guide.html`；趋势策略实战讲解见 `docs/trend_strategy_000852_guide.html`，通道策略实战讲解见 `docs/channel_strategy_000852_guide.html`，均可直接用浏览器打开。

首版能力：

- TDX `tqcenter` K 线：日 K `1d` 和分钟 K `5m / 15m / 30m / 60m`
- TDX 分钟兜底：`15m / 30m / 60m` 原生周期无数据时，自动用 TDX `5m` 聚合生成目标周期 K 线
- 本地 parquet 落地：兼容 `trend-backtest/data/market/<timeframe>/<adjust>/<symbol>.parquet`，其中 `1d` 写入既有 `market/daily/<adjust>` 目录
- 本地缓存库存：按 `1d / 5m / 15m / 30m / 60m` 列出 parquet 是否存在、行数、起止时间、文件大小和状态
- 标志K识别：振幅、量能、实体比例参数化
- 趋势通道识别：支持批量滚动 log 回归通道和摆动点趋势通道；上升通道优先用确认低点画上升支撑线，下降通道优先用确认高点画下降压力线，上下轨、斜率、R²、方向和锚点可追溯
- 市场结构识别：pivot 标在实际摆动 K 上，但 `last_swing_high / last_swing_low / structure_score / BOS / CHoCH` 通过向量化延迟确认，只在右侧确认 K 线完成后才更新，避免结构字段提前暴露未来信息
- 突破 trigger：基于上一根已完成通道上轨，避免把当前突破 K 纳入边界
- 多周期扫描：一次聚合 `5m / 15m / 30m / 60m` 的最新通道和突破状态
- 大周期方向过滤：用 60m/30m 判断主方向，只过滤低周期逆势订单，不改趋势/区间/通道/反转识别结果
- 末端假突破过滤：可选策略层开仓过滤，趋势或通道持续较久、价格远离中轴、反复贴近边缘、突破推进不足且影线失败时拒绝本次开仓；默认关闭，不改 detector、撮合和仓位分配
- 数据审计：本地 parquet 覆盖、日期/标的代码、字段质量、OHLC 合法性、重复 K、交易时段内缺失 K、缺口首尾时间、最长连续缺口边界、零流动性 K 和覆盖率显式输出
- 独立形态识别策略：趋势、区间、通道、反转模块解耦，单策略回测只消费本策略事件
- Detector 事件契约：非空事件表必须包含 `event_id / detector_name / stock_code / timeframe / date / bar_index / event_type / direction / signal_price / entry_price / stop_price / confidence / metadata`，策略层统一校验，缺字段会直接报错
- 订单契约：撮合层要求 `order_id / event_id / stock_code / signal_date / signal_bar_index / side / entry_price / stop_price / target_price` 必备，组合回测额外要求 `strategy_name`；空 `order_id` 或 `event_id` 记为 `invalid_order`，重复 `order_id` 记为 `duplicate_order_id`
- 高性能订单链路：信号 K 策略用列运算把 detector 事件转换为挂单，单策略、组合实验和外部订单回测都可复用已标准化 K 线，sweep 热路径复用订单和候选成交，避免重复标准化或逐行创建 pandas 行对象
- 趋势回撤事件：`TrendDetector` 输出 `trend_state / pullback_legs`，并把顺势回撤信号区分为 `bull_h1_setup / bull_h2_setup / bear_l1_setup / bear_l2_setup`
- 区间识别评分：`RangeDetector` 输出 `range_score / overlap_mean / ema_flatness / directional_efficiency`，过滤强趋势里的中部噪声
- 反转确认：`ReversalDetector` 默认第一次反转只观察，第二次反转必须满足旧极端失败测试和结构确认后才输出交易事件
- 组合回测：按真实入场时间和真实成交风险做策略优先级、持仓互斥、风险预算、行业/策略资金上限、空头保证金和逐 K 净值重估；回撤口径使用持仓方向的不利 K 线价格，多头看 low，空头看 high
- 止损风险过滤：信号 K 挂单策略透传 `max_actual_risk_pct` 和 `max_chase_pct`，由撮合层按真实成交价统一判定并记录拒绝原因
- Detector 参数透传：趋势强收盘/实体/回撤窗口、区间中部/失败突破/区间评分、通道突破缓冲/摆动锚点、反转强收盘/实体阈值都可配置
- 回测统计：逐笔统计与净值曲线统计分离，输出总收益、基于价格路径的最大回撤、最大回撤开始/触底/修复时间、当前回撤、当前水下 K 数、胜率置信区间、盈亏平衡胜率、胜率边际、平均收益标准误、正期望概率、平均回撤、Ulcer Index、水下时间比例、年化收益、年化波动、Calmar、`market_bar_count`、`exposure_bar_ratio`、现金占比、净暴露、总暴露、持仓数和策略/标的/方向/退出原因拆分
- 事件类型拆分：订单和成交透传 `event_type`，可单独评估 H1/H2、失败突破、通道突破、二次反转等 setup 表现
- 真实撮合边界：跳空穿越按开盘成交；同 K 同时触发止盈止损时默认保守止损优先，可显式改为乐观止盈优先；盈利通道回撤止盈按实际成交价和上一根已完成 K 的最大盈利价位或当前周期均线确认，避免同一根 K 同时启动和退出
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
单策略和组合策略参数区会在运行前展示“策略执行空间”，把当前样本、识别形态、适用空间、信号条件、触发成交、开仓过滤、退出条件、仓位规则、或然分支和复盘输出翻译成一张表，方便先确认哪些模块会生效、哪些参数只属于组合层、哪些订单会进入过滤或撮合。策略空间不是一个单项参数，而是样本、形态、过滤、执行、退出、统计六类边界；更细看则是可交易空间、参数空间、过滤空间、执行空间、退出空间、统计空间，完整清单还包括样本空间、标的空间、周期空间、订单空间、风险空间、持仓空间和失效空间。
信号条件会明确写出多头/空头信号 K、入场触发价、结构止损和成交确认方式；触发条件拆成背景条件、信号K条件、订单条件、风控条件：先确认 detector 背景状态，再确认信号K质量，再用信号K极值外侧挂单，最后按真实成交价检查结构止损风险、追价和目标价。触发成交条件写明为多头 high >= 入场触发价、空头 low <= 入场触发价；或然分支会按识别阶段、触发阶段、过滤阶段、撮合阶段、持仓阶段、退出阶段拆开，完整生命周期包括无信号、观察信号、有效信号、有效未触发、触发成交、触发后拒单、持仓冲突、容量/资金拒单和退出完成，并继续细分正常触发、跳空触发和触发后拒单。最常用的阅读顺序是：背景不满足 -> 无信号；背景满足但信号K质量不足 -> 观察信号；信号成立但未穿越挂单价 -> 有效但未触发；穿越后风险不合格 -> 触发后拒单；成交后再进入止损、固定目标、最大盈利回撤或到期退出。
单策略只绑定一种形态识别模块，不进入组合仓位分配层；组合策略才启用策略优先级、资金上限、行业上限和持仓互斥。
单策略和组合策略都支持交易方向：`both` 表示多/空都做，`long_only` 表示仅多，`short_only` 表示仅空；被方向模式过滤的信号会写入 `strategy_filter_decisions.csv`。
回测结果页会先展示“策略K线运行区间”，按股票切换完整样本 K 线，横轴按连续 K 序号压缩，横轴和价格轴都支持缩放；长样本不会受 5000 行默认渲染限制，并使用 SVG 矢量渲染保持放大后的清晰度。图上标注开多、开空、平仓、止损和盈利通道回撤止盈，悬停只显示价格、具体开仓/平仓时间和开仓/平仓原因，方便先检查信号和风控是否落在正确 K 线上；随后展示“核心绩效概览”、净值曲线、回撤曲线、回撤曲线明细、回撤区间明细、实验诊断摘要、交易路径分布、数据覆盖率检查、订单决策概览、拒绝原因分布、逐笔交易、净值明细、策略绩效、识别模块绩效、信号形态绩效、订单决策统计和策略过滤统计。
需要 60m 判主方向、15m/5m 触发时，可用 `HigherTimeframeAlignmentStrategy` 包装任一基础策略；它按订单 `signal_date` 向前匹配大周期上下文，拒绝方向不一致或上下文过旧的订单，基础形态识别仍保持独立，拒绝原因会单独写入策略层过滤日志。
“末端假突破过滤（可选）”只出现在单策略和组合策略参数区，不出现在旧突破回测。它只过滤开仓订单，交易方向沿用现有 `both / long_only / short_only`，拒绝原因写入 `strategy_filter_decisions.csv` 的 `terminal_false_breakout_risk`。
“高级形态识别参数”在 Web 里按单策略和组合策略分开配置，可单独调整趋势、区间、通道、反转识别阈值。
勾选“保存实验产物”后会把 `artifact_manifest.csv`、`config.json`、`strategy_space.csv`、`stats.json`、`experiment_diagnostics.csv`、`trades.csv`、`signal_lifecycle_stats.csv`、`trade_path_distribution.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`setup_order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`setup_strategy_filter_stats.csv`、`equity_curve.csv`、`drawdown_curve.csv`、`drawdown_episodes.csv`、`data_inventory.csv`、`symbol_metadata.csv` 和数据审计文件写入页面指定的输出目录。`artifact_manifest.csv` 是产物索引，按阅读优先级说明每个文件回答什么问题；运行后页面会直接展示产物索引，建议先看它再下钻明细。
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
`audit-data` 会同时输出 `invalid_date_rows / invalid_symbol_rows / zero_volume_amount_rows / expected_rows / missing_rows / coverage_ratio / max_missing_gap_minutes / first_missing_at / last_missing_at / max_missing_gap_start_at / max_missing_gap_end_at`；其中 `rows_in_window / missing_rows / coverage_ratio` 按剔除零流动性后的可交易 K 计算，`zero_volume_amount_rows` 单独保留原始零流动性数量。需要在 CLI 里直接看到逐段缺口时，加 `--show-gap-episodes`。实验保存时还会写入 `data_gap_episodes.csv`，按每段连续缺失K列出 `start_at / end_at / missing_rows / gap_minutes / previous_available_at / next_available_at`，便于直接定位要补哪一段缓存。
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
  --side-mode both \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --trailing-take-profit-activation-pct 0.04 \
  --trailing-take-profit-drawdown-pct 0.015 \
  --trailing-take-profit-ma-period 20 \
  --enable-terminal-false-breakout-filter \
  --terminal-false-breakout-detectors trend,channel \
  --terminal-false-breakout-lookback 40 \
  --terminal-false-breakout-atr-period 14 \
  --terminal-false-breakout-min-regime-bars 18 \
  --terminal-false-breakout-extension-atr-multiple 2.0 \
  --terminal-false-breakout-edge-lookback 8 \
  --terminal-false-breakout-edge-pos 0.90 \
  --terminal-false-breakout-edge-min-count 3 \
  --terminal-false-breakout-weak-progress-atr 0.35 \
  --terminal-false-breakout-wick-ratio 0.35 \
  --terminal-false-breakout-min-score 3 \
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

保存目录只包含单策略产物：`artifact_manifest.csv`、`config.json`、`strategy_space.csv`、`stats.json`、`experiment_diagnostics.csv`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`setup_order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`setup_strategy_filter_stats.csv`、`equity_curve.csv`、`drawdown_curve.csv`、`drawdown_episodes.csv`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv`、`data_gap_episodes.csv`、`limit_filter_audit.csv`、
`strategy_stats.csv`、`detector_stats.csv`、`setup_stats.csv`、`signal_lifecycle_stats.csv`、`trade_path_distribution.csv`、`symbol_stats.csv`、`side_stats.csv`、`exit_reason_stats.csv`、`event_type_stats.csv`、`monthly_returns.csv`。
`artifact_manifest.csv` 是产物索引，列出文件名、类别、阅读优先级、要回答的问题和简短说明，适合先判断该看 `strategy_space.csv`、净值回撤、订单拒绝还是数据缺口。
`monthly_returns.csv` 的周期收益以上一条净值作为本期起点，避免漏掉月初第一笔净值变化；同时包含周期内基于 `drawdown_net_value` 的最大回撤和净值样本数。
`stats.json` 会同步写入周期稳定性摘要，例如 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_worst_return_period / monthly_best_return_period / monthly_return_std / monthly_worst_drawdown / monthly_worst_drawdown_period / monthly_max_consecutive_losses / monthly_max_recovery_periods / monthly_current_underwater_periods`，避免参数对比时再手工汇总月度收益、最差月份和连续亏损风险。逐笔交易统计还包含 `breakeven_win_rate / win_rate_edge / win_rate_ci_lower / win_rate_ci_upper / avg_return_standard_error / avg_return_ci_lower / avg_return_ci_upper / positive_expectancy_probability`，用于判断当前赔率下需要多少胜率、实际胜率是否有边际，以及胜率和平均收益是否只是小样本波动；订单摘要包含 `primary_rejected_reason / primary_rejected_reason_count / primary_rejected_reason_rate` 和 `primary_strategy_rejected_reason / primary_strategy_rejected_reason_count / primary_strategy_rejected_reason_rate`，用于先定位最大拒单来源；数据摘要包含 `primary_data_issue / primary_data_issue_count / primary_data_issue_rate`，用于先定位最大数据问题来源；退出原因同步给出 `primary_exit_reason / primary_exit_reason_count / primary_exit_reason_rate / take_profit_exit_count / take_profit_exit_rate / trailing_take_profit_exit_count / trailing_take_profit_exit_rate / stop_loss_exit_count / stop_loss_exit_rate / max_holding_exit_count / max_holding_exit_rate`，方便先定位主要退出原因、数量和占比，再比较止盈、回撤止盈、止损和持有到期占比。
单策略和组合策略不使用旧突破的固定百分比止盈止损控件：结构止损价由各形态识别模块输出，页面会显示“结构止损价说明”；可调的止损参数是“结构止损最大风险”，用于限制开仓价到结构止损价的距离。`risk_reward` 指向平仓信号，不决定开仓信号；开仓信号先给出开仓价和结构止损价，盈亏比只把这段风险距离换算成固定目标平仓价：多头目标平仓价 = 开仓价 + (开仓价 - 止损价) × 盈亏比，空头方向相反。Web 页面可用“启用盈利通道回撤止盈”开关，把成交后的动态回撤止盈显式打开或关闭。
单策略有成交时 `equity_curve.csv` 按回测 K 时间轴输出逐 K 盯市净值；没有成交时保留 `trade_no=0` 的初始资金行。`net_value` 使用正常盯市/平仓后的净值，`drawdown_net_value` 用持仓方向的不利价格估算回撤净值：多头用当前 K 的 low，空头用当前 K 的 high，避免只按开平仓点漏掉持仓期间浮动回撤；组合策略会把同一时刻全部开放持仓合成组合回撤净值，退出 K 也会先按仍持仓状态估算不利价格，再按结算净值确认修复。`drawdown_curve.csv` 会把同一根 K 拆成 `adverse_price` 与 `settlement` 两类点，字段 `path_net_value / drawdown / point_type` 可直接复核回撤曲线；水下 K 数、最大回撤持续 K 数和修复 K 数按唯一 K 线时间点去重，平均回撤和 Ulcer 回撤压力指数按每根 K 的最差回撤计算，不会因为同一根 K 拆成两个估值点而重复加权。自然周期收益和年化统计使用 `net_value`，最大回撤、触底、修复和回撤曲线使用这条价格路径。`stats.json` 同时包含逐笔交易统计和净值曲线统计，例如 `annualized_return / annualized_volatility / equity_sharpe / calmar_ratio / ulcer_index / time_under_water_ratio / market_bar_count / exposure_bars / exposure_bar_ratio / max_drawdown_start_at / max_drawdown_trough_at / max_drawdown_recovery_at / current_drawdown / current_underwater_bars`。单策略按满仓进出，`market_bar_count` 是样本内唯一 K 线时间点数量，`exposure_bar_ratio` 是持仓 K 数占市场时间轴的比例，可直接定位最大回撤和场内时间压力。
`drawdown_episodes.csv` 按回撤幅度从深到浅列出水下区间，包含 `start_at / trough_at / recovery_at / depth / underwater_bars / recovery_bars / recovered`，用于定位每轮回撤从哪根净值高点开始、在哪根触底、用了多少 K 修复，以及当前水下区间是否仍未修复。
`experiment_diagnostics.csv` 把数据覆盖、交易样本、订单接受率、策略过滤、回撤压力、收益质量、胜率边际、正期望概率、退出结构、月度稳定性、路径风险和资金暴露汇总为 `通过 / 关注 / 失败`，用于优先定位该先修数据、调信号、调风控、调胜率边际、调正期望概率、调退出还是调仓位。诊断里的主要原因会优先显示中文说明并保留原因代码，方便直接阅读和回查原始 CSV。
`trade_path_distribution.csv` 按持有 K 数、R 倍数、最大不利 R、最大有利 R 分桶，输出 `trade_count / win_rate / avg_return / avg_r_multiple / avg_mae_r / avg_mfe_r / avg_holding_bars`，用于定位交易质量来自持仓周期、风险暴露还是盈利空间。
`--trailing-take-profit-activation-pct`、`--trailing-take-profit-drawdown-pct` 和 `--trailing-take-profit-ma-period` 是盈利通道回撤止盈参数：启动浮盈是可选门槛，表示实际入场后上一根已完成 K 达到多少浮盈才开始跟踪，设为 `0` 表示不设门槛；最大盈利回撤幅度是比例止盈参数，表示从上一根已完成 K 的最大盈利价位回撤多少退出，例如多头入场 100、最高浮盈到 108，参数 0.020 时回撤线约 105.84，跌破即平仓；空头按最低价后的反弹计算。当前周期均线周期由用户输入 K 数，用当前回测周期上一根已完成 K 的均线作为均线回撤止盈线。三个参数同时为 `0` 表示关闭；最大盈利回撤幅度大于 `0` 或当前周期均线周期至少为 `2` 即可启用，二者可单独使用，也可叠加。当前 K 可以刷新最大盈利价位或均线输入，但不会在同一根 K 里同时完成启动和退出。
`--enable-terminal-false-breakout-filter` 会启用末端假突破开仓过滤。它默认只作用于 `trend,channel`，数学条件为：`terminal_score = 1[regime_run >= min_regime_bars] + 1[abs(close - channel_mid) / ATR >= extension_atr_multiple] + 1[edge_count >= edge_min_count] + 1[progress_atr <= weak_progress_atr] + 1[wick_ratio >= wick_ratio]`，当 `terminal_score >= min_score` 时拒绝开仓。多头用上轨、创新高推进和上影线；空头对称使用下轨、创新低推进和下影线。该过滤只使用信号 K 及之前已完成 K 线，不读取未来 K。
`trades.csv` 保留 `order_id / event_id / event_type / signal_date / signal_bar_index / side / planned_entry_price / stop_price / target_price / risk_per_share / r_multiple / mae_pct / mfe_pct / mae_r / mfe_r / metadata`，
可直接回查每笔成交来自哪根信号 K、哪个形态识别事件和哪类信号形态。
`order_decisions.csv` 记录单策略订单是否 `accepted`，以及 `invalid_order`、`duplicate_order_id`、`no_fill`、`no_liquidity`、`already_open`、`no_bars`、`actual_risk_too_high`、`chase_too_far`、`target_not_favorable` 等未成交原因；同时写入 `actual_entry_price / actual_risk_pct / actual_chase_pct / actual_reward_to_risk`，用于解释坏订单字段、重复订单身份、零流动性入场、跳空成交、过度追价和目标价失效。
`strategy_filter_decisions.csv` 记录策略层过滤结果，例如形态输出观察/中部不交易方向、交易方向过滤、信号 K 无流动性、大周期方向不一致、无可用大周期上下文、上下文过旧或末端假突破风险；基础策略过滤、大周期方向过滤和末端假突破过滤会叠加保留，它早于撮合层，不和 `order_decisions.csv` 混用。
`limit_filter_audit.csv` 记录日 K 一字涨停过滤是否真实执行；严格模式下日线缺失或损坏会直接失败，只有显式关闭严格数据质量检查时才会继续输出 `daily_missing / daily_read_error / daily_missing_columns / daily_quality_error` 审计。
`order_decision_stats.csv` 和 `strategy_filter_stats.csv` 按策略、状态和原因聚合决策分布；`setup_order_decision_stats.csv` 和 `setup_strategy_filter_stats.csv` 按 `detector_name / event_type / side` 进一步拆解信号形态的撮合失败、风控拒绝和策略过滤拒绝。`decision_rate` 表示占全部决策的比例，`group_decision_rate` 表示在当前策略、信号形态或过滤器组内的比例。订单聚合表还会汇总止损风险、追价和实际盈亏比，用于定位哪类参数在撮合层失效。
`strategy_stats.csv` 按 `strategy_name` 汇总策略成交绩效；已启用但没有成交的策略也会保留零值行，方便区分“策略无信号/全被拒”和“策略未参与”。`detector_stats.csv` 按 `detector_name` 独立汇总趋势、区间、通道、反转的成交绩效；已启用但没有成交的识别模块也会保留零值行，方便区分“模块无机会/全被拒”和“模块未参与”。`setup_stats.csv` 按 `detector_name / event_type / side` 汇总标志 K、失败突破、通道突破、H1/H2/L1/L2 等信号形态的多空表现，只有信号但全部被撮合拒绝或策略过滤的 setup 也会保留零值行。`signal_lifecycle_stats.csv` 按 `detector_name / event_type / side / exit_reason` 汇总开平仓路径绩效，用来查看同一类开仓信号最后主要死在止损、持有到期，还是通过固定目标或盈利通道回撤止盈退出。`symbol_stats.csv` 优先给出 `stock_name`，同时保留 `stock_code` 作为复核键；即使某只样本股票没有成交，也会保留零值统计行，方便确认它参与了回测但未产生交易。单策略和组合回测的分组统计都会给出 `return_contribution / return_per_exposure_bar`，用于拆解策略、识别模块、信号形态、标的和方向的收益贡献及单位持仓 K 效率；组合回测还会给出 `capital_turnover / capital_weighted_raw_return / return_per_capital_exposure_bar / return_per_margin_exposure_bar`，按仓位或保证金占用衡量资金效率。`capital_exposure_bars / margin_exposure_bars` 按仓位或保证金占用乘以持仓 K 数，衡量长期占资压力。组合 `stats.json` 还会从逐 K 净值曲线计算 `avg_cash_ratio / min_cash_ratio / max_cash_ratio / avg_margin_exposure / max_margin_exposure / avg_net_exposure / min_net_exposure / max_net_exposure`，用于判断现金拖累、保证金压力、空头资金占用和多空偏向。
`data_inventory.csv` 保存本次实验涉及的日 K、主周期和高周期 parquet 缓存快照，包含是否存在、行数、起止时间、文件大小、缺失字段、修改时间和路径；`symbol_metadata.csv` 保存股票代码、股票名称和名称来源，优先读取行情目录 `symbols.csv / stock_names.csv`，其次读取 TDX `hq_cache` 的 `shm/szm/bjm.tnf`；`stats.json` 同步写入 `data_inventory_row_count / data_inventory_cached_count / data_inventory_unavailable_count / data_inventory_missing_file_count / data_inventory_read_error_count / data_inventory_missing_columns_count / data_inventory_no_valid_rows_count / data_inventory_signature` 等摘要，其中 `data_inventory_signature` 是不含本机绝对路径和文件修改时间的数据快照指纹，便于 Mac/Win 对照同一批缓存。
`stats.json` 同步写入 `order_count / accepted_order_count / rejected_order_count / acceptance_rate / rejected_no_fill_count / rejected_no_liquidity_count / rejected_no_bars_count / rejected_invalid_order_count / rejected_duplicate_order_id_count / rejected_already_open_count`
以及已接受订单的平均/最大 `capital_fraction / risk_fraction / margin_fraction`，若启用策略层过滤，还会写入 `strategy_signal_count / strategy_filter_acceptance_rate / strategy_rejected_signal_bar_no_liquidity_count / strategy_rejected_higher_timeframe_mismatch_count` 等摘要。数据审计摘要也会进入同一个文件，包括 `primary_data_issue / primary_data_issue_count / primary_data_issue_rate / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / data_audit_missing_file_count / data_audit_missing_columns_count / data_audit_no_window_data_count / data_audit_read_error_count / limit_filter_failed_count / limit_filter_daily_missing_count / limit_filter_daily_read_error_count / limit_filter_daily_missing_columns_count / limit_filter_daily_quality_error_count / limit_filter_filtered_days`。
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
  --side-mode both \
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
  --trailing-take-profit-activation-pct 0.04 \
  --trailing-take-profit-drawdown-pct 0.015 \
  --trailing-take-profit-ma-period 20 \
  --output-dir runs/case-001 \
  --benchmark
```

保存目录会包含 `artifact_manifest.csv`、`config.json`、`strategy_space.csv`、`stats.json`、`experiment_diagnostics.csv`、`trades.csv`、`order_decisions.csv`、`order_decision_stats.csv`、`setup_order_decision_stats.csv`、`strategy_filter_decisions.csv`、`strategy_filter_stats.csv`、`setup_strategy_filter_stats.csv`、`equity_curve.csv`、`drawdown_curve.csv`、`drawdown_episodes.csv`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv`、`data_gap_episodes.csv`、`limit_filter_audit.csv`、
`strategy_stats.csv`、`detector_stats.csv`、`setup_stats.csv`、`signal_lifecycle_stats.csv`、`trade_path_distribution.csv`、`symbol_stats.csv`、`side_stats.csv`、`exit_reason_stats.csv`、`event_type_stats.csv`、`monthly_returns.csv` 和可选 `benchmark.json`。
`artifact_manifest.csv` 是产物索引，列出文件名、类别、阅读优先级、要回答的问题和简短说明，适合先判断该看 `strategy_space.csv`、净值回撤、订单拒绝还是数据缺口。
`monthly_returns.csv` 的周期收益以上一条净值作为本期起点，避免漏掉月初第一笔净值变化；同时包含周期内基于 `drawdown_net_value` 的最大回撤和净值样本数。
`stats.json` 会同步写入周期稳定性摘要，例如 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_worst_return_period / monthly_best_return_period / monthly_return_std / monthly_worst_drawdown / monthly_worst_drawdown_period / monthly_max_consecutive_losses / monthly_max_recovery_periods / monthly_current_underwater_periods`，用于比较不同形态、仓位参数和大周期方向过滤下的月度稳定性。逐笔交易统计还包含 `breakeven_win_rate / win_rate_edge / win_rate_ci_lower / win_rate_ci_upper / avg_return_standard_error / avg_return_ci_lower / avg_return_ci_upper / positive_expectancy_probability`，用于判断当前赔率下需要多少胜率、实际胜率是否有边际，以及胜率和平均收益是否只是小样本波动；订单摘要包含 `primary_rejected_reason / primary_rejected_reason_count / primary_rejected_reason_rate` 和 `primary_strategy_rejected_reason / primary_strategy_rejected_reason_count / primary_strategy_rejected_reason_rate`，用于先定位最大拒单来源；数据摘要包含 `primary_data_issue / primary_data_issue_count / primary_data_issue_rate`，用于先定位最大数据问题来源；退出原因同步给出 `primary_exit_reason / primary_exit_reason_count / primary_exit_reason_rate / take_profit_exit_count / take_profit_exit_rate / trailing_take_profit_exit_count / trailing_take_profit_exit_rate / stop_loss_exit_count / stop_loss_exit_rate / max_holding_exit_count / max_holding_exit_rate`，方便先定位主要退出原因、数量和占比，再比较止盈、回撤止盈、止损和持有到期占比。
单策略和组合策略的固定目标平仓价由信号 K 止损距离乘以 `risk_reward` 得出，Web 页面不会再展示旧突破专用的固定百分比止盈止损输入，避免参数改了但现代策略不生效；结构止损价仍在每笔订单和 K 线图止损横线中展示，可调止损参数是“结构止损最大风险”。
组合仓位容量按实际 `entry_date` 分配，不按信号时间提前占用资金；风险预算按真实 `entry_price / risk_per_share` 计算，不用计划挂单价低估跳空成交风险。
`--capital-per-trade` 是固定单笔名义仓位，`--risk-per-trade` 是按真实入场风险反推仓位；两者都为空时按最大持仓数均分。
`--reserve-cash` 预留现金，`--allow-same-symbol-overlap` 允许同一股票多策略重叠持仓。
趋势、区间、通道、反转的识别参数各自传入 detector，组合层只负责排序、容量和资金分配。
`--strategy-priority`、`--strategy-capital-limit`、`--sector-capital-limit` 和 `--symbol-sector-map` 使用 `key=value,key=value` 格式，分别控制策略排序、策略资金上限、行业资金上限和股票所属行业。
`order_decisions.csv` 记录组合层接受或拒绝每个候选订单的原因，例如 `invalid_order`、`duplicate_order_id`、`no_fill`、`no_liquidity`、`no_bars`、`actual_risk_too_high`、`chase_too_far`、`target_not_favorable`、`max_open_positions`、`same_symbol_overlap`、`no_capital`；组合容量或资金拒绝也会保留候选成交的 `actual_entry_price / actual_risk_pct / actual_chase_pct / actual_reward_to_risk`。
`strategy_filter_decisions.csv` 记录订单进入组合撮合前被策略过滤的原因，包括观察/中部不交易方向、交易方向过滤、信号 K 无流动性、大周期方向不一致和末端假突破风险；包装策略会保留内层过滤日志，便于单策略回测和组合回测分别定位问题。
`limit_filter_audit.csv` 记录日线过滤状态；严格模式下 `daily_missing / daily_read_error / daily_missing_columns / daily_quality_error` 会中止回测，关闭严格数据质量检查排查时重点看这些状态和 `filtered_days`。
`order_decision_stats.csv` 和 `strategy_filter_stats.csv` 分别汇总撮合层与策略门控层的决策分布；`setup_order_decision_stats.csv` 和 `setup_strategy_filter_stats.csv` 用同一口径按 setup 继续拆分拒绝结构；`decision_rate` 是全局占比，`group_decision_rate` 是当前策略、setup 或过滤器组内占比。撮合层聚合表同时保留最终成交订单口径和触发成交候选口径：`avg_accepted_actual_risk_pct / avg_accepted_actual_chase_pct / avg_accepted_actual_reward_to_risk` 只看最终接受的订单，`avg_executed_actual_risk_pct / avg_executed_actual_chase_pct / avg_executed_actual_reward_to_risk` 则包含已经触发成交价但后续被容量、资金或风控拒绝的候选单。
手续费率、滑点 bps 和初始资金会写入 `config.json`，并直接传给单策略、组合策略和参数遍历撮合层。
组合回测同样支持 `--trailing-take-profit-activation-pct / --trailing-take-profit-drawdown-pct / --trailing-take-profit-ma-period`，口径也是实际成交后上一根已完成 K 确认；退出原因会在 `trades.csv` 和 `exit_reason_stats.csv` 中显示为 `trailing_take_profit`，并在 `stats.json`、`sweep.csv` 和 `parameter_summary.csv` 中同步汇总盈利通道回撤止盈退出次数和占比。
`--benchmark` 复用本次组合回测结果生成 `benchmark.json`，不再重复加载数据或重复撮合。
`stats.json` 同时保存逐笔交易指标和净值曲线指标：`annualized_return`、`annualized_volatility`、
`annualized_sharpe`、`annualized_sortino`、`calmar_ratio`、`avg_drawdown`、`ulcer_index`、`time_under_water_ratio`、`avg_gross_exposure`、`max_gross_exposure`、`avg_margin_exposure`、`max_margin_exposure`、
`exposure_bar_ratio`、`avg_open_positions`、`max_open_positions`、`avg_cash_ratio`、`min_cash_ratio`、`max_cash_ratio`、`avg_net_exposure`、`min_net_exposure`、`max_net_exposure`、`max_drawdown_start_at`、`max_drawdown_trough_at`、`max_drawdown_recovery_at`、`current_drawdown`、`current_underwater_bars`、`avg_r_multiple`、`r_profit_factor`、`system_quality_number`、`avg_mae_pct`、`avg_mfe_pct`、
`return_p05`、`return_p25`、`return_p50`、`return_p75`、`return_p95`、`cvar_95`、`win_rate_ci_lower`、`win_rate_ci_upper`、`avg_return_standard_error`、`avg_return_ci_lower`、`avg_return_ci_upper`、`positive_expectancy_probability`、`return_per_exposure_bar`、`capital_exposure_bars`、`margin_exposure_bars`、`return_per_capital_exposure_bar`、`return_per_margin_exposure_bar`、
以及 `order_count / accepted_order_count / rejected_order_count / executed_order_count / accepted_executed_order_count / acceptance_rate / rejection_rate /
rejected_invalid_order_count / rejected_duplicate_order_id_count / rejected_max_open_positions_count / rejected_no_capital_count / rejected_actual_risk_too_high_count / rejected_chase_too_far_count / rejected_target_not_favorable_count /
avg_accepted_actual_risk_pct / max_accepted_actual_risk_pct / avg_accepted_actual_chase_pct / max_accepted_actual_chase_pct / avg_accepted_actual_reward_to_risk / min_accepted_actual_reward_to_risk /
avg_executed_actual_risk_pct / max_executed_actual_risk_pct / avg_executed_actual_chase_pct / max_executed_actual_chase_pct / avg_executed_actual_reward_to_risk / min_executed_actual_reward_to_risk`
等订单摘要、策略过滤摘要和已接受订单的资金、风险、保证金占用统计。数据审计摘要也会进入同一个文件，包括 `primary_data_issue / primary_data_issue_count / primary_data_issue_rate / data_inventory_signature / data_inventory_unavailable_count / data_inventory_missing_columns_count / data_inventory_no_valid_rows_count / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / data_audit_missing_file_count / data_audit_missing_columns_count / data_audit_no_window_data_count / data_audit_read_error_count / limit_filter_failed_count / limit_filter_daily_missing_count / limit_filter_daily_read_error_count / limit_filter_daily_missing_columns_count / limit_filter_daily_quality_error_count / limit_filter_filtered_days`。

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
  --side-mode long_only \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --trailing-take-profit-activation-pct 0.04 \
  --trailing-take-profit-drawdown-pct 0.015 \
  --trailing-take-profit-ma-period 20 \
  --output-dir runs/single-sweep-001
```

`single-sweep` 只绑定一个 detector，不进入组合仓位分配层；一次加载数据，多组参数复用同一批 K 线，订单参数不变时复用已生成订单。
除固定的 `--risk-rewards / --max-holding-bars-list` 外，也可以重复传 `--grid 字段=值1,值2` 遍历任意实验配置字段，例如
`--grid side_mode=both,long_only --grid range_min_score=0.7,0.9 --grid fee_rate=0,0.0003`；布尔字段用 `true/false`。
重复的 grid 值会在笛卡尔积展开前按稳定指纹去重，避免重复配置进入订单生成和回测热路径；`summary.json` 仍保留原始 `grid_case_count` 和去重后的 `case_count`。
固定启用 detector 的 sweep 只保留已启用 detector 相关参数、通用撮合参数和实际启用的高周期 trend 上下文字段；只改未启用模块参数时，不会生成额外 case，也不会改变 `case_config_hash`。
结果按收益、回撤、月度稳定性、交易数和 case 名稳定排序保存 `sweep.csv`，首列 `sweep_rank` 是可直接筛选的参数排名；`pareto_rank=1` 表示按收益、回撤、Ulcer、月度最差收益、月度收益波动和交易样本数得到的第一层非支配候选集，分层时使用 NumPy 批量支配矩阵，避免逐候选嵌套比较拖慢大网格。`risk_adjusted_score` 是 0-100 的风险质量评分：用收益、组合回撤、月度最差收益、月度波动、Ulcer、单位持仓 K 效率和交易样本数的分位得分加权，再扣除诊断失败/关注惩罚；`risk_adjusted_rank` 是独立风险质量排名，不替换 `sweep_rank`。`case_config_hash` 是实验配置的 SHA-256 指纹，但不包含本机 `name / data_root / output_dir`；`data_inventory_signature` 是本次 K 线缓存快照指纹，方便跨机器复现和对照。同时保存 `artifact_manifest.csv`、`config.json`、`summary.json`、`pareto.csv`、`parameter_summary.csv`、`case_diagnostics.csv`、`case_strategy_stats.csv`、`case_detector_stats.csv`、`case_setup_stats.csv`、`case_symbol_stats.csv`、`case_setup_order_decision_stats.csv`、`case_setup_strategy_filter_stats.csv`、`case_configs.jsonl`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv`、`data_gap_episodes.csv` 和 `limit_filter_audit.csv`；`artifact_manifest.csv` 是产物索引，先按阅读优先级提示看 `sweep.csv`、`pareto.csv`、`parameter_summary.csv` 还是数据质量文件。`summary.json` 汇总 `grid_case_count / case_count / pareto_case_count / best_case_name / best_case_config_hash / best_risk_adjusted_case_name / best_risk_adjusted_score / avg_risk_adjusted_score / median_risk_adjusted_score / worst_risk_adjusted_score / data_inventory_signature / primary_exit_reason / primary_exit_reason_count / primary_exit_reason_rate / primary_data_issue / primary_data_issue_count / primary_data_issue_rate / data_inventory_unavailable_count / data_inventory_missing_columns_count / data_inventory_no_valid_rows_count / order_cache_hit_rate / data_coverage_below_min_count / data_weighted_coverage_ratio / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / limit_filter_filtered_days / case_diagnostic_failed_count / case_diagnostic_attention_count / case_diagnostic_failed_case_count / case_strategy_row_count / case_strategy_trade_count / case_strategy_zero_trade_row_count / case_detector_row_count / case_detector_trade_count / case_detector_zero_trade_row_count / case_setup_row_count / case_setup_trade_count / case_setup_zero_trade_row_count / case_symbol_row_count / case_symbol_trade_count / case_symbol_zero_trade_row_count / case_setup_order_decision_count / case_setup_order_rejected_count / case_setup_strategy_filter_decision_count / case_setup_strategy_filter_rejected_count`，其中 `grid_case_count` 是原始笛卡尔积组合数，`case_count` 是去重后实际运行数，`pareto.csv` 只保留第一层 Pareto 候选，`parameter_summary.csv` 按 grid 字段和值聚合风险质量评分、收益、回撤、月度稳定性、盈亏平衡胜率、胜率边际、持仓 K 效率、仓位/保证金效率、止盈/回撤止盈/止损/持有到期退出比例、撮合/策略过滤接受率、参数遍历成交质量、Pareto 命中率、正收益率和收益离散度，避免只看平均收益选参数；参数遍历成交质量包含 `avg_accepted_actual_risk_pct / avg_accepted_actual_chase_pct / avg_accepted_actual_reward_to_risk / avg_executed_actual_risk_pct / avg_executed_actual_chase_pct / avg_executed_actual_reward_to_risk`，用于区分最终成交质量和触发成交价但被拒的候选质量。`case_diagnostics.csv` 按 `case_config_hash` 下钻每个参数组的数据覆盖、样本量、接受率、回撤、收益质量、路径风险和资金暴露诊断；`sweep.csv` 同步带 `diagnostic_failed_count / diagnostic_attention_count / diagnostic_max_severity / diagnostic_primary_issue`，便于先筛掉失败项。`case_strategy_stats.csv` 和 `case_detector_stats.csv` 按 `case_config_hash` 下钻每个参数组的策略与识别模块绩效，已启用但无成交的策略或模块也保留零值行；`case_setup_stats.csv` 按 `case_config_hash / detector_name / event_type / side` 下钻每个参数组的 setup 绩效，没有成交但出现过信号或拒单的 setup 也保留零值行，`case_symbol_stats.csv` 按 `case_config_hash / stock_name / stock_code` 下钻每个参数组的股票表现，未成交样本也保留零值行，`case_setup_order_decision_stats.csv` 和 `case_setup_strategy_filter_stats.csv` 则按同一 setup 维度拆分撮合拒绝和策略过滤拒绝。`case_configs.jsonl` 按 `sweep.csv` 排名顺序逐行保存完整 case 配置。每行会带 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_return_std / monthly_max_consecutive_losses / monthly_max_recovery_periods`、`breakeven_win_rate / win_rate_edge`、`take_profit_exit_rate / trailing_take_profit_exit_rate / stop_loss_exit_rate / max_holding_exit_rate`、`primary_exit_reason / primary_exit_reason_count / primary_exit_reason_rate / primary_data_issue / primary_data_issue_count / primary_data_issue_rate / data_inventory_signature / data_inventory_cached_count / data_inventory_unavailable_count / data_inventory_missing_columns_count / data_inventory_no_valid_rows_count / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / limit_filter_filtered_days`，参数结果不脱离周期稳定性和数据质量语境。

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
  --side-mode both \
  --min-coverage-ratio 0.95 \
  --fee-rate 0.0003 \
  --slippage-bps 5 \
  --initial-equity 1 \
  --max-open-positions-list 3,5 \
  --trailing-take-profit-activation-pct 0.04 \
  --trailing-take-profit-drawdown-pct 0.015 \
  --trailing-take-profit-ma-period 20 \
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
固定参数可直接用 `--trailing-take-profit-activation-pct / --trailing-take-profit-drawdown-pct / --trailing-take-profit-ma-period`；关闭时三个参数都设为 `0`，启用时最大盈利回撤幅度大于 `0` 或均线周期至少为 `2` 即可，启动浮盈只作为可选过滤门槛。遍历盈利通道回撤止盈时不要把关闭值和开启值在多个 grid 里随意交叉，可先固定开启后遍历正数区间，关闭状态单独跑一组。
重复的 grid 值会在展开前去重，未启用 detector 的参数也会被过滤，避免同一组合重复生成订单、候选成交或组合分配结果。
参数笛卡尔积结果会按收益、回撤、月度稳定性、交易数和 case 名稳定排序保存 `sweep.csv`，首列 `sweep_rank` 是参数排名；`pareto_rank` 用收益、回撤、Ulcer、月度最差收益、月度收益波动和交易样本数标记非支配层级；`case_config_hash` 记录去除本机路径和实验名后的 case 配置指纹，`data_inventory_signature` 记录本次 K 线缓存快照指纹。
保存目录同时包含 `artifact_manifest.csv`、`config.json`、`summary.json`、`pareto.csv`、`parameter_summary.csv`、`case_diagnostics.csv`、`case_strategy_stats.csv`、`case_detector_stats.csv`、`case_setup_stats.csv`、`case_symbol_stats.csv`、`case_setup_order_decision_stats.csv`、`case_setup_strategy_filter_stats.csv`、`case_configs.jsonl`、`data_inventory.csv`、`symbol_metadata.csv`、`data_coverage.csv`、`data_gap_episodes.csv` 和 `limit_filter_audit.csv`；`artifact_manifest.csv` 是产物索引，先按阅读优先级提示看 `sweep.csv`、`pareto.csv`、`parameter_summary.csv` 还是数据质量文件。`config.json` 写入基础配置和 `sweep_grid`，`summary.json` 写入原始组合数、去重后实际 case 数、最佳 case、数据快照指纹、主要退出原因、Pareto 候选数量、订单/候选缓存命中率、数据覆盖率、缓存不可用数、最长连续数据缺口、日 K 涨停开盘过滤摘要、case 诊断失败/关注数量和信号形态决策摘要，`pareto.csv` 写入 `pareto_rank=1` 的候选集合，`parameter_summary.csv` 按每个 grid 参数值聚合 `case_count / pareto_case_count / pareto_hit_rate / positive_return_case_count / positive_return_rate / std_total_return / best_total_return / worst_total_return / best_sweep_rank / avg_risk_adjusted_score / median_risk_adjusted_score / avg_total_return / avg_max_drawdown / avg_monthly_worst_return / avg_monthly_return_std / avg_breakeven_win_rate / avg_win_rate_edge / avg_return_per_exposure_bar / avg_return_per_capital_exposure_bar / avg_return_per_margin_exposure_bar / avg_take_profit_exit_rate / avg_trailing_take_profit_exit_rate / avg_stop_loss_exit_rate / avg_max_holding_exit_rate / avg_acceptance_rate / avg_rejection_rate / avg_rejected_no_fill_count / avg_accepted_actual_risk_pct / avg_accepted_actual_chase_pct / avg_accepted_actual_reward_to_risk / avg_executed_actual_risk_pct / avg_executed_actual_chase_pct / avg_executed_actual_reward_to_risk / avg_strategy_filter_acceptance_rate / avg_strategy_filter_rejection_rate`，其中这组参数遍历成交质量字段可直接比较最终成交质量和候选订单质量。`case_diagnostics.csv` 保留每个 case 的诊断明细，`sweep.csv` 同步带 `risk_adjusted_rank / risk_adjusted_score / diagnostic_failed_count / diagnostic_attention_count / diagnostic_max_severity / diagnostic_primary_issue`，用于按风险状态筛选参数组。`case_strategy_stats.csv` 和 `case_detector_stats.csv` 按 case 排名保留策略级与识别模块级绩效，`case_setup_stats.csv` 按 case 排名保留信号形态级绩效，`case_symbol_stats.csv` 按 case 排名保留股票名称、代码和标的级绩效，两张 `case_setup_*_decision_stats.csv` 按 case 排名保留信号形态级撮合拒绝和策略过滤拒绝。`case_configs.jsonl` 写入每个参数组的完整配置。`sweep.csv` 会附带 `monthly_count / monthly_win_rate / monthly_worst_return / monthly_return_std / monthly_max_consecutive_losses / monthly_max_recovery_periods`、`breakeven_win_rate / win_rate_edge`、`take_profit_exit_count / trailing_take_profit_exit_count / stop_loss_exit_count / max_holding_exit_count`、`primary_exit_reason / primary_exit_reason_count / primary_exit_reason_rate`、`order_cache_status / candidate_cache_status / generated_order_count / candidate_count / candidate_rejection_count`、`order_count / acceptance_rate / rejected_no_fill_count`、`data_inventory_signature / data_inventory_cached_count / data_inventory_unavailable_count / data_inventory_missing_columns_count / data_inventory_no_valid_rows_count / data_min_coverage_threshold / data_coverage_below_min_count / data_weighted_coverage_ratio / data_coverage_p05 / data_coverage_p50 / data_coverage_p95 / data_max_missing_gap_minutes / data_max_missing_gap_start_at / data_max_missing_gap_end_at / data_missing_rows / data_audit_failed_count / limit_filter_filtered_days` 等周期稳定性、性能、订单决策和数据质量摘要，方便解释参数组表现；如果只改变 `higher_timeframe_max_age_minutes`，会重新生成大周期方向过滤后的订单，不复用旧订单。

从遍历结果回放某个 case：

```bash
python -m trending_winning.cli replay-case \
  --case-configs runs/sweep-001/case_configs.jsonl \
  --case-config-hash <sweep.csv里的case_config_hash> \
  --output-dir runs/replay-001
```

`replay-case` 会读取完整 case 配置并运行对应的单策略或组合回测；回放前会重新计算配置指纹，发现 `case_config_hash` 与交易逻辑配置不一致时拒绝回放，本机路径和实验名不参与指纹；传入 `--output-dir` 时会保存新的 `config.json / stats.json / trades.csv / equity_curve.csv / drawdown_curve.csv` 等回测产物。

## Docker

```bash
docker build -t trending-winning .
docker run --rm -p 8520:8501 \
  -v /Users/a1234/Desktop/trend-backtest/data:/data \
  trending-winning
```

容器内默认可把页面数据目录改成 `/data/market/daily`。TDX 真取数仍建议在 Windows/Parallels 侧运行；Docker 更适合读取已落地 parquet 后做扫描和回测。
