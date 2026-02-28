"""
回测入口 - 运行双均线策略的历史回测

使用方式：
    python main_backtest.py

功能：
  1. 通过 AKShare 下载/加载缓存历史数据（贵州茅台 + 宁德时代）
  2. 计算技术指标（MA/MACD/布林带等）
  3. 运行双均线主策略回测（唯一实际下单的策略）
  4. 风控模块接入（仓位限制 + 止损 + 最大回撤）
  5. 控制台打印绩效摘要
  6. 生成可视化图表（需要 matplotlib）

注意：
  - 回测仅验证策略逻辑，不做参数优化
  - MACD 和布林带是观察策略，回测中不参与下单
  - 修改参数只需编辑 config.py
"""
import sys
import os
import logging
from datetime import datetime

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from backtest.engine import BacktestEngine
from risk.manager import RiskManager
from report.plotter import plot_backtest_result

# ----------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------

def _setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        config.LOG_DIR,
        f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ----------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------

def _validate_config():
    """启动前校验 config.py 的关键参数格式，发现问题立即报错"""
    errors = []

    # 日期格式检查（只检查格式，不干预用户设定的值）
    for field, val in [("BACKTEST_START", config.BACKTEST_START),
                       ("BACKTEST_END",   config.BACKTEST_END)]:
        if len(str(val)) != 8 or not str(val).isdigit():
            errors.append(
                f"  {field} = '{val}' 格式错误，应为 8 位数字，如 '20220101'"
            )

    # 开始必须早于结束
    if not errors and config.BACKTEST_START >= config.BACKTEST_END:
        errors.append(
            f"  BACKTEST_START ({config.BACKTEST_START}) "
            f"必须早于 BACKTEST_END ({config.BACKTEST_END})"
        )

    if errors:
        print("\n[配置错误] config.py 存在以下问题，请修正后重新运行：")
        for e in errors:
            print(e)
        sys.exit(1)


def main():
    _validate_config()
    logger = _setup_logging()

    print("=" * 60)
    print("  A股量化交易系统 - 回测模式")
    print(f"  策略  : 双均线 MA{config.MA_FAST}/MA{config.MA_SLOW}")
    print(f"  标的  : {', '.join(config.SYMBOLS_RAW)}")
    print(f"  区间  : {config.BACKTEST_START} ~ {config.BACKTEST_END}")
    print(f"  资金  : {config.INITIAL_CAPITAL:,} 元")
    print("=" * 60)

    # 1. 下载/加载历史数据
    # 注意：为确保 MA60 等长周期指标有足够预热数据，
    #       实际加载范围从缓存起始日期（2022-01-01）开始，
    #       引擎内部通过 start_date 过滤实际回测交易日。
    logger.info("正在加载历史数据...")
    dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
    # 预热窗口：取 BACKTEST_START 往前至少 MA_TREND×2 个日历日
    warmup_days  = max(getattr(config, "MA_TREND", 0), 60) * 2
    load_start   = (
        pd.to_datetime(config.BACKTEST_START) - pd.Timedelta(days=warmup_days)
    ).strftime("%Y%m%d")
    data = dm.get_multi(
        symbols    = config.SYMBOLS_RAW,
        start_date = load_start,
        end_date   = config.BACKTEST_END,
    )

    if not data:
        logger.error("数据加载失败，请检查网络或股票代码")
        sys.exit(1)

    # 检查是否有股票的数据覆盖不到回测结束日，给出明确提示
    bt_end_ts = pd.to_datetime(config.BACKTEST_END)
    for sym, df in data.items():
        name      = config.SYMBOL_NAMES.get(sym, sym)
        actual_end = df.index[-1]
        logger.info(
            f"  {name}({sym}): {len(df)} 个交易日，"
            f"{df.index[0].date()} ~ {actual_end.date()}"
        )
        if actual_end < bt_end_ts:
            logger.warning(
                f"  ⚠ {name}({sym}) 数据只到 {actual_end.date()}，"
                f"未能覆盖回测结束日 {config.BACKTEST_END}。"
                f"（网络不通或东方财富限流，已使用现有缓存继续回测）"
            )

    # 2. 初始化策略
    strategy = MACrossStrategy(symbols=config.SYMBOLS_RAW)

    # 3. 初始化风控
    risk_manager = RiskManager(initial_capital=config.INITIAL_CAPITAL)

    # 4. 运行回测
    logger.info("开始回测...")
    engine = BacktestEngine()
    result = engine.run(
        strategy     = strategy,
        data         = data,
        start_date   = config.BACKTEST_START,
        end_date     = config.BACKTEST_END,
        risk_manager = risk_manager,
    )

    # 5. 打印结果
    result.print_summary()

    # 6. 生成图表
    logger.info("正在生成回测图表...")
    chart_path = plot_backtest_result(result, show=True)
    if chart_path:
        print(f"\n图表已保存: {chart_path}")

    # 7. 保存交易记录
    if result.trades:
        os.makedirs(config.REPORT_DIR, exist_ok=True)
        records_path = os.path.join(
            config.REPORT_DIR,
            f"trades_{strategy.name}_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        trades_df = pd.DataFrame([
            {
                "日期":   t.date.strftime("%Y-%m-%d"),
                "股票":   config.SYMBOL_NAMES.get(t.symbol, t.symbol),
                "代码":   t.symbol,
                "操作":   "买入" if t.action == "BUY" else "卖出",
                "数量":   int(t.quantity),
                "价格":   round(t.price, 2),
                "金额":   round(t.amount, 2),
                "手续费": round(t.commission, 2),
                "盈亏":   round(t.pnl, 2) if t.action == "SELL" else "",
                "原因":   t.reason,
            }
            for t in result.trades
        ])
        trades_df.to_csv(records_path, index=False, encoding="utf-8-sig")
        print(f"交易记录已保存: {records_path}")

        # 8. 自动生成 K 线买卖点图
        logger.info("正在生成 K 线买卖点图...")
        try:
            from plot_trades import draw_kline_with_trades, _set_chinese_font
            _set_chinese_font()
            kline_paths = []
            for symbol in config.SYMBOLS_RAW:
                sym_trades = trades_df[trades_df["代码"] == symbol]
                if sym_trades.empty:
                    continue
                sym_trades = sym_trades.copy()
                sym_trades["日期"] = pd.to_datetime(sym_trades["日期"])
                name = config.SYMBOL_NAMES.get(symbol, symbol)
                kline_path = draw_kline_with_trades(
                    symbol   = symbol,
                    name     = name,
                    ohlcv    = data[symbol],
                    trades   = sym_trades,
                    save_dir = config.REPORT_DIR,
                )
                kline_paths.append(kline_path)
            for p in kline_paths:
                print(f"K 线图已保存: {p}")
        except Exception as e:
            logger.error(f"K 线图生成失败: {e}")

    print("\n回测完成。")


if __name__ == "__main__":
    main()
