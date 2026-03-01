"""
实盘入口 - 掘金量化标准结构

使用方式：
    python main_live.py

必须先在 config.py 中填写：
    GM_TOKEN    = "你的掘金Token"
    ACCOUNT_ID  = "你的账户ID"
    STRATEGY_ID = "你的策略ID"
    TRADE_MODE  = "paper"    # 模拟盘（推荐先用模拟盘验证）
                 # "live"    # 实盘（真实下单，请谨慎！）
    SIGNAL_MODE = "notify"   # 仅提醒（红色弹窗+声音，不自动下单）
                 # "auto"    # 自动下单（配合 TRADE_MODE 使用）

掘金量化回调函数说明：
    init(context)           - 策略初始化，设置Token，订阅K线
    on_bar(context, bars)   - 每根K线结束后触发（30分钟K线）
    on_order_status(...)    - 委托回报（成交/撤单/拒单）
    on_error(...)           - 错误回调

信号执行逻辑：
    SIGNAL_MODE = "notify":
        主策略（MACross）  → 生成信号 → 红色醒目提醒+声音+弹窗 → 不下单
    SIGNAL_MODE = "auto":
        主策略（MACross）  → 生成信号 → 调用 broker 下单 → 成功后提醒
    观察策略（MACD/布林带） → 生成信号 → 仅打印日志，不下单
"""
import sys
import os
import logging
import pandas as pd
from datetime import datetime

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from strategy.macd import MACDStrategy
from strategy.boll import BollStrategy
from risk.manager import RiskManager
from broker.paper import PaperBroker
from notifier import notify_signal

# 尝试导入掘金实盘 Broker（未安装 SDK 时降级为模拟盘）
try:
    from broker.gm import GmBroker
    GM_AVAILABLE = True
except Exception:
    GM_AVAILABLE = False

# 掘金量化 SDK
try:
    from gm.api import (
        set_token,
        subscribe,
        run,
        history_n,
        get_cash,
        get_position,
        MODE_LIVE,
        MODE_BACKTEST,
    )
    GM_SDK_AVAILABLE = True
except ImportError:
    GM_SDK_AVAILABLE = False

# ----------------------------------------------------------------
# 全局对象（在 init 中初始化，在 on_bar 中使用）
# ----------------------------------------------------------------
_strategy_main:   MACrossStrategy = None
_strategy_macd:   MACDStrategy    = None
_strategy_boll:   BollStrategy    = None
_risk_manager:    RiskManager     = None
_broker           = None
_data_manager:    DataManager     = None
_logger:          logging.Logger  = None


# ----------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        config.LOG_DIR,
        f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("main_live")


# ----------------------------------------------------------------
# 掘金回调函数
# ----------------------------------------------------------------

def init(context):
    """
    策略初始化（掘金在策略启动时调用一次）

    职责：
      1. 设置 Token（鉴权）
      2. 订阅30分钟K线
      3. 初始化策略、风控、broker
      4. 从账户同步真实持仓（实盘模式）
    """
    global _strategy_main, _strategy_macd, _strategy_boll
    global _risk_manager, _broker, _data_manager, _logger

    _logger = _setup_logging()

    signal_mode = getattr(config, 'SIGNAL_MODE', 'auto')
    mode_desc = {
        'notify': '仅提醒（不自动下单，红色弹窗提示）',
        'auto':   '自动下单',
    }.get(signal_mode, signal_mode)

    _logger.info("=" * 50)
    _logger.info("A股量化交易系统启动")
    _logger.info(f"运行模式: {'实盘' if config.TRADE_MODE == 'live' else '模拟盘'}")
    _logger.info(f"信号模式: {mode_desc}")
    _logger.info(f"监控标的: {', '.join(config.SYMBOLS_GM)}")
    _logger.info("=" * 50)

    # 设置掘金 Token
    set_token(config.GM_TOKEN)

    # 订阅30分钟K线（60根 = 足够计算 MA20）
    subscribe(
        symbols   = ",".join(config.SYMBOLS_GM),
        frequency = config.BAR_FREQUENCY,
        count     = config.BAR_COUNT,
    )

    # 初始化数据管理器
    _data_manager = DataManager(cache_dir=config.DATA_CACHE_DIR)

    # 初始化三个策略
    _strategy_main = MACrossStrategy(symbols=config.SYMBOLS_RAW)
    _strategy_macd = MACDStrategy(symbols=config.SYMBOLS_RAW)
    _strategy_boll = BollStrategy(symbols=config.SYMBOLS_RAW)

    # 初始化风控
    _risk_manager = RiskManager()

    # 初始化 broker
    if config.TRADE_MODE == "live" and GM_AVAILABLE:
        _broker = GmBroker(account_id=config.ACCOUNT_ID)
        _logger.info(f"[实盘] 使用掘金账户 {config.ACCOUNT_ID}")
    else:
        _broker = PaperBroker(initial_cash=config.INITIAL_CAPITAL)
        _logger.info(f"[模拟盘] 初始资金 {config.INITIAL_CAPITAL:,} 元")

    # 实盘模式：从账户同步真实持仓
    if config.TRADE_MODE == "live" and GM_AVAILABLE:
        _sync_positions_from_account()

    # 启动主策略（注入初始资金）
    cash = _broker.get_cash()
    _strategy_main.on_start(cash)
    _logger.info(f"策略初始化完成，可用资金 {cash:,} 元")


def on_bar(context, bars):
    """
    每根30分钟K线收盘后触发

    流程：
      1. 获取每只股票最近60根K线历史数据
      2. 计算技术指标
      3. 主策略生成信号 → 风控过滤 → 下单
      4. 观察策略生成信号 → 仅打印日志
    """
    global _strategy_main, _strategy_macd, _strategy_boll, _risk_manager, _broker

    if _strategy_main is None:
        _logger.error("策略未初始化，跳过本次 on_bar")
        return

    current_time = pd.Timestamp(context.now)
    _logger.info(f"--- on_bar {current_time} ---")

    # 1. 为每只股票构建 DataFrame（含技术指标）
    data = {}
    prices = {}

    for gm_sym in config.SYMBOLS_GM:
        sym = gm_sym.split(".")[-1]
        try:
            # 从掘金获取最近 BAR_COUNT 根K线
            raw_bars = history_n(
                symbol    = gm_sym,
                frequency = config.BAR_FREQUENCY,
                count     = config.BAR_COUNT,
                end_time  = context.now,
                fields    = "open,high,low,close,volume",
                adjust    = 1,   # 1=前复权
            )
            if raw_bars is None or len(raw_bars) == 0:
                _logger.warning(f"{gm_sym} 无K线数据，跳过")
                continue

            df = pd.DataFrame(raw_bars)
            df["eob"] = pd.to_datetime(df["eob"])
            df = df.set_index("eob").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].astype(float)

            # 计算技术指标
            df = _data_manager.build_from_bars(df)

            data[sym]   = df
            prices[sym] = float(df["close"].iloc[-1])

        except Exception as e:
            _logger.error(f"获取 {gm_sym} K线异常: {e}")

    if not data:
        _logger.warning("所有股票数据获取失败，跳过本次信号生成")
        return

    # 2. 更新策略持仓的当前市价
    _strategy_main.update_prices(prices)

    # 3. 更新风控峰值
    _risk_manager.update_peak(_strategy_main.total_assets)

    # 4. 主策略信号 → 风控过滤 → 下单
    try:
        main_signals = _strategy_main.generate_signals(data, current_time)
    except Exception as e:
        _logger.error(f"主策略异常: {e}")
        main_signals = []

    # 风控过滤（止损检查也在其中）
    if main_signals:
        main_signals = _risk_manager.filter_signals(
            main_signals, _strategy_main, prices
        )
    else:
        # 即使无新信号，也要检查止损
        stop_signals = _risk_manager.check_stop_loss(_strategy_main, prices)
        if stop_signals:
            _logger.info(f"风控止损触发 {len(stop_signals)} 个信号")
            main_signals = stop_signals

    # 执行信号
    signal_mode = getattr(config, 'SIGNAL_MODE', 'auto')

    for sig in main_signals:
        if sig.quantity <= 0:
            continue   # 防御：quantity=0 的信号不下单

        _logger.info(
            f"[主策略信号] {sig.action} {sig.symbol} "
            f"x{sig.quantity} | {sig.reason}"
        )

        # ---- notify 模式：只提醒，不下单 ----
        if signal_mode == "notify":
            notify_signal(sig, prices)
            continue

        # ---- auto 模式：自动下单 ----
        if sig.action == "BUY":
            result = _broker.buy(sig.symbol, sig.quantity, sig.price)
        elif sig.action == "SELL":
            result = _broker.sell(sig.symbol, sig.quantity, sig.price)
        else:
            continue

        if result.success:
            _logger.info(
                f"[下单成功] {result.action} {result.symbol} "
                f"x{result.filled_qty} @{result.filled_price:.2f} "
                f"手续费={result.commission:.2f}"
            )
            # 下单成功后也弹出提醒（auto 模式下作为确认通知）
            notify_signal(sig, prices)

            # 更新策略内部状态（模拟盘立即成交；实盘等 on_order_status 回报）
            if config.TRADE_MODE == "paper":
                sig.quantity = result.filled_qty
                _strategy_main.on_trade(
                    sig,
                    result.filled_price,
                    result.filled_qty,
                    result.commission,
                )
        else:
            _logger.warning(f"[下单失败] {result.symbol}: {result.message}")

    # 5. 观察策略 MACD → 仅打印日志
    try:
        macd_signals = _strategy_macd.generate_signals(data, current_time)
        for sig in macd_signals:
            _logger.info(f"[观察] {sig.reason}")
    except Exception as e:
        _logger.error(f"MACD 观察策略异常: {e}")

    # 6. 观察策略布林带 → 仅打印日志
    try:
        boll_signals = _strategy_boll.generate_signals(data, current_time)
        for sig in boll_signals:
            _logger.info(f"[观察] {sig.reason}")
    except Exception as e:
        _logger.error(f"布林带观察策略异常: {e}")

    # 7. 打印当前账户状态
    cash = _broker.get_cash()
    total = _strategy_main.total_assets
    _logger.info(
        f"账户状态: 现金={cash:,.0f} 总资产={total:,.0f} "
        f"持仓={list(_strategy_main.positions.keys())}"
    )


def on_order_status(context, order):
    """
    委托回报（掘金异步通知）

    实盘模式下在此更新策略内部持仓状态
    """
    if _logger is None:
        return

    status_map = {
        1: "已报",
        3: "已成交",
        5: "已撤",
        8: "已拒绝",
    }
    status_name = status_map.get(order.status, str(order.status))
    sym = order.symbol.split(".")[-1]

    _logger.info(
        f"[委托回报] {order.side_effect} {order.symbol} "
        f"x{order.volume} @{order.price:.2f} "
        f"成交={order.filled_volume} 状态={status_name}"
    )

    # 实盘成交后更新策略内部状态
    if config.TRADE_MODE == "live" and order.status == 3 and order.filled_volume > 0:
        from strategy.base import Signal
        side = "BUY" if order.side == 1 else "SELL"
        sig  = Signal(
            symbol   = sym,
            action   = side,
            quantity = order.filled_volume,
            price    = order.filled_price,
        )
        # 估算手续费（实盘以交易所结算为准，此处近似）
        amount = order.filled_price * order.filled_volume
        comm   = max(amount * config.COMMISSION, config.MIN_COMMISSION)
        if side == "SELL":
            comm += amount * config.STAMP_DUTY

        if _strategy_main:
            _strategy_main.on_trade(sig, order.filled_price, order.filled_volume, comm)
            _logger.info(
                f"[持仓更新] {side} {sym} x{order.filled_volume} "
                f"@{order.filled_price:.2f}"
            )


def on_error(context, code, info):
    """掘金错误回调"""
    if _logger:
        _logger.error(f"[掘金错误] code={code} info={info}")


# ----------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------

def _sync_positions_from_account():
    """从掘金真实账户同步持仓到策略"""
    try:
        cash_info = get_cash(account=config.ACCOUNT_ID)
        cash      = float(cash_info.available) if cash_info else 0.0

        positions = {}
        for gm_sym in config.SYMBOLS_GM:
            pos = get_position(account=config.ACCOUNT_ID, symbol=gm_sym)
            if pos and pos.volume > 0:
                sym = gm_sym.split(".")[-1]
                positions[sym] = {
                    "quantity":      float(pos.volume),
                    "avg_cost":      float(pos.cost_price),
                    "current_price": float(pos.price),
                }

        _strategy_main.sync_from_broker(cash, positions)
        _logger.info(f"[实盘] 持仓同步完成：现金={cash:,.0f}")

    except Exception as e:
        _logger.error(f"[实盘] 持仓同步失败: {e}")


# ----------------------------------------------------------------
# 入口
# ----------------------------------------------------------------

def main():
    if not GM_SDK_AVAILABLE:
        print("=" * 60)
        print("错误：掘金 SDK 未安装")
        print("安装命令：pip install gm")
        print("")
        print("如需运行回测，请使用：python main_backtest.py")
        print("=" * 60)
        sys.exit(1)

    if config.GM_TOKEN == "YOUR_GM_TOKEN_HERE":
        print("=" * 60)
        print("错误：请先在 config.py 中填写你的掘金 Token")
        print("获取方式：myquant.cn → 个人中心 → 系统设置 → Token")
        print("=" * 60)
        sys.exit(1)

    mode = MODE_LIVE   # 掘金 MODE_LIVE 对应实盘和模拟盘（由账户类型决定）

    sig_mode = getattr(config, 'SIGNAL_MODE', 'auto')
    sig_mode_cn = '仅提醒（不自动下单）' if sig_mode == 'notify' else '自动下单'

    print("=" * 60)
    print(f"  A股量化交易系统 - {'实盘' if config.TRADE_MODE == 'live' else '模拟盘'}模式")
    print(f"  信号模式: {sig_mode_cn}")
    print(f"  策略: 双均线 MA{config.MA_FAST}/MA{config.MA_SLOW}")
    print(f"  标的: {' + '.join(config.SYMBOLS_GM)}")
    print(f"  K线: {config.BAR_FREQUENCY}（每根K线触发一次策略）")
    print("=" * 60)
    print("正在连接掘金服务器...")

    run(
        strategy_id = config.STRATEGY_ID,
        mode        = mode,
        token       = config.GM_TOKEN,
        account     = config.ACCOUNT_ID,
    )


if __name__ == "__main__":
    main()
