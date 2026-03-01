"""
信号监控入口 - 不需要掘金 SDK，用 AKShare 定时轮询

使用方式：
    python main_signal.py

功能：
  - 启动时盘前区间分析：每只股票判断处于买入/卖出/中性区间，弹窗+彩色提醒
  - 盘中每 15 分钟获取最新行情，运行策略生成买卖信号
  - 有信号时红色/绿色醒目提醒 + 声音 + 系统弹窗
  - 不自动下单，需要自己在交易软件中手动操作
  - 交易时间外自动休眠，盘前自动唤醒
  - 不需要掘金账号，不需要任何额外配置

配置（在 config.py 中）：
  - SYMBOLS_RAW   : 监控的股票代码列表
  - MA_FAST / MA_SLOW / MA_TREND : 策略参数
  - 其他策略和风控参数

无需修改：
  - TRADE_MODE / SIGNAL_MODE / GM_TOKEN 等实盘配置与本脚本无关
"""
import sys
import os
import time
import logging
from datetime import datetime, timedelta

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
from data.manager import DataManager
from data.provider import AKShareProvider
from strategy.ma_cross import MACrossStrategy
from strategy.macd import MACDStrategy
from strategy.boll import BollStrategy
from risk.manager import RiskManager
from notifier import notify_signal, _console_alert, notify_zone_analysis

# ----------------------------------------------------------------
# 常量
# ----------------------------------------------------------------
POLL_INTERVAL = 15 * 60   # 轮询间隔（秒）：15 分钟
PRE_MARKET_WAKE = 5       # 盘前提前唤醒（分钟）


# ----------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        config.LOG_DIR,
        f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("main_signal")


# ----------------------------------------------------------------
# A 股交易时间判断
# ----------------------------------------------------------------
def is_trading_time(now: datetime = None) -> bool:
    """判断当前是否为 A 股交易时间（周一至周五 9:30-11:30, 13:00-15:00）"""
    if now is None:
        now = datetime.now()

    # 周末不交易
    if now.weekday() >= 5:
        return False

    t = now.hour * 100 + now.minute
    # 上午 9:30 - 11:30
    if 930 <= t <= 1130:
        return True
    # 下午 13:00 - 15:00
    if 1300 <= t <= 1500:
        return True

    return False


def seconds_until_next_session(now: datetime = None) -> int:
    """计算距离下一个交易时段的秒数"""
    if now is None:
        now = datetime.now()

    t = now.hour * 100 + now.minute

    # 今天还有交易时段的情况
    if now.weekday() < 5:
        if t < 925:
            # 今天盘前：到 9:25 的秒数
            target = now.replace(hour=9, minute=25, second=0, microsecond=0)
            return max(int((target - now).total_seconds()), 0)
        if 1130 < t < 1255:
            # 午休：到 12:55 的秒数
            target = now.replace(hour=12, minute=55, second=0, microsecond=0)
            return max(int((target - now).total_seconds()), 0)

    # 今天已收盘或周末：找下一个工作日 9:25
    next_day = now + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    target = next_day.replace(hour=9, minute=25, second=0, microsecond=0)
    return max(int((target - now).total_seconds()), 0)


# ----------------------------------------------------------------
# 数据获取
# ----------------------------------------------------------------
def load_history_data(dm: DataManager, symbols: list, logger) -> dict:
    """
    加载历史数据（用于计算技术指标）。

    优先读本地缓存，没有缓存时联网拉取一次并缓存。
    后续运行中只靠实时价格更新，不再重复拉取历史数据。
    """
    from pathlib import Path
    cache_dir = Path(config.DATA_CACHE_DIR)

    # 回测区间：往前推 250 天，确保 MA60 等有足够数据
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")

    data = {}
    for sym in symbols:
        df = None

        # 1) 尝试读本地缓存
        candidates = sorted(cache_dir.glob(f"{sym}_*.pkl"), key=lambda f: f.stat().st_mtime)
        if candidates:
            pkl_file = candidates[-1]
            try:
                raw = pd.read_pickle(pkl_file)
                keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
                df = raw[keep].copy()
                logger.info(
                    f"  {config.SYMBOL_NAMES.get(sym, sym)}({sym}): "
                    f"缓存 {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}"
                )
            except Exception as e:
                logger.warning(f"  {sym} 缓存读取失败: {e}")

        # 2) 没有缓存 → 联网下载一次（下载后会自动缓存到 pkl）
        if df is None or len(df) < 60:
            name = config.SYMBOL_NAMES.get(sym, sym)
            logger.info(f"  {name}({sym}): 首次运行，正在下载历史数据（仅此一次）...")
            try:
                df = dm.get(sym, start_date, end_date, adjust="qfq", use_cache=True)
                if df is not None and len(df) > 60:
                    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                    df = df[keep].copy()
                    logger.info(
                        f"  {name}({sym}): "
                        f"下载成功 {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}"
                    )
                else:
                    logger.error(f"  {sym} 下载数据不足，跳过")
                    continue
            except Exception as e:
                logger.error(
                    f"  {sym} 下载失败: {e}\n"
                    f"    提示: 东方财富可能限流了，等几分钟再试，或先运行 main_backtest.py 缓存数据"
                )
                continue

        # 3) 计算技术指标
        df = dm.build_from_bars(df)
        data[sym] = df

    return data


def update_with_realtime(
    data: dict,
    provider: AKShareProvider,
    symbols: list,
    logger,
) -> dict:
    """
    获取实时价格，追加到历史数据末尾（如果今日还没有数据）。
    返回更新后的 data 字典和 prices 字典。
    """
    prices = provider.get_realtime_prices(symbols)

    for sym in symbols:
        price = prices.get(sym, 0)
        if price <= 0 or sym not in data:
            continue

        df = data[sym]
        today = pd.Timestamp(datetime.now().date())

        # 如果今日数据已存在（缓存中已有），更新收盘价
        if today in df.index:
            df.loc[today, "close"] = price
        else:
            # 追加今日行作为最新数据点
            new_row = pd.DataFrame({
                "open": [price],
                "high": [price],
                "low": [price],
                "close": [price],
                "volume": [0],
            }, index=[today])
            df = pd.concat([df, new_row])
            data[sym] = df

    return data, prices


# ----------------------------------------------------------------
# 盘前区间分析
# ----------------------------------------------------------------
def analyze_stock_zones(data: dict, dm: DataManager, logger) -> list:
    """
    分析每只股票当前处于买入区间还是卖出区间。

    判断维度：
      1. 均线排列：MA_FAST vs MA_SLOW vs MA_TREND
      2. MACD 状态：柱正/负、趋势方向
      3. RSI 水平：超买/超卖/中性
      4. 价格与趋势线的关系
      5. 布林带位置

    Returns:
        list of dict，每个元素包含分析结果
    """
    results = []

    for sym, df in data.items():
        if len(df) < 2:
            continue

        name = config.SYMBOL_NAMES.get(sym, sym)

        # 重新计算指标确保最新
        df = dm.build_from_bars(df)

        curr = df.iloc[-1]
        prev = df.iloc[-1 - 1] if len(df) >= 2 else curr
        price = curr["close"]

        details = []
        buy_score = 0    # 正分偏买入，负分偏卖出
        total_weight = 0

        # ---- 1. 均线排列 (权重 3) ----
        fast_col = f"ma{config.MA_FAST}"
        slow_col = f"ma{config.MA_SLOW}"
        trend_col = f"ma{config.MA_TREND}"

        ma_fast = curr.get(fast_col, None)
        ma_slow = curr.get(slow_col, None)
        ma_trend = curr.get(trend_col, None)

        if ma_fast is not None and ma_slow is not None and not pd.isna(ma_fast) and not pd.isna(ma_slow):
            if ma_fast > ma_slow:
                buy_score += 3
                ma_status = f"MA{config.MA_FAST}({ma_fast:.2f}) > MA{config.MA_SLOW}({ma_slow:.2f}) 多头排列"
            else:
                buy_score -= 3
                ma_status = f"MA{config.MA_FAST}({ma_fast:.2f}) < MA{config.MA_SLOW}({ma_slow:.2f}) 空头排列"
            total_weight += 3

            # 金叉/死叉临近判断
            pf = prev.get(fast_col, None)
            ps = prev.get(slow_col, None)
            if pf is not None and ps is not None and not pd.isna(pf) and not pd.isna(ps):
                gap_now = ma_fast - ma_slow
                gap_prev = pf - ps
                if gap_prev <= 0 and gap_now > 0:
                    details.append(f"  ** 刚刚金叉！MA{config.MA_FAST} 上穿 MA{config.MA_SLOW}")
                    buy_score += 2
                elif gap_prev >= 0 and gap_now < 0:
                    details.append(f"  ** 刚刚死叉！MA{config.MA_FAST} 下穿 MA{config.MA_SLOW}")
                    buy_score -= 2
                elif gap_now > 0 and gap_now < gap_prev:
                    details.append(f"  ! 多头收窄，注意死叉风险")
                elif gap_now < 0 and abs(gap_now) < abs(gap_prev):
                    details.append(f"  ! 空头收窄，关注金叉机会")

            details.append(f"  均线: {ma_status}")

        # ---- 2. 趋势线 (权重 2) ----
        if ma_trend is not None and not pd.isna(ma_trend):
            if price > ma_trend:
                buy_score += 2
                details.append(f"  趋势: 价格({price:.2f}) 在 MA{config.MA_TREND}({ma_trend:.2f}) 之上 ↑")
            else:
                buy_score -= 2
                pct_below = (ma_trend - price) / ma_trend * 100
                details.append(f"  趋势: 价格({price:.2f}) 在 MA{config.MA_TREND}({ma_trend:.2f}) 之下 ↓ ({pct_below:.1f}%)")
            total_weight += 2

        # ---- 3. MACD 状态 (权重 2) ----
        macd_bar = curr.get("macd_bar", None)
        macd_dif = curr.get("macd_dif", None)
        macd_dea = curr.get("macd_dea", None)
        prev_bar = prev.get("macd_bar", None)

        if macd_bar is not None and not pd.isna(macd_bar):
            if macd_bar > 0:
                buy_score += 2
                bar_trend = ""
                if prev_bar is not None and not pd.isna(prev_bar):
                    if macd_bar > prev_bar:
                        bar_trend = "，柱放大 ↑"
                    else:
                        bar_trend = "，柱缩小 ↓"
                details.append(f"  MACD: 柱={macd_bar:.4f} 红柱(多头){bar_trend}")
            else:
                buy_score -= 2
                bar_trend = ""
                if prev_bar is not None and not pd.isna(prev_bar):
                    if macd_bar < prev_bar:
                        bar_trend = "，柱放大 ↓"
                    else:
                        bar_trend = "，柱缩小 ↑"
                details.append(f"  MACD: 柱={macd_bar:.4f} 绿柱(空头){bar_trend}")

            # MACD 金叉/死叉
            if prev_bar is not None and not pd.isna(prev_bar):
                if prev_bar <= 0 and macd_bar > 0:
                    details.append(f"  ** MACD 刚刚金叉！")
                    buy_score += 1
                elif prev_bar > 0 and macd_bar <= 0:
                    details.append(f"  ** MACD 刚刚死叉！")
                    buy_score -= 1
            total_weight += 2

        # ---- 4. RSI (权重 2) ----
        rsi = curr.get("rsi", None)
        if rsi is not None and not pd.isna(rsi):
            if rsi < 30:
                buy_score += 2
                details.append(f"  RSI: {rsi:.1f} 超卖区（反弹机会）")
            elif rsi < 40:
                buy_score += 1
                details.append(f"  RSI: {rsi:.1f} 偏弱")
            elif rsi > config.RSI_OVERBOUGHT:
                buy_score -= 2
                details.append(f"  RSI: {rsi:.1f} 超买区（回调风险）")
            elif rsi > 70:
                buy_score -= 1
                details.append(f"  RSI: {rsi:.1f} 偏强")
            else:
                details.append(f"  RSI: {rsi:.1f} 中性")
            total_weight += 2

        # ---- 5. 布林带位置 (权重 1) ----
        boll_upper = curr.get("boll_upper", None)
        boll_lower = curr.get("boll_lower", None)
        boll_mid = curr.get("boll_mid", None)
        if boll_upper is not None and boll_lower is not None and not pd.isna(boll_upper):
            boll_width = boll_upper - boll_lower
            if boll_width > 0:
                boll_pos = (price - boll_lower) / boll_width  # 0~1
                if boll_pos > 0.95:
                    buy_score -= 1
                    details.append(f"  布林: 触及上轨({boll_upper:.2f})，回调概率大")
                elif boll_pos < 0.05:
                    buy_score += 1
                    details.append(f"  布林: 触及下轨({boll_lower:.2f})，反弹概率大")
                elif boll_pos > 0.8:
                    details.append(f"  布林: 偏上轨区域 ({boll_pos*100:.0f}%)")
                elif boll_pos < 0.2:
                    details.append(f"  布林: 偏下轨区域 ({boll_pos*100:.0f}%)")
                else:
                    details.append(f"  布林: 中轨附近 ({boll_pos*100:.0f}%)")
            total_weight += 1

        # ---- 综合判定 ----
        if total_weight > 0:
            score_pct = buy_score / total_weight
        else:
            score_pct = 0

        if score_pct >= 0.4:
            zone = "BUY"
            zone_cn = "处于买入区间 — 多头趋势，可关注买入机会"
        elif score_pct <= -0.4:
            zone = "SELL"
            zone_cn = "处于卖出区间 — 空头趋势，注意风险或止盈"
        else:
            zone = "NEUTRAL"
            zone_cn = "中性观望 — 趋势不明朗，等待方向确认"

        results.append({
            "symbol": sym,
            "name": name,
            "zone": zone,
            "zone_cn": zone_cn,
            "price": price,
            "score": buy_score,
            "max_score": total_weight,
            "details": details,
        })

    return results


# ----------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------
def main():
    logger = _setup_logging()

    symbols = config.SYMBOLS_RAW
    sym_names = [f"{config.SYMBOL_NAMES.get(s, s)}({s})" for s in symbols]

    print("=" * 60)
    print("  A股量化信号监控")
    print(f"  模式  : 仅提醒（不自动下单）")
    print(f"  策略  : 双均线 MA{config.MA_FAST}/MA{config.MA_SLOW}")
    print(f"  标的  : {', '.join(sym_names)}")
    print(f"  频率  : 每 {POLL_INTERVAL // 60} 分钟检查一次")
    print("=" * 60)

    # 初始化组件
    dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
    provider = AKShareProvider()
    strategy = MACrossStrategy(symbols=symbols)
    strategy_macd = MACDStrategy(symbols=symbols)
    strategy_boll = BollStrategy(symbols=symbols)
    risk_manager = RiskManager()

    # 启动策略（注入初始资金用于仓位计算）
    strategy.on_start(config.INITIAL_CAPITAL)

    # 加载历史数据
    logger.info("正在加载历史数据...")
    data = load_history_data(dm, symbols, logger)

    if not data:
        logger.error("无法加载任何历史数据")
        print("\n" + "=" * 60)
        print("  错误：没有可用的历史数据")
        print("=" * 60)
        print("  可能原因：首次运行且东方财富限流了")
        print()
        print("  解决方法（任选一种）：")
        print("  1. 等 2~3 分钟后重新运行（限流会自动恢复）")
        print("  2. 关闭 VPN/代理后重试")
        print("  3. 切换到手机热点后重试")
        print("  4. 先运行 python main_backtest.py 缓存数据")
        print("=" * 60)
        sys.exit(1)

    logger.info(f"历史数据加载完成，监控 {len(data)} 只股票")

    # ---- 盘前区间分析（启动时立即执行） ----
    logger.info("正在进行盘前区间分析...")
    try:
        zone_results = analyze_stock_zones(data, dm, logger)
        if zone_results:
            notify_zone_analysis(zone_results)
            for r in zone_results:
                logger.info(
                    f"[区间分析] {r['name']}({r['symbol']}): "
                    f"{r['zone']} | {r['zone_cn']} | "
                    f"得分={r['score']}/{r['max_score']}"
                )
    except Exception as e:
        logger.error(f"盘前区间分析异常: {e}", exc_info=True)

    logger.info("开始信号监控循环（Ctrl+C 退出）...")

    check_count = 0

    while True:
        try:
            now = datetime.now()

            # ---- 非交易时间：休眠 ----
            if not is_trading_time(now):
                wait = seconds_until_next_session(now)
                if wait > 60:
                    hours = wait // 3600
                    mins = (wait % 3600) // 60
                    logger.info(
                        f"非交易时间，休眠至下一交易时段"
                        f"（约 {hours}小时{mins}分钟）"
                    )
                    # 分段休眠，每 5 分钟检查一次（方便 Ctrl+C 退出）
                    slept = 0
                    while slept < wait:
                        chunk = min(300, wait - slept)
                        time.sleep(chunk)
                        slept += chunk
                    continue
                else:
                    time.sleep(30)
                    continue

            # ---- 交易时间：获取实时行情 + 跑策略 ----
            check_count += 1
            logger.info(f"--- 第 {check_count} 次检查 ({now.strftime('%H:%M:%S')}) ---")

            # 获取实时价格并更新数据
            try:
                data, prices = update_with_realtime(data, provider, symbols, logger)
            except Exception as e:
                logger.warning(f"获取实时行情失败: {e}，跳过本次")
                time.sleep(POLL_INTERVAL)
                continue

            if not prices or all(p <= 0 for p in prices.values()):
                logger.warning("所有股票实时价格获取失败，跳过本次")
                time.sleep(POLL_INTERVAL)
                continue

            # 打印当前价格
            for sym in symbols:
                p = prices.get(sym, 0)
                if p > 0:
                    name = config.SYMBOL_NAMES.get(sym, sym)
                    logger.info(f"  {name}({sym}): {p:.2f}")

            # 重新计算技术指标
            data_with_indicators = {}
            for sym, df in data.items():
                data_with_indicators[sym] = dm.build_from_bars(df)

            # 更新策略持仓市价
            strategy.update_prices(prices)

            # 更新风控峰值
            risk_manager.update_peak(strategy.total_assets)

            # 生成主策略信号
            current_time = pd.Timestamp(now)
            try:
                main_signals = strategy.generate_signals(
                    data_with_indicators, current_time
                )
            except Exception as e:
                logger.error(f"策略异常: {e}")
                main_signals = []

            # 风控过滤
            if main_signals:
                main_signals = risk_manager.filter_signals(
                    main_signals, strategy, prices
                )
            else:
                stop_signals = risk_manager.check_stop_loss(strategy, prices)
                if stop_signals:
                    logger.info(f"风控止损触发 {len(stop_signals)} 个信号")
                    main_signals = stop_signals

            # 处理信号：弹出提醒
            for sig in main_signals:
                if sig.quantity <= 0:
                    continue
                logger.info(
                    f"[信号] {sig.action} {sig.symbol} "
                    f"x{sig.quantity} | {sig.reason}"
                )
                notify_signal(sig, prices)

            # 观察策略日志
            try:
                for sig in strategy_macd.generate_signals(
                    data_with_indicators, current_time
                ):
                    logger.info(f"[MACD观察] {sig.reason}")
            except Exception:
                pass

            try:
                for sig in strategy_boll.generate_signals(
                    data_with_indicators, current_time
                ):
                    logger.info(f"[布林观察] {sig.reason}")
            except Exception:
                pass

            # 当前状态
            total = strategy.total_assets
            logger.info(
                f"策略状态: 资金={strategy.cash:,.0f} "
                f"总资产={total:,.0f} "
                f"持仓={list(strategy.positions.keys()) or '空仓'}"
            )

            if not main_signals:
                logger.info("本次无交易信号")

            # ---- 等待下一次轮询 ----
            logger.info(f"下次检查: {POLL_INTERVAL // 60} 分钟后")
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n用户中断，信号监控已停止")
            print("\n信号监控已停止。")
            break
        except Exception as e:
            logger.error(f"主循环异常: {e}", exc_info=True)
            time.sleep(60)  # 出错后等 1 分钟再重试


if __name__ == "__main__":
    main()
