"""
8只大白马策略参数优化 - 第二轮

锁定第一轮最优基础参数，细化搜索：
  - 止盈档位 PROFIT_TARGET_1/2/3
  - RSI 参数 RSI_OVERBOUGHT / MACD_SELL_RSI_MIN
  - MACD 买入开关
  - 冷却期 SELL_COOLDOWN_BARS
  - 在 MA10/20/30 和 MA10/20/0 两种趋势配置上搜索

使用方式：
    py optimize_8stocks_r2.py
"""
import sys
import os
import itertools
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from backtest.engine import BacktestEngine
from risk.manager import RiskManager

logging.basicConfig(level=logging.WARNING)


def run_backtest_with_params(data, params):
    original = {}
    for key, val in params.items():
        original[key] = getattr(config, key, None)
        setattr(config, key, val)

    try:
        strategy = MACrossStrategy(symbols=config.SYMBOLS_RAW)
        risk_manager = RiskManager(initial_capital=config.INITIAL_CAPITAL)
        engine = BacktestEngine()
        result = engine.run(
            strategy=strategy,
            data=data,
            start_date=config.BACKTEST_START,
            end_date=config.BACKTEST_END,
            risk_manager=risk_manager,
        )
        return {
            "total_return": result.total_return * 100,
            "annual_return": result.annual_return * 100,
            "max_drawdown": result.max_drawdown * 100,
            "sharpe": result.sharpe_ratio,
            "calmar": result.calmar_ratio,
            "win_rate": result.win_rate * 100,
            "trades": result.total_trades,
        }
    except Exception as e:
        return {
            "total_return": -999, "annual_return": -999,
            "max_drawdown": 99, "sharpe": -999, "calmar": -999,
            "win_rate": 0, "trades": 0, "error": str(e),
        }
    finally:
        for key, val in original.items():
            setattr(config, key, val)


def main():
    print("=" * 70)
    print("  8只大白马策略参数优化 - 第二轮（止盈/RSI/MACD细化）")
    print("=" * 70)

    # 加载数据
    print("\n加载数据中...")
    dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
    warmup_days = 120
    load_start = (
        pd.to_datetime(config.BACKTEST_START) - pd.Timedelta(days=warmup_days)
    ).strftime("%Y%m%d")
    data = dm.get_multi(
        symbols=config.SYMBOLS_RAW,
        start_date=load_start,
        end_date=config.BACKTEST_END,
    )
    print(f"加载完成：{len(data)} 只股票\n")

    # ========== 锁定第一轮最优基础参数 ==========
    base_params = {
        "MA_FAST": 10,
        "MA_SLOW": 20,
        "POSITION_PCT": 0.60,
        "MAX_POSITION_PCT": 0.40,
        "MAX_TOTAL_POS_PCT": 0.95,
        "MIN_CASH_RESERVE": 0.05,
    }

    # ========== 第二轮搜索空间 ==========
    param_grid = {
        "MA_TREND":           [0, 30],       # 两种趋势配置
        "STOP_LOSS_PCT":      [0.05, 0.07, 0.08],
        "TRAILING_STOP_PCT":  [0.08, 0.10, 0.12],
        "MAX_DRAWDOWN_LIMIT": [0.15, 0.20, 0.25],
        "PROFIT_TARGET_1":    [0.04, 0.06, 0.08],
        "PROFIT_TARGET_2":    [0.10, 0.12, 0.15],
        "PROFIT_TARGET_3":    [0.18, 0.22, 0.25],
        "RSI_OVERBOUGHT":     [80, 85, 90],
        "MACD_SELL_RSI_MIN":  [55, 65, 75],
        "MACD_BUY_ENABLED":   [True, False],
        "SELL_COOLDOWN_BARS": [2, 3, 5],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"搜索空间：{total} 种参数组合")

    MAX_RUNS = 2000
    if total > MAX_RUNS:
        import random
        random.seed(42)
        combos = random.sample(combos, MAX_RUNS)
        total = MAX_RUNS
        print(f"组合过多，随机采样 {MAX_RUNS} 组")

    results = []
    best_annual = -999
    start_time = datetime.now()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params.update(base_params)

        # 止盈档位必须递增
        if params["PROFIT_TARGET_1"] >= params["PROFIT_TARGET_2"]:
            continue
        if params["PROFIT_TARGET_2"] >= params["PROFIT_TARGET_3"]:
            continue

        r = run_backtest_with_params(data, params)
        r.update(params)
        results.append(r)

        if r.get("annual_return", -999) > best_annual:
            best_annual = r["annual_return"]
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(
                f"[{i+1:4d}/{total}] 新最优! "
                f"年化={r['annual_return']:+.2f}% "
                f"总收益={r['total_return']:+.2f}% "
                f"回撤={r['max_drawdown']:.1f}% "
                f"夏普={r['sharpe']:.3f} "
                f"胜率={r['win_rate']:.1f}% "
                f"| T={params['MA_TREND']} "
                f"SL={params['STOP_LOSS_PCT']} "
                f"TS={params['TRAILING_STOP_PCT']} "
                f"DD={params['MAX_DRAWDOWN_LIMIT']} "
                f"PT={params['PROFIT_TARGET_1']}/{params['PROFIT_TARGET_2']}/{params['PROFIT_TARGET_3']} "
                f"OB={params['RSI_OVERBOUGHT']} "
                f"MACD_RSI={params['MACD_SELL_RSI_MIN']} "
                f"MACD_BUY={params['MACD_BUY_ENABLED']} "
                f"CD={params['SELL_COOLDOWN_BARS']} "
                f"(ETA {eta:.0f}s)"
            )
        elif (i + 1) % 200 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"[{i+1:4d}/{total}] 进度 {(i+1)/total*100:.1f}% (ETA {eta:.0f}s)")

    # ========== 汇总 ==========
    df = pd.DataFrame(results)
    df = df.sort_values("annual_return", ascending=False)

    print("\n" + "=" * 70)
    print("  TOP 15 参数组合（按年化收益排序）")
    print("=" * 70)
    for _, row in df.head(15).iterrows():
        print(
            f"  年化={row['annual_return']:+6.2f}% "
            f"总收={row['total_return']:+7.2f}% "
            f"回撤={row['max_drawdown']:5.1f}% "
            f"夏普={row['sharpe']:6.3f} "
            f"胜率={row['win_rate']:5.1f}% "
            f"交易={int(row['trades']):3d} "
            f"| T={int(row['MA_TREND'])} "
            f"SL={row['STOP_LOSS_PCT']} "
            f"TS={row['TRAILING_STOP_PCT']} "
            f"DD={row['MAX_DRAWDOWN_LIMIT']} "
            f"PT={row['PROFIT_TARGET_1']}/{row['PROFIT_TARGET_2']}/{row['PROFIT_TARGET_3']} "
            f"OB={int(row['RSI_OVERBOUGHT'])} "
            f"MACD_RSI={int(row['MACD_SELL_RSI_MIN'])} "
            f"MACD_BUY={row['MACD_BUY_ENABLED']} "
            f"CD={int(row['SELL_COOLDOWN_BARS'])}"
        )

    # 卡玛TOP10
    print("\n" + "=" * 70)
    print("  TOP 10（按卡玛比率，收益/风险最优）")
    print("=" * 70)
    top_calmar = df[df['annual_return'] > 0].sort_values("calmar", ascending=False).head(10)
    for _, row in top_calmar.iterrows():
        print(
            f"  卡玛={row['calmar']:6.3f} "
            f"年化={row['annual_return']:+6.2f}% "
            f"回撤={row['max_drawdown']:5.1f}% "
            f"夏普={row['sharpe']:6.3f} "
            f"| T={int(row['MA_TREND'])} "
            f"SL={row['STOP_LOSS_PCT']} "
            f"TS={row['TRAILING_STOP_PCT']} "
            f"DD={row['MAX_DRAWDOWN_LIMIT']} "
            f"PT={row['PROFIT_TARGET_1']}/{row['PROFIT_TARGET_2']}/{row['PROFIT_TARGET_3']} "
            f"OB={int(row['RSI_OVERBOUGHT'])} "
            f"MACD_BUY={row['MACD_BUY_ENABLED']} "
            f"CD={int(row['SELL_COOLDOWN_BARS'])}"
        )

    csv_path = "reports/optimize_8stocks_r2.csv"
    os.makedirs("reports", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存: {csv_path}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"优化完成，耗时 {elapsed:.0f} 秒，共 {len(results)} 次回测")


if __name__ == "__main__":
    main()
