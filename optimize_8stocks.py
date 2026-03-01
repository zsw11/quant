"""
8只大白马策略参数优化 - 第一轮

核心搜索维度：
  1. 仓位分配 POSITION_PCT (0.30~0.90) — 8只股票需要分散
  2. 单票上限 MAX_POSITION_PCT (0.15~0.50) — 控制集中度
  3. MA 周期组合 (fast/slow/trend)
  4. 止损/回撤参数

使用方式：
    py optimize_8stocks.py
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

logging.basicConfig(level=logging.WARNING)  # 优化时只输出警告以上


def run_backtest_with_params(data, params):
    """用指定参数跑一次回测，返回结果指标"""
    # 临时覆盖 config
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
            "final_equity": result.final_value,
        }
    except Exception as e:
        return {
            "total_return": -999, "annual_return": -999,
            "max_drawdown": 99, "sharpe": -999, "calmar": -999,
            "win_rate": 0, "trades": 0, "final_equity": 0,
            "error": str(e),
        }
    finally:
        for key, val in original.items():
            setattr(config, key, val)


def main():
    print("=" * 70)
    print("  8只大白马策略参数优化 - 第一轮")
    print("=" * 70)

    # 加载数据
    print("\n加载数据中...")
    dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
    warmup_days = max(config.MA_TREND, 60) * 2
    load_start = (
        pd.to_datetime(config.BACKTEST_START) - pd.Timedelta(days=warmup_days)
    ).strftime("%Y%m%d")
    data = dm.get_multi(
        symbols=config.SYMBOLS_RAW,
        start_date=load_start,
        end_date=config.BACKTEST_END,
    )
    print(f"加载完成：{len(data)} 只股票\n")

    # ========== 搜索空间 ==========
    # 第一轮重点：仓位分配 + MA周期 + 风控
    param_grid = {
        "POSITION_PCT":      [0.30, 0.45, 0.60, 0.80],
        "MAX_POSITION_PCT":  [0.20, 0.30, 0.40],
        "MA_FAST":           [5, 7, 10],
        "MA_SLOW":           [15, 20, 30],
        "MA_TREND":          [0, 30, 40, 60],
        "MAX_DRAWDOWN_LIMIT":[0.15, 0.20, 0.25],
        "STOP_LOSS_PCT":     [0.05, 0.08],
        "TRAILING_STOP_PCT": [0.06, 0.08, 0.10],
    }

    # 生成所有组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"搜索空间：{total} 种参数组合")
    print(f"参数维度：{', '.join(keys)}")
    print()

    # 如果组合太多，采样
    MAX_RUNS = 2000
    if total > MAX_RUNS:
        import random
        random.seed(42)
        combos = random.sample(combos, MAX_RUNS)
        print(f"组合过多，随机采样 {MAX_RUNS} 组")
        total = MAX_RUNS

    results = []
    best_annual = -999
    best_params = None
    start_time = datetime.now()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # 跳过不合理的组合
        if params["MA_FAST"] >= params["MA_SLOW"]:
            continue
        if params["MA_TREND"] > 0 and params["MA_TREND"] <= params["MA_SLOW"]:
            continue

        r = run_backtest_with_params(data, params)
        r.update(params)
        results.append(r)

        # 进度输出
        if r.get("annual_return", -999) > best_annual:
            best_annual = r["annual_return"]
            best_params = params
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
                f"交易={r['trades']}次 "
                f"| POS={params['POSITION_PCT']} "
                f"MAX_POS={params['MAX_POSITION_PCT']} "
                f"MA={params['MA_FAST']}/{params['MA_SLOW']}/{params['MA_TREND']} "
                f"DD={params['MAX_DRAWDOWN_LIMIT']} "
                f"SL={params['STOP_LOSS_PCT']} "
                f"TS={params['TRAILING_STOP_PCT']} "
                f"(ETA {eta:.0f}s)"
            )
        elif (i + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"[{i+1:4d}/{total}] 进度 {(i+1)/total*100:.1f}% (ETA {eta:.0f}s)")

    # ========== 汇总结果 ==========
    df = pd.DataFrame(results)
    df = df.sort_values("annual_return", ascending=False)

    print("\n" + "=" * 70)
    print("  TOP 20 参数组合（按年化收益排序）")
    print("=" * 70)
    display_cols = [
        "annual_return", "total_return", "max_drawdown", "sharpe", "calmar",
        "win_rate", "trades",
    ] + keys
    top20 = df.head(20)
    for idx, row in top20.iterrows():
        print(
            f"  年化={row['annual_return']:+6.2f}% "
            f"总收={row['total_return']:+7.2f}% "
            f"回撤={row['max_drawdown']:5.1f}% "
            f"夏普={row['sharpe']:6.3f} "
            f"胜率={row['win_rate']:5.1f}% "
            f"交易={int(row['trades']):3d} "
            f"| POS={row['POSITION_PCT']} "
            f"MAX={row['MAX_POSITION_PCT']} "
            f"MA={int(row['MA_FAST'])}/{int(row['MA_SLOW'])}/{int(row['MA_TREND'])} "
            f"DD={row['MAX_DRAWDOWN_LIMIT']} "
            f"SL={row['STOP_LOSS_PCT']} "
            f"TS={row['TRAILING_STOP_PCT']}"
        )

    # 按夏普排序的 TOP 10
    print("\n" + "=" * 70)
    print("  TOP 10 参数组合（按夏普比率排序）")
    print("=" * 70)
    top_sharpe = df.sort_values("sharpe", ascending=False).head(10)
    for idx, row in top_sharpe.iterrows():
        print(
            f"  夏普={row['sharpe']:6.3f} "
            f"年化={row['annual_return']:+6.2f}% "
            f"回撤={row['max_drawdown']:5.1f}% "
            f"卡玛={row['calmar']:6.3f} "
            f"| POS={row['POSITION_PCT']} "
            f"MAX={row['MAX_POSITION_PCT']} "
            f"MA={int(row['MA_FAST'])}/{int(row['MA_SLOW'])}/{int(row['MA_TREND'])} "
            f"DD={row['MAX_DRAWDOWN_LIMIT']} "
            f"SL={row['STOP_LOSS_PCT']} "
            f"TS={row['TRAILING_STOP_PCT']}"
        )

    # 按卡玛比率排序（收益/回撤）
    print("\n" + "=" * 70)
    print("  TOP 10 参数组合（按卡玛比率排序，收益/风险最优）")
    print("=" * 70)
    top_calmar = df[df['annual_return'] > 0].sort_values("calmar", ascending=False).head(10)
    for idx, row in top_calmar.iterrows():
        print(
            f"  卡玛={row['calmar']:6.3f} "
            f"年化={row['annual_return']:+6.2f}% "
            f"回撤={row['max_drawdown']:5.1f}% "
            f"夏普={row['sharpe']:6.3f} "
            f"| POS={row['POSITION_PCT']} "
            f"MAX={row['MAX_POSITION_PCT']} "
            f"MA={int(row['MA_FAST'])}/{int(row['MA_SLOW'])}/{int(row['MA_TREND'])} "
            f"DD={row['MAX_DRAWDOWN_LIMIT']} "
            f"SL={row['STOP_LOSS_PCT']} "
            f"TS={row['TRAILING_STOP_PCT']}"
        )

    # 保存完整结果
    csv_path = "reports/optimize_8stocks_r1.csv"
    os.makedirs("reports", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存: {csv_path}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"优化完成，耗时 {elapsed:.0f} 秒，共 {len(results)} 次回测")


if __name__ == "__main__":
    main()
