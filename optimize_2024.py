"""
策略参数优化 - 2024-2026 专项优化
目标：总收益率 75%+

分析发现的问题：
1. 止盈/止损过紧，在大趋势中过早出场
2. 移动止盈8%太紧，波动市中频繁触发
3. 分级止盈过早锁定利润，截断了大行情
4. 隆基绿能/茅台是拖累，但不剔除（保持8股池不变）

优化策略：
- Round 1: MA周期 + 仓位 + 止损/止盈宽度
- Round 2: 基于R1最优，细化MACD/RSI/移动止盈参数
"""
import sys, os, time, itertools, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from backtest.engine import BacktestEngine
from risk.manager import RiskManager

# ---- 加载数据 ----
print("Loading data...")
dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
SYMBOLS = config.SYMBOLS_RAW

# 预热窗口
warmup_days = 120
load_start = (pd.to_datetime("20240101") - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
DATA = dm.get_multi(SYMBOLS, load_start, "20260227")
print(f"Loaded {len(DATA)} stocks")

START_DATE = "20240101"
END_DATE = "20260227"


def run_backtest(params: dict) -> dict:
    """运行单次回测，返回绩效指标"""
    # 临时覆盖 config
    originals = {}
    for k, v in params.items():
        originals[k] = getattr(config, k, None)
        setattr(config, k, v)

    try:
        strategy = MACrossStrategy(symbols=SYMBOLS)
        risk_mgr = RiskManager(initial_capital=config.INITIAL_CAPITAL)
        engine = BacktestEngine()
        result = engine.run(
            strategy=strategy,
            data=DATA,
            start_date=START_DATE,
            end_date=END_DATE,
            risk_manager=risk_mgr,
        )
        return {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe_ratio,
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "final_value": result.final_value,
        }
    except Exception as e:
        return {"total_return": -999, "annual_return": -999, "error": str(e)}
    finally:
        for k, v in originals.items():
            if v is not None:
                setattr(config, k, v)
            else:
                try:
                    delattr(config, k)
                except:
                    pass


def grid_search(param_grid: dict, desc: str = "") -> pd.DataFrame:
    """网格搜索"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  参数组合: {total}")
    print(f"  搜索维度: {', '.join(keys)}")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        metrics = run_backtest(params)
        row = {**params, **metrics}
        results.append(row)

        if (i + 1) % 100 == 0 or i == total - 1:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            best_so_far = max(results, key=lambda x: x.get("total_return", -999))
            print(
                f"  [{i+1}/{total}] "
                f"{elapsed:.0f}s, ETA {eta:.0f}s | "
                f"best={best_so_far['total_return']*100:.1f}%"
            )

    df = pd.DataFrame(results)
    df = df.sort_values("total_return", ascending=False).reset_index(drop=True)
    return df


# ================================================================
# Round 1: MA + 仓位 + 止盈止损核心参数
# ================================================================
print("\n" + "="*60)
print("  ROUND 1: 核心参数搜索")
print("="*60)

r1_grid = {
    "MA_FAST":             [5, 7, 10],
    "MA_SLOW":             [15, 20, 30],
    "MA_TREND":            [0, 30, 60],           # 0=关闭趋势过滤
    "POSITION_PCT":        [0.50, 0.70, 0.90],
    "MAX_POSITION_PCT":    [0.35, 0.50],
    "STOP_LOSS_PCT":       [0.10, 0.20],           # 放宽止损
    "TRAILING_STOP_PCT":   [0.15, 0.25, 0.50],    # 大幅放宽移动止盈
    "MAX_DRAWDOWN_LIMIT":  [0.30, 0.50],           # 放宽回撤容忍
    "PARTIAL_PROFIT_ENABLED": [True, False],        # 测试关闭分级止盈
}
# 3*3*3*3*2*2*3*2*2 = 3888 combos, ~26 min

r1_df = grid_search(r1_grid, "Round 1: MA + 仓位 + 止损止盈")

# 保存结果
os.makedirs("reports", exist_ok=True)
r1_df.to_csv("reports/optimize_2024_r1.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print("  Round 1 Top 20:")
print("="*60)
cols = list(r1_grid.keys()) + ["total_return", "annual_return", "max_drawdown", "sharpe", "trades", "win_rate"]
display_cols = [c for c in cols if c in r1_df.columns]
print(r1_df.head(20)[display_cols].to_string())

# 提取 R1 最优参数
best_r1 = r1_df.iloc[0]
r1_best_params = {k: best_r1[k] for k in r1_grid.keys()}
print(f"\nR1 Best: total_return={best_r1['total_return']*100:.1f}%, params={r1_best_params}")


# ================================================================
# Round 2: 基于 R1 最优，搜索止盈档位 + RSI + MACD 参数
# ================================================================
print("\n" + "="*60)
print("  ROUND 2: 止盈/RSI/MACD 细化")
print("="*60)

# 固定 R1 最优参数
r2_fixed = {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in r1_best_params.items()}

r2_grid = {
    "PROFIT_TARGET_1":     [0.10, 0.15, 0.25],
    "PROFIT_TARGET_2":     [0.30, 0.50],
    "PROFIT_TARGET_3":     [0.60, 1.00, 2.00],   # 非常宽的第三档
    "RSI_OVERBOUGHT":      [88, 95],
    "MACD_SELL_RSI_MIN":   [75, 85, 95],
    "MACD_BUY_ENABLED":    [True, False],
    "SELL_COOLDOWN_BARS":  [1, 3, 5],
}
# 3*2*3*2*3*2*3 = 648 combos, ~4.5 min

# 合并固定参数
r2_combos_keys = list(r2_grid.keys())
r2_combos_values = list(r2_grid.values())
r2_all = list(itertools.product(*r2_combos_values))
total_r2 = len(r2_all)

print(f"  R2 固定参数: {r2_fixed}")
print(f"  R2 搜索组合: {total_r2}")

r2_results = []
t0 = time.time()

for i, combo in enumerate(r2_all):
    params = {**r2_fixed}
    for j, k in enumerate(r2_combos_keys):
        params[k] = combo[j]
    metrics = run_backtest(params)
    row = {**params, **metrics}
    r2_results.append(row)

    if (i + 1) % 100 == 0 or i == total_r2 - 1:
        elapsed = time.time() - t0
        speed = (i + 1) / elapsed
        eta = (total_r2 - i - 1) / speed if speed > 0 else 0
        best_so_far = max(r2_results, key=lambda x: x.get("total_return", -999))
        print(
            f"  [{i+1}/{total_r2}] "
            f"{elapsed:.0f}s, ETA {eta:.0f}s | "
            f"best={best_so_far['total_return']*100:.1f}%"
        )

r2_df = pd.DataFrame(r2_results).sort_values("total_return", ascending=False).reset_index(drop=True)
r2_df.to_csv("reports/optimize_2024_r2.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print("  Round 2 Top 20:")
print("="*60)
r2_display = list(r2_grid.keys()) + ["total_return", "annual_return", "max_drawdown", "sharpe", "trades", "win_rate"]
r2_display = [c for c in r2_display if c in r2_df.columns]
print(r2_df.head(20)[r2_display].to_string())

# ---- 最终最优参数 ----
best_r2 = r2_df.iloc[0]
final_params = {}
for k in list(r1_grid.keys()) + list(r2_grid.keys()):
    if k in best_r2:
        final_params[k] = best_r2[k]

print("\n" + "="*60)
print("  最终最优参数")
print("="*60)
for k, v in final_params.items():
    print(f"  {k} = {v}")
print(f"\n  总收益率: {best_r2['total_return']*100:.2f}%")
print(f"  年化收益率: {best_r2['annual_return']*100:.2f}%")
print(f"  最大回撤: {best_r2['max_drawdown']*100:.2f}%")
print(f"  夏普比率: {best_r2['sharpe']:.3f}")
print(f"  交易次数: {best_r2['trades']}")
print(f"  胜率: {best_r2['win_rate']*100:.1f}%")
print("="*60)
