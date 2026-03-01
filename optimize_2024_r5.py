"""
策略参数优化 R5 - 2024-2026 策略级改进
目标：总收益率 75%+

基于 R3 最优基础参数（63.8%），搜索三个新维度：
1. 动量过滤 - 跳过弱势股
2. 死叉确认 - 延迟卖出让趋势延续
3. 盈利保护 - 大盈利时不因死叉卖出

固定 R3 最优参数：
  MA10/20, MA_TREND=0, PARTIAL_PROFIT=False, MACD_BUY=True,
  COOLDOWN=3, POSITION_PCT=0.90, MAX_POSITION_PCT=0.35,
  STOP_LOSS_PCT=0.15, TRAILING_STOP_PCT=0.20, MAX_DRAWDOWN=0.50
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from backtest.engine import BacktestEngine
from risk.manager import RiskManager

# ---- 加载数据 ----
print("Loading data...")
dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
SYMBOLS = config.SYMBOLS_RAW

warmup_days = 120
load_start = (pd.to_datetime("20240101") - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
DATA = dm.get_multi(SYMBOLS, load_start, "20260227")
print(f"Loaded {len(DATA)} stocks")

START_DATE = "20240101"
END_DATE = "20260227"


def run_backtest(params: dict) -> dict:
    """运行单次回测"""
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


def grid_search(param_grid: dict, fixed_params: dict, desc: str = "") -> pd.DataFrame:
    """网格搜索（固定参数 + 搜索参数）"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  固定参数: {fixed_params}")
    print(f"  搜索参数: {', '.join(keys)}")
    print(f"  参数组合: {total}")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = {**fixed_params}
        for j, k in enumerate(keys):
            params[k] = combo[j]
        metrics = run_backtest(params)
        row = {}
        for k in keys:
            row[k] = params[k]
        row.update(metrics)
        results.append(row)

        if (i + 1) % 50 == 0 or i == total - 1:
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
# R3 最优基础参数（固定）
# ================================================================
R3_BEST = {
    "MA_FAST": 10,
    "MA_SLOW": 20,
    "MA_TREND": 0,
    "POSITION_PCT": 0.90,
    "MAX_POSITION_PCT": 0.35,
    "STOP_LOSS_PCT": 0.15,
    "TRAILING_STOP_PCT": 0.20,
    "MAX_DRAWDOWN_LIMIT": 0.50,
    "PARTIAL_PROFIT_ENABLED": False,
    "MACD_BUY_ENABLED": True,
    "SELL_COOLDOWN_BARS": 3,
    "MACD_SELL_RSI_MIN": 90,
    "RSI_OVERBOUGHT": 90,
}

# ================================================================
# Round 5A: 动量过滤搜索
# ================================================================
print("\n" + "="*60)
print("  ROUND 5A: 动量过滤搜索")
print("="*60)

r5a_grid = {
    "MOMENTUM_FILTER_ENABLED": [True, False],
    "MOMENTUM_LOOKBACK":       [5, 10, 15, 20, 30, 40, 60],
    "MOMENTUM_MIN_PCT":        [-0.15, -0.10, -0.05, 0.0, 0.02, 0.05, 0.10],
}
# 2 * 7 * 7 = 98 combos

r5a_df = grid_search(r5a_grid, R3_BEST, "Round 5A: 动量过滤")
os.makedirs("reports", exist_ok=True)
r5a_df.to_csv("reports/optimize_2024_r5a.csv", index=False, encoding="utf-8-sig")

print("\n  R5A Top 10:")
print(r5a_df.head(10).to_string())

# 提取 R5A 最优动量参数
best_5a = r5a_df.iloc[0]
r5a_best = {k: best_5a[k] for k in r5a_grid.keys()}
# 将 numpy 类型转为 python 原生类型
for k in r5a_best:
    v = r5a_best[k]
    if isinstance(v, (np.integer,)):
        r5a_best[k] = int(v)
    elif isinstance(v, (np.floating,)):
        r5a_best[k] = float(v)
    elif isinstance(v, (np.bool_,)):
        r5a_best[k] = bool(v)

print(f"\n  R5A Best: total_return={best_5a['total_return']*100:.1f}%, params={r5a_best}")


# ================================================================
# Round 5B: 死叉确认 + 盈利保护 搜索
# ================================================================
print("\n" + "="*60)
print("  ROUND 5B: 死叉确认 + 盈利保护")
print("="*60)

r5b_grid = {
    "DEATH_CROSS_CONFIRM":    [1, 2, 3],
    "PROFIT_PROTECT_ENABLED": [True, False],
    "PROFIT_PROTECT_PCT":     [0.05, 0.10, 0.15, 0.20, 0.30],
}
# 3 * 2 * 5 = 30 combos

r5b_fixed = {**R3_BEST, **r5a_best}
r5b_df = grid_search(r5b_grid, r5b_fixed, "Round 5B: 死叉确认 + 盈利保护")
r5b_df.to_csv("reports/optimize_2024_r5b.csv", index=False, encoding="utf-8-sig")

print("\n  R5B Top 10:")
print(r5b_df.head(10).to_string())

best_5b = r5b_df.iloc[0]
r5b_best = {k: best_5b[k] for k in r5b_grid.keys()}
for k in r5b_best:
    v = r5b_best[k]
    if isinstance(v, (np.integer,)):
        r5b_best[k] = int(v)
    elif isinstance(v, (np.floating,)):
        r5b_best[k] = float(v)
    elif isinstance(v, (np.bool_,)):
        r5b_best[k] = bool(v)

print(f"\n  R5B Best: total_return={best_5b['total_return']*100:.1f}%, params={r5b_best}")


# ================================================================
# Round 5C: 综合微调（核心参数 + 新参数联合搜索）
# ================================================================
print("\n" + "="*60)
print("  ROUND 5C: 综合联合搜索")
print("="*60)

# 固定确定的参数
r5c_fixed = {
    "MA_FAST": 10,
    "MA_SLOW": 20,
    "MA_TREND": 0,
    "PARTIAL_PROFIT_ENABLED": False,
    "MACD_BUY_ENABLED": True,
    "MAX_DRAWDOWN_LIMIT": 0.50,
    "RSI_OVERBOUGHT": 90,
}

# 联合搜索核心+新参数
r5c_grid = {
    "POSITION_PCT":           [0.85, 0.90, 0.95],
    "MAX_POSITION_PCT":       [0.30, 0.35, 0.40],
    "STOP_LOSS_PCT":          [0.15, 0.20, 0.30],
    "TRAILING_STOP_PCT":      [0.15, 0.20, 0.30],
    "SELL_COOLDOWN_BARS":     [2, 3, 5],
    "MACD_SELL_RSI_MIN":      [85, 90, 95],
    "MOMENTUM_FILTER_ENABLED": [r5a_best.get("MOMENTUM_FILTER_ENABLED", False)],
    "MOMENTUM_LOOKBACK":       [r5a_best.get("MOMENTUM_LOOKBACK", 20)],
    "MOMENTUM_MIN_PCT":        [r5a_best.get("MOMENTUM_MIN_PCT", 0.0)],
    "DEATH_CROSS_CONFIRM":     [r5b_best.get("DEATH_CROSS_CONFIRM", 1)],
    "PROFIT_PROTECT_ENABLED":  [r5b_best.get("PROFIT_PROTECT_ENABLED", False)],
    "PROFIT_PROTECT_PCT":      [r5b_best.get("PROFIT_PROTECT_PCT", 0.15)],
}
# 3*3*3*3*3*3 = 729 combos (new params fixed at best)

r5c_df = grid_search(r5c_grid, r5c_fixed, "Round 5C: 综合联合搜索")
r5c_df.to_csv("reports/optimize_2024_r5c.csv", index=False, encoding="utf-8-sig")

print("\n  R5C Top 20:")
display_cols = [c for c in r5c_grid.keys()] + ["total_return", "annual_return", "max_drawdown", "sharpe", "trades", "win_rate"]
display_cols = [c for c in display_cols if c in r5c_df.columns]
print(r5c_df.head(20)[display_cols].to_string())

# ================================================================
# Round 5D: 如果R5C仍未达75%，尝试更激进的VOL_CONFIRM + RSI组合
# ================================================================
best_5c = r5c_df.iloc[0]
if best_5c['total_return'] < 0.75:
    print("\n" + "="*60)
    print(f"  R5C best = {best_5c['total_return']*100:.1f}%, still < 75%")
    print("  ROUND 5D: 更激进的过滤参数搜索")
    print("="*60)

    # 收集 R5C 最优
    r5c_best = {}
    for k in r5c_grid.keys():
        v = best_5c[k]
        if isinstance(v, (np.integer,)):
            r5c_best[k] = int(v)
        elif isinstance(v, (np.floating,)):
            r5c_best[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            r5c_best[k] = bool(v)
        else:
            r5c_best[k] = v

    r5d_fixed = {**r5c_fixed, **r5c_best}

    r5d_grid = {
        "VOL_CONFIRM_RATIO":   [0.0, 0.3, 0.5, 0.8, 1.0],
        "RSI_BUY_MAX":         [70, 75, 80, 85, 100],
        "SURGE_MAX_PCT":       [0.0, 0.10, 0.15, 0.20, 0.30],
        "SELL_COOLDOWN_BARS":  [1, 2, 3, 5],
    }
    # 5*5*5*4 = 500 combos

    r5d_df = grid_search(r5d_grid, r5d_fixed, "Round 5D: 过滤参数微调")
    r5d_df.to_csv("reports/optimize_2024_r5d.csv", index=False, encoding="utf-8-sig")

    print("\n  R5D Top 20:")
    display_cols = [c for c in r5d_grid.keys()] + ["total_return", "annual_return", "max_drawdown", "sharpe", "trades", "win_rate"]
    display_cols = [c for c in display_cols if c in r5d_df.columns]
    print(r5d_df.head(20)[display_cols].to_string())

    best_5d = r5d_df.iloc[0]
    print(f"\n  R5D Best: total_return={best_5d['total_return']*100:.1f}%")


# ================================================================
# 最终汇总
# ================================================================
print("\n" + "="*60)
print("  最终结果汇总")
print("="*60)

# 收集所有轮次最优
all_bests = [
    ("R3 baseline", 0.6381),
    ("R5A 动量过滤", float(r5a_df.iloc[0]["total_return"])),
    ("R5B 死叉确认", float(r5b_df.iloc[0]["total_return"])),
    ("R5C 综合搜索", float(r5c_df.iloc[0]["total_return"])),
]
if 'r5d_df' in dir():
    all_bests.append(("R5D 过滤微调", float(r5d_df.iloc[0]["total_return"])))

for name, ret in all_bests:
    status = "OK" if ret >= 0.75 else "NEED MORE"
    print(f"  {name}: {ret*100:.1f}% [{status}]")

print("="*60)
